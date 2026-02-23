"""
LLM API interface: prompt construction, HTTP calls, and robust JSON parsing.

The LLM receives a compact JSON payload describing the current latent
state (last latent, velocity, acceleration) and optionally a kNN prior,
and returns a ``predicted_delta`` vector in normalised PCA space.
"""

import json
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests

from . import config as cfg

# ── JSON serialisation helpers ───────────────────────────────

def _json_safe(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.astype(float).tolist()
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    return x


# ── Prompt construction ──────────────────────────────────────

def build_llm_prompt(
    latents_norm_hist_top: np.ndarray,
    dt_h: float,
    rollout_nonce: Optional[float] = None,
    meta_row: Optional[Dict] = None,
    delta_prior: Optional[np.ndarray] = None,
    prior_std: Optional[np.ndarray] = None,
    residual_norm_cap: Optional[float] = None,
    residual_comp_cap: Optional[float] = None,
) -> str:
    """
    Build the user-message prompt for the LLM.

    The prompt embeds the numeric context as a single JSON blob and
    instructs the model to return ``{"predicted_delta": [...]}``.
    """
    T_obs, D = latents_norm_hist_top.shape
    last = latents_norm_hist_top[-1].astype(np.float32)
    v = (last - latents_norm_hist_top[-2]) if T_obs >= 2 else np.zeros(D, dtype=np.float32)
    a = np.zeros(D, dtype=np.float32)
    if T_obs >= 3:
        v_prev = latents_norm_hist_top[-2] - latents_norm_hist_top[-3]
        a = v - v_prev

    obj: Dict[str, Any] = {
        "D": int(D),
        "dt_hours": float(dt_h),
        "last_latent": last,
        "velocity": v,
        "acceleration": a,
        "rollout_nonce": float(rollout_nonce) if rollout_nonce is not None else 0.0,
    }

    if delta_prior is not None:
        obj["delta_prior"] = np.asarray(delta_prior, dtype=np.float32)
    if prior_std is not None:
        obj["prior_std"] = np.asarray(prior_std, dtype=np.float32)
    if residual_norm_cap is not None:
        obj["residual_norm_cap"] = float(residual_norm_cap)
    if residual_comp_cap is not None:
        obj["residual_comp_cap"] = float(residual_comp_cap)
    if meta_row is not None:
        obj["metadata_context"] = meta_row

    obj = _json_safe(obj)

    prompt = (
        f"Output ONLY JSON with exactly one key predicted_delta.\n"
        f"predicted_delta must be a list of exactly D={int(D)} floats.\n"
        f"Interpret predicted_delta as a RESIDUAL to add to delta_prior.\n"
        f"Keep it small: L2(residual) <= residual_norm_cap and abs(component) <= residual_comp_cap.\n"
        f"No extra keys, text, or markdown.\n"
        f"INPUT={json.dumps(obj, separators=(',', ':'))}"
    )
    return prompt


# ── Response parsing ─────────────────────────────────────────

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _salvage_predicted_delta_from_text(text: str) -> Optional[list]:
    """Extract floats from a possibly truncated JSON response."""
    if "predicted_delta" not in text:
        return None
    i = text.find("predicted_delta")
    j = text.find("[", i)
    if j < 0:
        return None
    tail = text[j + 1 :]
    nums = _FLOAT_RE.findall(tail)
    if not nums:
        return None
    try:
        return [float(x) for x in nums]
    except Exception:
        return None


def parse_llm_delta(
    text: str,
    D: int,
    allow_length_repair: bool = False,
    allow_text_salvage: bool = False,
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Parse the LLM response into a delta vector of length *D*.

    Strategy:
        1. Strict ``json.loads``
        2. Regex extraction of embedded ``{...}``
        3. (optional) float salvage from truncated text
        4. (optional) trim / zero-pad if length mismatches
    """
    info: Dict[str, Any] = {
        "returned_len": None,
        "length_repaired": False,
        "salvaged_from_text": False,
        "parse_ok": False,
    }

    # Try JSON direct
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        pass

    # Try embedded JSON blob
    if obj is None:
        try:
            m = re.search(r"\{.*\}", text.replace("\n", ""), re.DOTALL)
            if m:
                obj = json.loads(m.group())
        except Exception:
            pass

    if isinstance(obj, dict) and "predicted_delta" in obj:
        try:
            vec = np.array(obj["predicted_delta"], dtype=np.float32).reshape(-1)
            info["returned_len"] = int(vec.size)
            if np.all(np.isfinite(vec)):
                info["parse_ok"] = True
                if vec.size != D:
                    if not allow_length_repair:
                        return None, info
                    info["length_repaired"] = True
                    vec = vec[:D] if vec.size > D else np.pad(vec, (0, D - vec.size))
                return vec, info
        except Exception:
            pass

    # Optional salvage
    if allow_text_salvage:
        salv = _salvage_predicted_delta_from_text(text)
        if salv is not None:
            vec = np.array(salv, dtype=np.float32).reshape(-1)
            info["salvaged_from_text"] = True
            info["returned_len"] = int(vec.size)
            info["parse_ok"] = True
            if not np.all(np.isfinite(vec)):
                return None, info
            if vec.size != D:
                if not allow_length_repair:
                    return None, info
                info["length_repaired"] = True
                vec = vec[:D] if vec.size > D else np.pad(vec, (0, D - vec.size))
            return vec, info

    return None, info


# ── LLM call with retries ───────────────────────────────────

def call_llm_delta(
    latents_norm_hist_top: np.ndarray,
    dt_h: float,
    rollout_nonce: Optional[float] = None,
    meta_row: Optional[Dict] = None,
    horizon: int = 1,
    delta_prior: Optional[np.ndarray] = None,
    prior_std: Optional[np.ndarray] = None,
    residual_norm_cap: Optional[float] = None,
    residual_comp_cap: Optional[float] = None,
) -> Tuple[np.ndarray, str, int, Dict]:
    """
    Call the LLM and return the parsed delta vector.

    Returns ``(delta_vec, raw_text, http_status, info_dict)``.
    Raises ``RuntimeError`` on unrecoverable failure.
    """
    if not cfg.OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Export it as an environment variable before running."
        )

    D = latents_norm_hist_top.shape[1]
    base_prompt = build_llm_prompt(
        latents_norm_hist_top,
        dt_h,
        rollout_nonce=rollout_nonce,
        meta_row=meta_row,
        delta_prior=delta_prior,
        prior_std=prior_std,
        residual_norm_cap=residual_norm_cap,
        residual_comp_cap=residual_comp_cap,
    )

    headers = {
        "Authorization": f"Bearer {cfg.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://example.com",
        "X-Title": "Corrosion-Research",
    }

    temp_eff = float(cfg.LLM_TEMPERATURE) * float(
        cfg.HORIZON_TEMP_DECAY ** max(0, horizon - 1)
    )

    def _post(messages):
        payload = {
            "model": cfg.LLM_MODEL,
            "messages": messages,
            "temperature": float(temp_eff),
            "max_tokens": int(cfg.LLM_MAX_TOKENS),
        }
        if cfg.USE_RESPONSE_FORMAT_JSON:
            payload["response_format"] = {"type": "json_object"}
        return requests.post(
            cfg.CHAT_ENDPOINT, headers=headers, json=payload, timeout=cfg.LLM_TIMEOUT
        )

    messages = [
        {"role": "system", "content": "Return ONLY JSON. No prose. No markdown."},
        {"role": "user", "content": base_prompt},
    ]

    last_raw = None
    last_info: Dict[str, Any] = {
        "returned_len": None,
        "length_repaired": False,
        "retries_used": 0,
        "salvaged_from_text": False,
        "parse_ok": False,
    }

    for attempt in range(cfg.LLM_MAX_RETRIES + 1):
        resp = _post(messages)
        last_status = resp.status_code
        if last_status != 200:
            raise RuntimeError(f"LLM HTTP {last_status}: {resp.text[:800]}")

        last_raw = resp.json()["choices"][0]["message"]["content"]

        # Strict parse first
        delta, info = parse_llm_delta(last_raw, D)
        last_info.update(info)
        last_info["retries_used"] = attempt
        if delta is not None:
            return delta, last_raw, last_status, last_info

        wrong_len = info.get("returned_len") is not None and info["returned_len"] != D
        parse_failed = not info.get("parse_ok", False)

        if wrong_len and cfg.LLM_RETRY_ON_LENGTH_MISMATCH and attempt < cfg.LLM_MAX_RETRIES:
            repair_msg = (
                f"predicted_delta length was {info['returned_len']}, must be exactly D={D}. "
                f"Return ONLY JSON: {{\"predicted_delta\":[...]}}, {D} floats."
            )
            messages += [
                {"role": "assistant", "content": last_raw},
                {"role": "user", "content": repair_msg},
            ]
            continue

        if parse_failed and cfg.LLM_RETRY_ON_PARSE_FAILURE and attempt < cfg.LLM_MAX_RETRIES:
            repair_msg = (
                f"Output was not valid JSON. Return ONLY valid JSON with "
                f"predicted_delta containing exactly {D} floats."
            )
            messages += [
                {"role": "assistant", "content": last_raw},
                {"role": "user", "content": repair_msg},
            ]
            continue

        # Last-resort salvage
        if cfg.LLM_ALLOW_TEXT_SALVAGE or cfg.LLM_ALLOW_LENGTH_REPAIR:
            delta2, info2 = parse_llm_delta(
                last_raw, D,
                allow_length_repair=cfg.LLM_ALLOW_LENGTH_REPAIR,
                allow_text_salvage=cfg.LLM_ALLOW_TEXT_SALVAGE,
            )
            info2["retries_used"] = attempt
            if delta2 is not None:
                return delta2, last_raw, last_status, info2

        raise RuntimeError(
            f"LLM returned invalid output after retries. "
            f"returned_len={info.get('returned_len')} expected={D}. "
            f"Raw: {last_raw[:600]}"
        )

    # Should not reach here, but just in case
    raise RuntimeError("LLM call exhausted retries without returning a valid delta.")
