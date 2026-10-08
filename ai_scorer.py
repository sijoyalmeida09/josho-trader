"""
JOSHO TRADER — AI Confidence Scorer

Wraps Groq's free `llama-3.3-70b-versatile` model to score F&O / equity
signals on a 0-100 confidence scale with a short human-readable reason.

Contract (see _CONTRACT.md "Contract: AI scorer"):
    AIScorer().score(signal: dict) -> {"confidence": int 0..100, "reason": str}

Fail-soft behavior:
  - Missing GROQ_API_KEY            → deterministic rule-based fallback
  - Groq SDK unavailable             → rule-based fallback
  - API/network/parse error          → rule-based fallback, log WARNING
  - Cache hit (<= cache_ttl_sec)     → same answer returned without API call

In-memory cache only. Never raises. Bot keeps scanning even if Groq is down.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger("josho.ai_scorer")

# ─────────────────────────────────────────────────────────────
# Optional Groq SDK import (fail-soft)
# ─────────────────────────────────────────────────────────────
try:
    from groq import Groq  # type: ignore
    _GROQ_AVAILABLE = True
except ImportError:  # pragma: no cover — covered by smoke test when SDK absent
    Groq = None  # type: ignore
    _GROQ_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# System prompt (baked-in trader context)
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an Indian-market F&O trading analyst. The trader (Sijoy) prefers "
    "deep-OTM far-month options as concentration lottery bets — his largest win "
    "was TATASTEEL +580%. He values fast moves, high OI/volume, and clean "
    "momentum or oversold bounces. He distrusts overcrowded names.\n"
    "Given the signal below, return a JSON object exactly like:\n"
    '{"confidence": <0-100>, "reason": "<<= 25 words, plain text>"}\n'
    "Confidence > 80 only when (a) score>=70 AND volume+OI rising AND no "
    "adverse situational. 50-70 is the typical band. < 40 means skip.\n"
    "Reply with JSON only, no markdown."
)


# ─────────────────────────────────────────────────────────────
# Rule-based fallback
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RuleBasedScorer:
    """Deterministic confidence scorer used when Groq is unavailable."""

    @staticmethod
    def score(signal: dict) -> dict:
        confidence = 40
        factors: list[str] = []

        strategy = (signal.get("strategy") or "").upper()
        score = float(signal.get("score") or 0)
        volume = float(signal.get("volume") or 0)
        oi = float(signal.get("oi") or 0)
        otm_pct = float(signal.get("otm_pct") or 0)
        cost_inr = float(signal.get("cost_inr") or signal.get("cost") or 0)
        active_situational = signal.get("active_situational") or []

        if strategy in ("OVERSOLD", "MOMENTUM") and score >= 60:
            confidence += 15
            factors.append(f"{strategy.lower()} score>=60 (+15)")

        if (volume + oi) > 100_000:
            confidence += 10
            factors.append("volume+oi>100k (+10)")

        if 3 <= otm_pct <= 6:
            confidence += 5
            factors.append("otm in sweet spot 3-6% (+5)")

        if cost_inr > 2500:
            confidence -= 10
            factors.append("cost>2500 capital risk (-10)")

        # Adverse situational signals: anything truthy and non-empty counts as active
        if any(_is_active_adverse(s) for s in active_situational):
            confidence -= 20
            factors.append("active adverse situational (-20)")

        confidence = max(0, min(100, confidence))

        dominant = factors[0] if factors else "no strong factor"
        reason = f"rule-based fallback (no Groq key); {dominant}"

        return {"confidence": int(confidence), "reason": reason}


def _is_active_adverse(situational: Any) -> bool:
    """Heuristic — situational is adverse if it's a dict with negative bias or any truthy non-empty value."""
    if not situational:
        return False
    if isinstance(situational, dict):
        bias = (situational.get("bias") or situational.get("direction") or "").lower()
        if bias in ("negative", "bearish", "down", "adverse"):
            return True
        # Any score field < 0 is also adverse
        s = situational.get("score")
        if isinstance(s, (int, float)) and s < 0:
            return True
        # Default: presence of any active situational entry is treated as adverse
        return True
    return True


# ─────────────────────────────────────────────────────────────
# AI Scorer
# ─────────────────────────────────────────────────────────────
class AIScorer:
    """Groq-backed confidence scorer with in-memory TTL cache and rule-based fallback."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        cache_ttl_sec: int = 300,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.cache_ttl_sec = max(1, int(cache_ttl_sec))
        self._cache: dict[tuple[str, int], dict] = {}

        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "").strip()
        self._client = None
        self.enabled = False

        if not self._api_key:
            log.info("AIScorer: GROQ_API_KEY missing — using rule-based fallback.")
            return
        if not _GROQ_AVAILABLE:
            log.warning("AIScorer: groq SDK not installed — using rule-based fallback.")
            return

        try:
            self._client = Groq(api_key=self._api_key)
            self.enabled = True
            log.info("AIScorer: Groq enabled (model=%s, cache_ttl=%ds).", self.model, self.cache_ttl_sec)
        except Exception as e:  # pragma: no cover — network/SDK init failures
            log.warning("AIScorer: failed to init Groq client (%s) — using fallback.", e)
            self._client = None
            self.enabled = False

    # ── public ──────────────────────────────────────────────
    def score(self, signal: dict) -> dict:
        """Return {"confidence": 0..100, "reason": str}. Never raises."""
        if not isinstance(signal, dict):
            return RuleBasedScorer.score({})

        # Cache lookup — keyed by trading symbol + 5-min bucket
        symbol = self._extract_symbol(signal)
        bucket = math.floor(time.time() / self.cache_ttl_sec)
        cache_key = (symbol, bucket)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return dict(cached)  # defensive copy

        if not self.enabled or self._client is None:
            result = RuleBasedScorer.score(signal)
        else:
            result = self._call_groq(signal)
            if result is None:
                result = RuleBasedScorer.score(signal)

        # Layer-4 advisory adjustment (active situational signals only, cap ±30)
        try:
            from trader_db import TraderDB  # type: ignore
            sym = signal.get("symbol") or signal.get("trading_symbol") or signal.get("underlying")
            if sym:
                db = TraderDB()
                active = [s for s in db.fetch_active_signals_for_symbol(sym) if s.get("status") == "active"]
                adj = 0.0
                for sig in active:
                    direction_map = sig.get("direction") or {}
                    d = direction_map.get(sym, 0)
                    mag = float(sig.get("magnitude", 0) or 0)
                    conf = float(sig.get("confidence", 0) or 0)
                    adj += d * mag * (conf / 100.0) * 30.0   # cap weight 0.3 → ±30 pts max contribution per signal
                adj = max(-30.0, min(30.0, adj))
                result["confidence"] = max(0, min(100, int(round(result["confidence"] + adj))))
                if adj != 0:
                    result["reason"] = f"{result['reason']} [+SA{adj:+.0f}]"
        except Exception:
            pass  # failsoft — never let Layer-4 break ai_scorer

        self._cache[cache_key] = result
        self._evict_stale(bucket)
        return dict(result)

    # ── internals ───────────────────────────────────────────
    @staticmethod
    def _extract_symbol(signal: dict) -> str:
        return str(
            signal.get("trading_symbol")
            or signal.get("symbol")
            or signal.get("underlying")
            or "UNKNOWN"
        )

    def _call_groq(self, signal: dict) -> Optional[dict]:
        try:
            payload = json.dumps(signal, default=str, separators=(",", ":"))
        except (TypeError, ValueError) as e:
            log.warning("AIScorer: signal not JSON-serializable (%s) — falling back.", e)
            return None

        user_msg = f"SIGNAL:\n{payload}"

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.2,
            "max_tokens": 200,
        }

        # Try with JSON response_format first; some Groq builds support it.
        for attempt_kwargs in (
            {**kwargs, "response_format": {"type": "json_object"}},
            kwargs,
        ):
            try:
                completion = self._client.chat.completions.create(**attempt_kwargs)  # type: ignore[union-attr]
                raw = completion.choices[0].message.content or ""
                parsed = self._parse_response(raw)
                if parsed is not None:
                    return parsed
                log.warning("AIScorer: failed to parse Groq response: %r", raw[:200])
                return None
            except TypeError:
                # response_format param not supported — retry without it
                continue
            except Exception as e:
                log.warning("AIScorer: Groq call failed (%s) — falling back.", e)
                return None
        return None

    @staticmethod
    def _parse_response(raw: str) -> Optional[dict]:
        if not raw:
            return None
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.strip("`")
            # Drop language tag like "json\n..."
            if "\n" in text:
                text = text.split("\n", 1)[1]
        # Extract first JSON object if model added prose
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            text = text[start : end + 1]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        confidence = data.get("confidence")
        reason = data.get("reason")
        if not isinstance(confidence, (int, float)):
            return None
        if not isinstance(reason, str):
            reason = str(reason) if reason is not None else ""

        confidence_int = max(0, min(100, int(round(float(confidence)))))
        return {"confidence": confidence_int, "reason": reason.strip()[:240]}

    def _evict_stale(self, current_bucket: int) -> None:
        """Drop cache entries older than 4 buckets to bound memory."""
        if len(self._cache) < 256:
            return
        cutoff = current_bucket - 4
        for key in list(self._cache.keys()):
            if key[1] < cutoff:
                self._cache.pop(key, None)


# ─────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    fake_signal = {
        "strategy": "FNO_DEEP_OTM",
        "symbol": "COALINDIA26JUN420CE",
        "underlying": "COALINDIA",
        "premium": 2.15,
        "lot": 1350,
        "cost_inr": 2903,
        "otm_pct": 4.2,
        "volume": 18540,
        "oi": 92100,
        "underlying_change_pct": -1.8,
        "score": 72,
        "active_situational": [],
    }

    scorer = AIScorer()
    result = scorer.score(fake_signal)
    print(
        f"SMOKE OK enabled={scorer.enabled} "
        f"confidence={result['confidence']} reason={result['reason']!r}"
    )
