"""
ML Signal Model V2 — Fixed infrastructure + WorldQuant alphas + advanced features.

Fixes from research:
  1. Binary classification (UP/DOWN) instead of 3-class
  2. Purged train/test split (no label leakage)
  3. Feature selection via importance pruning
  4. Class imbalance handling via sample weights
  5. WorldQuant alpha factors (042, 101, 004, 020, 041)
  6. Parkinson/Garman-Klass volatility + vol regime
  7. Kyle's lambda + Amihud illiquidity (microstructure)
  8. Cyclical time encoding (day/month as sin/cos)
  9. Stronger regularization to prevent overfitting
"""

import logging
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.indicators import (
    rsi, macd, adx, bollinger_bands, atr, ema, sma,
    stochastic, cci, roc, obv, mfi, volume_surge,
    supertrend, vwap, momentum_score,
)

log = logging.getLogger("josho.ml.signal_v2")

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Enhanced Feature Engineering ──────────────────────────────────

def engineer_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    V2: 60 curated features (pruned from 80+, added alpha factors).
    Focused on signal, not noise.
    """
    f = pd.DataFrame(index=df.index)

    # ── Price Action (8) ──────────────────────────────────────
    f["returns_1d"] = df["close"].pct_change(1)
    f["returns_5d"] = df["close"].pct_change(5)
    f["returns_10d"] = df["close"].pct_change(10)
    f["log_return"] = np.log(df["close"] / df["close"].shift(1))
    f["hl_range"] = (df["high"] - df["low"]) / df["close"]
    f["oc_range"] = (df["close"] - df["open"]) / df["open"]
    f["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    f["close_vs_range"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)

    # ── Trend (8) ─────────────────────────────────────────────
    f["ema_9_21_cross"] = (ema(df, 9) - ema(df, 21)) / df["close"]
    f["sma_20_50_cross"] = (sma(df, 20) - sma(df, 50)) / df["close"]
    f["sma_200_dist"] = (df["close"] - sma(df, 200)) / sma(df, 200)
    f["price_vs_vwap"] = (df["close"] - vwap(df)) / df["close"]
    f["sma_slope_20"] = sma(df, 20).pct_change(5)
    st = supertrend(df)
    f["supertrend_dir"] = st["direction"]
    f["adx_14"] = adx(df, 14)
    f["adx_trend"] = (f["adx_14"] > 25).astype(int)

    # ── Momentum (6) ──────────────────────────────────────────
    f["rsi_14"] = rsi(df, 14)
    m = macd(df)
    f["macd_histogram"] = m["histogram"]
    f["macd_hist_slope"] = m["histogram"].diff()
    stoch = stochastic(df)
    f["stoch_k"] = stoch["k"]
    f["cci_20"] = cci(df, 20)
    f["roc_12"] = roc(df, 12)

    # ── Volatility (8) — UPGRADED with Parkinson + Garman-Klass ──
    f["atr_14_norm"] = atr(df, 14) / df["close"]
    f["atr_ratio"] = atr(df, 7) / (atr(df, 14) + 1e-8)
    bb = bollinger_bands(df)
    f["bb_pct"] = bb["pct"]
    f["bb_width"] = bb["width"]
    f["bb_squeeze"] = (bb["width"] < bb["width"].rolling(20).quantile(0.1)).astype(int)

    # Parkinson volatility (more efficient than close-to-close)
    f["parkinson_vol"] = np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(df["high"] / df["low"]) ** 2)
    ).rolling(20).mean()

    # Garman-Klass volatility (uses all OHLC)
    f["gk_vol"] = np.sqrt(
        (0.5 * np.log(df["high"] / df["low"]) ** 2 -
         (2 * np.log(2) - 1) * np.log(df["close"] / df["open"]) ** 2).clip(lower=0)
    ).rolling(20).mean()

    # Vol regime: short vs long term
    f["vol_regime"] = f["parkinson_vol"].rolling(5).mean() / (f["parkinson_vol"].rolling(60).mean() + 1e-8)

    # ── Volume (5) ────────────────────────────────────────────
    f["volume_surge"] = volume_surge(df, 20)
    f["obv_slope"] = obv(df).pct_change(5)
    f["mfi_14"] = mfi(df, 14)
    f["volume_price_confirm"] = (
        (f["returns_1d"] > 0) & (f["volume_surge"] > 1.2)
    ).astype(int) - (
        (f["returns_1d"] < 0) & (f["volume_surge"] > 1.2)
    ).astype(int)
    f["volume_acceleration"] = df["volume"].pct_change().diff()

    # ── WorldQuant Alpha Factors (5) — NEW ────────────────────
    # Alpha 042: VWAP deviation (normalized)
    vwap_val = vwap(df)
    f["wq_alpha042"] = (vwap_val - df["close"]) / (vwap_val + df["close"] + 1e-8)

    # Alpha 101: Intraday range-normalized movement
    f["wq_alpha101"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-8)

    # Alpha 004: Mean-reversion in lows
    f["wq_alpha004"] = -1 * df["low"].rolling(9).apply(
        lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
    )

    # Alpha 041: Geometric mean vs VWAP
    f["wq_alpha041"] = np.sqrt(df["high"] * df["low"]) - vwap_val

    # Alpha 020: Overnight gap multi-factor
    f["wq_alpha020"] = (
        -1 * (df["open"] / (df["high"].shift(1) + 1e-8)).rank(pct=True) *
        (df["open"] / (df["close"].shift(1) + 1e-8)).rank(pct=True)
    )

    # ── Microstructure (3) — NEW ──────────────────────────────
    # Kyle's Lambda proxy (price impact per volume)
    price_change = df["close"].diff()
    signed_volume = df["volume"] * np.sign(price_change)
    f["kyle_lambda"] = (
        price_change.rolling(20).cov(signed_volume) /
        (signed_volume.rolling(20).var() + 1e-10)
    )

    # Amihud illiquidity ratio
    f["amihud"] = (
        abs(df["close"].pct_change()) / (df["volume"] * df["close"] + 1e-10)
    ).rolling(20).mean()

    # Spread proxy from daily data
    f["spread_proxy"] = 2 * (df["high"] - df["low"]) / (df["high"] + df["low"] + 1e-8)

    # ── Pattern (4) ───────────────────────────────────────────
    f["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
    f["higher_low"] = (df["low"] > df["low"].shift(1)).astype(int)
    f["inside_bar"] = (
        (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    ).astype(int)
    f["nr7"] = (
        (df["high"] - df["low"]) == (df["high"] - df["low"]).rolling(7).min()
    ).astype(int)

    # ── Time (4) — Cyclical encoding (better than one-hot) ────
    if hasattr(df.index, 'dayofweek'):
        f["day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 5)
        f["day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 5)
        f["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        f["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # ── Composite (1) ─────────────────────────────────────────
    f["momentum_score"] = momentum_score(df)

    return f.replace([np.inf, -np.inf], np.nan).fillna(0)


def create_binary_labels(df: pd.DataFrame, forward_periods: int = 5, threshold: float = 0.003) -> pd.Series:
    """
    BINARY labels: 1 = UP (above threshold), 0 = DOWN (below threshold).
    NaN = neutral zone (dropped during training).
    This is the #1 fix — binary >>> 3-class for financial prediction.
    """
    forward_returns = df["close"].shift(-forward_periods) / df["close"] - 1
    labels = pd.Series(np.nan, index=df.index)
    labels[forward_returns > threshold] = 1
    labels[forward_returns < -threshold] = 0
    return labels


def purged_train_test_split(X, y, test_ratio=0.2, purge_gap=10):
    """
    Time-series split with purge gap to prevent label leakage.
    From Marcos Lopez de Prado's 'Advances in Financial Machine Learning'.
    """
    split = int(len(X) * (1 - test_ratio))
    X_train = X.iloc[:split - purge_gap]
    y_train = y.iloc[:split - purge_gap]
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]
    return X_train, X_test, y_train, y_test


class XGBoostSignalModelV2:
    """
    V2: Binary classification + purged split + feature selection + alpha factors.
    Expected: 55-65% accuracy (vs 45% in V1).
    """

    def __init__(self, model_name: str = "v2_default"):
        self.model_name = model_name
        self.model = None
        self.feature_names = None
        self.selected_features = None
        self.model_path = MODEL_DIR / f"xgb_v2_{model_name}.pkl"
        self.meta_path = MODEL_DIR / f"xgb_v2_{model_name}_meta.json"
        self._load()

    def train(self, df: pd.DataFrame, forward_periods: int = 5) -> dict:
        """Train V2 model with all fixes applied."""
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return {"error": "xgboost not installed"}

        log.info(f"Training V2 model '{self.model_name}' on {len(df)} candles...")

        # V2 features + binary labels
        features = engineer_features_v2(df)
        labels = create_binary_labels(df, forward_periods)

        # Remove warmup rows and future-less rows
        valid = features.iloc[60:-forward_periods].copy()
        valid_labels = labels.iloc[60:-forward_periods].copy()

        # Drop NaN labels (neutral zone) — this is the key fix
        mask = valid.notna().all(axis=1) & valid_labels.notna()
        X = valid[mask]
        y = valid_labels[mask]

        if len(X) < 100:
            return {"error": f"Not enough data: {len(X)} samples"}

        log.info(f"Samples: {len(X)} | UP: {(y == 1).sum()} | DOWN: {(y == 0).sum()}")

        # Purged split (gap=10 to prevent leakage)
        X_train, X_test, y_train, y_test = purged_train_test_split(X, y, purge_gap=10)

        # Class imbalance handling
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        scale = neg_count / pos_count if pos_count > 0 else 1

        # V2 XGBoost: stronger regularization, binary objective
        self.model = XGBClassifier(
            n_estimators=150,  # reduced from 200 to prevent overfit
            max_depth=4,  # reduced from 6
            learning_rate=0.03,  # slower learning
            subsample=0.7,
            colsample_bytree=0.6,  # use fewer features per tree
            min_child_weight=5,  # increased from 3
            reg_alpha=0.5,  # increased L1
            reg_lambda=2.0,  # increased L2
            gamma=0.1,  # minimum loss reduction for split
            scale_pos_weight=scale,
            objective="binary:logistic",
            eval_metric="auc",  # AUC > accuracy for trading
            random_state=42,
            early_stopping_rounds=20,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Feature importance pruning — keep top 40 features
        importances = dict(zip(X.columns, self.model.feature_importances_))
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        self.selected_features = [f for f, imp in sorted_features if imp > 0][:40]
        self.feature_names = list(X.columns)

        # Retrain on selected features only (cleaner model)
        if len(self.selected_features) < len(self.feature_names):
            log.info(f"Pruned: {len(self.feature_names)} → {len(self.selected_features)} features")
            X_train_sel = X_train[self.selected_features]
            X_test_sel = X_test[self.selected_features]

            self.model = XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.5, reg_lambda=2.0, gamma=0.1,
                scale_pos_weight=scale, objective="binary:logistic",
                eval_metric="auc", random_state=42, early_stopping_rounds=20,
            )
            self.model.fit(
                X_train_sel, y_train,
                eval_set=[(X_test_sel, y_test)],
                verbose=False,
            )
            self.feature_names = self.selected_features

        # Evaluate
        X_test_final = X_test[self.feature_names]
        train_acc = self.model.score(X_train[self.feature_names], y_train)
        test_acc = self.model.score(X_test_final, y_test)

        # AUC score
        try:
            from sklearn.metrics import roc_auc_score
            y_proba = self.model.predict_proba(X_test_final)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = 0

        # Precision/recall
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            y_pred = self.model.predict(X_test_final)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        except Exception:
            precision = recall = f1 = 0

        top_features = sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )[:15]

        meta = {
            "model_name": self.model_name,
            "version": "v2",
            "trained_at": datetime.now().isoformat(),
            "samples": len(X),
            "up_count": int((y == 1).sum()),
            "down_count": int((y == 0).sum()),
            "features_total": len(features.columns),
            "features_selected": len(self.feature_names),
            "train_accuracy": round(float(train_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "auc": round(float(auc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "forward_periods": forward_periods,
            "purge_gap": 10,
            "top_features": {k: round(float(v), 4) for k, v in top_features},
        }

        self._save()
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        log.info(
            f"V2 trained: acc={test_acc:.3f}, AUC={auc:.3f}, "
            f"precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}, "
            f"features={len(self.feature_names)}"
        )
        return meta

    def predict(self, df: pd.DataFrame) -> dict:
        """Predict market direction with V2 model."""
        if self.model is None:
            return {"signal": "NEUTRAL", "confidence": 0, "error": "no trained model"}

        features = engineer_features_v2(df)
        latest = features.iloc[-1:][self.feature_names]

        if latest.isna().any().any():
            return {"signal": "NEUTRAL", "confidence": 0, "error": "incomplete features"}

        proba = self.model.predict_proba(latest)[0]
        pred = self.model.predict(latest)[0]

        up_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        down_prob = float(proba[0])

        if pred == 1:
            signal = "UP"
            confidence = up_prob
        else:
            signal = "DOWN"
            confidence = down_prob

        # Only signal if confidence > 55% (edge threshold)
        if confidence < 0.55:
            signal = "NEUTRAL"

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "up_probability": round(up_prob, 4),
            "down_probability": round(down_prob, 4),
            "model": self.model_name,
            "version": "v2",
        }

    def _save(self):
        if self.model:
            with open(self.model_path, "wb") as fp:
                pickle.dump({
                    "model": self.model,
                    "features": self.feature_names,
                    "selected": self.selected_features,
                }, fp)

    def _load(self):
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as fp:
                    data = pickle.load(fp)
                    self.model = data["model"]
                    self.feature_names = data["features"]
                    self.selected_features = data.get("selected", self.feature_names)
                    log.info(f"Loaded V2 model: {self.model_name}")
            except Exception as e:
                log.warning(f"Failed to load V2 model: {e}")
