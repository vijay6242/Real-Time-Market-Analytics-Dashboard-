# ml_models.py  ─  Machine Learning Prediction Engine
# ════════════════════════════════════════════════════════════════════
# Purpose : Predict tomorrow's market direction (Up / Down / Sideways)
#           and next-close price regression, from historical features.
#
# Models  : RandomForest, XGBoost, LightGBM (classification + regression)
# Features: Price, RSI, MACD, ADX, ATR, VWAP, SuperTrend, EMA,
#           VIX, Rolling Vol, Gap, PCR, Volume, Advance-Decline
#
# Evaluation:
#   Classification → Accuracy, Precision, Recall, F1, ROC-AUC
#   Regression     → MAE, RMSE, R²
# ════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from logger import get_logger
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Price features
    "Open", "High", "Low", "Close",
    "Daily_Ret", "Gap_Pct",
    # Technical
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "EMA_20", "EMA_50", "EMA_200",
    "BB_Upper", "BB_Lower", "BB_Width",
    "ATR", "ATR_Pct",
    "DI_Plus", "DI_Minus", "ADX",
    "SuperTrend",
    "VWAP",
    # Volatility
    "RollingVol_20",
    # Volume
    "Volume",
    # Derived ratios
    "Price_EMA20_Ratio",
    "Price_VWAP_Ratio",
    "RSI_Lag1", "RSI_Lag2",
    "Ret_Lag1",  "Ret_Lag2",  "Ret_Lag3",
    "High_Low_Range",
    "Close_Open_Range",
]


def build_feature_matrix(df: pd.DataFrame,
                          vix_series: pd.Series = None) -> pd.DataFrame:
    """
    Constructs ML feature matrix from a compute_technicals()-processed DataFrame.
    Also adds lag features, ratio features, and optional VIX column.

    Returns: X (features), y_cls (Up/Down/Sideways), y_reg (next close)
    """
    df = df.copy()

    # ── Derived ratio features ──
    df["Price_EMA20_Ratio"]  = df["Close"] / df["EMA_20"].replace(0, np.nan)
    df["Price_VWAP_Ratio"]   = df["Close"] / df["VWAP"].replace(0, np.nan) \
                                if "VWAP" in df.columns else 1.0
    df["High_Low_Range"]     = df["High"] - df["Low"]
    df["Close_Open_Range"]   = df["Close"] - df["Open"]

    # ── Lag features ──
    df["RSI_Lag1"]  = df["RSI"].shift(1)
    df["RSI_Lag2"]  = df["RSI"].shift(2)
    df["Ret_Lag1"]  = df["Daily_Ret"].shift(1)
    df["Ret_Lag2"]  = df["Daily_Ret"].shift(2)
    df["Ret_Lag3"]  = df["Daily_Ret"].shift(3)

    # ── VIX ──
    if vix_series is not None:
        df["VIX"] = vix_series.values[:len(df)] if len(vix_series) >= len(df) \
                    else np.nan
        if "VIX" not in FEATURE_COLS:
            FEATURE_COLS.append("VIX")

    # ── Targets ──
    future_ret = df["Close"].pct_change().shift(-1)
    df["Target_Reg"] = df["Close"].shift(-1)          # Regression target: next close
    df["Target_Cls"] = future_ret.apply(              # Classification: direction
        lambda r: "Up" if r > 0.003 else ("Down" if r < -0.003 else "Sideways")
    )

    # Keep only available columns
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    # Drop rows with NaN in any feature or target
    df_ml = df[feat_cols + ["Target_Cls", "Target_Reg"]].dropna()

    X       = df_ml[feat_cols]
    y_cls   = df_ml["Target_Cls"]
    y_reg   = df_ml["Target_Reg"]

    return X, y_cls, y_reg, feat_cols


# ─────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────

def _cls_models() -> dict:
    models = {
        "Random Forest":     RandomForestClassifier(n_estimators=200, max_depth=6,
                                                     min_samples_leaf=5, random_state=42,
                                                     n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                                         learning_rate=0.05, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0,
                                                   multi_class="multinomial",
                                                   solver="lbfgs", random_state=42),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=5,
                                           learning_rate=0.05, use_label_encoder=False,
                                           eval_metric="mlogloss", random_state=42,
                                           n_jobs=-1, verbosity=0)
    if _HAS_LGB:
        models["LightGBM"] = LGBMClassifier(n_estimators=200, max_depth=5,
                                              learning_rate=0.05, random_state=42,
                                              n_jobs=-1, verbose=-1)
    return models


def _reg_models() -> dict:
    models = {
        "Random Forest":  RandomForestRegressor(n_estimators=200, max_depth=6,
                                                 random_state=42, n_jobs=-1),
        "Linear Regression": LinearRegression(),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=5,
                                          learning_rate=0.05, random_state=42,
                                          n_jobs=-1, verbosity=0)
    if _HAS_LGB:
        models["LightGBM"] = LGBMRegressor(n_estimators=200, max_depth=5,
                                             learning_rate=0.05, random_state=42,
                                             n_jobs=-1, verbose=-1)
    return models


# ─────────────────────────────────────────────────────────
# Time-Series Cross Validation
# ─────────────────────────────────────────────────────────

def _make_pipe(model, is_cls: bool):
    """Wraps model in imputer + scaler pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])


def evaluate_classification(X: pd.DataFrame, y: pd.Series,
                              n_splits: int = 5) -> dict:
    """
    Walk-forward (time-series) cross-validation for all classifiers.
    Returns dict of {model_name: {accuracy, precision, recall, f1, roc_auc, report}}
    """
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    results = {}
    le      = LabelEncoder()
    classes = sorted(y.unique())

    for name, model in _cls_models().items():
        accs, precs, recs, f1s, aucs = [], [], [], [], []

        for train_idx, test_idx in tscv.split(X):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

            if len(ytr.unique()) < 2:
                continue

            pipe = _make_pipe(model, True)
            try:
                pipe.fit(Xtr, ytr)
                pred  = pipe.predict(Xte)
                proba = pipe.predict_proba(Xte) if hasattr(pipe.named_steps["model"], "predict_proba") else None

                accs.append(accuracy_score(yte, pred))
                precs.append(precision_score(yte, pred, average="macro", zero_division=0))
                recs.append(recall_score(yte, pred, average="macro", zero_division=0))
                f1s.append(f1_score(yte, pred, average="macro", zero_division=0))
                if proba is not None and len(classes) == len(pipe.classes_):
                    try:
                        aucs.append(roc_auc_score(yte, proba, multi_class="ovr",
                                                   labels=classes, average="macro"))
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"[ML] {name} fold error: {e}")

        results[name] = {
            "Accuracy":  round(np.mean(accs) * 100, 2) if accs else 0,
            "Precision": round(np.mean(precs) * 100, 2) if precs else 0,
            "Recall":    round(np.mean(recs) * 100, 2) if recs else 0,
            "F1 Score":  round(np.mean(f1s) * 100, 2) if f1s else 0,
            "ROC-AUC":   round(np.mean(aucs), 4) if aucs else None,
        }
        logger.info(f"[ML] {name} → Acc {results[name]['Accuracy']:.1f}%  F1 {results[name]['F1 Score']:.1f}%")

    return results


def evaluate_regression(X: pd.DataFrame, y: pd.Series,
                          n_splits: int = 5) -> dict:
    """
    Walk-forward cross-validation for regression models.
    Returns dict of {model_name: {MAE, RMSE, R2}}
    """
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    results = {}

    for name, model in _reg_models().items():
        maes, rmses, r2s = [], [], []
        for train_idx, test_idx in tscv.split(X):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
            pipe = _make_pipe(model, False)
            try:
                pipe.fit(Xtr, ytr)
                pred = pipe.predict(Xte)
                maes.append(mean_absolute_error(yte, pred))
                rmses.append(np.sqrt(mean_squared_error(yte, pred)))
                r2s.append(r2_score(yte, pred))
            except Exception as e:
                logger.warning(f"[ML] {name} reg fold error: {e}")

        results[name] = {
            "MAE":  round(np.mean(maes),  2) if maes else 0,
            "RMSE": round(np.mean(rmses), 2) if rmses else 0,
            "R²":   round(np.mean(r2s),   4) if r2s  else 0,
        }
        logger.info(f"[ML] {name} → MAE {results[name]['MAE']:.1f}  R² {results[name]['R²']:.3f}")

    return results


# ─────────────────────────────────────────────────────────
# Live prediction (train on all data, predict next bar)
# ─────────────────────────────────────────────────────────

def predict_tomorrow(df: pd.DataFrame,
                      vix_series: pd.Series = None) -> dict:
    """
    Trains all classifiers on full history and predicts the next session.

    Returns:
      {
        "direction":      "Up" | "Down" | "Sideways"
        "confidence":     0-100 (%)
        "probabilities":  {"Up": x, "Down": y, "Sideways": z}
        "price_forecast": float
        "model_votes":    {model_name: direction}
        "feature_importance": {feature: importance}
        "signal":         str
      }
    """
    X, y_cls, y_reg, feat_cols = build_feature_matrix(df, vix_series)

    if len(X) < 30:
        return {"error": "Not enough data (need ≥30 rows after feature engineering)."}

    X_train, X_last = X.iloc[:-1], X.iloc[[-1]]
    y_cls_train     = y_cls.iloc[:-1]
    y_reg_train     = y_reg.iloc[:-1]

    votes      = {}
    proba_sum  = {"Up": 0, "Down": 0, "Sideways": 0}
    n_proba    = 0
    feat_imp   = {}

    # Classification vote
    for name, model in _cls_models().items():
        pipe = _make_pipe(model, True)
        try:
            pipe.fit(X_train, y_cls_train)
            pred = pipe.predict(X_last)[0]
            votes[name] = pred
            if hasattr(pipe.named_steps["model"], "predict_proba"):
                proba = pipe.predict_proba(X_last)[0]
                for cls, p in zip(pipe.classes_, proba):
                    proba_sum[cls] = proba_sum.get(cls, 0) + p
                n_proba += 1
            # Feature importance
            m = pipe.named_steps["model"]
            if hasattr(m, "feature_importances_"):
                for feat, imp in zip(feat_cols, m.feature_importances_):
                    feat_imp[feat] = feat_imp.get(feat, 0) + imp
        except Exception as e:
            logger.warning(f"[ML] {name} predict error: {e}")

    # Majority vote
    if votes:
        from collections import Counter
        vote_counts = Counter(votes.values())
        direction   = vote_counts.most_common(1)[0][0]
        confidence  = vote_counts[direction] / len(votes) * 100
    else:
        direction, confidence = "Sideways", 33.3

    # Avg probabilities
    if n_proba > 0:
        avg_proba = {k: round(v / n_proba * 100, 1) for k, v in proba_sum.items()}
    else:
        avg_proba = {"Up": 33, "Down": 33, "Sideways": 34}

    # Price regression
    price_forecast = None
    for name, model in _reg_models().items():
        pipe = _make_pipe(model, False)
        try:
            pipe.fit(X_train, y_reg_train)
            price_forecast = round(float(pipe.predict(X_last)[0]), 2)
            break
        except Exception:
            pass

    # Normalise feature importances
    if feat_imp:
        total_imp = sum(feat_imp.values()) or 1
        feat_imp  = {k: round(v / total_imp * 100, 2)
                     for k, v in sorted(feat_imp.items(), key=lambda x: -x[1])[:10]}

    # Signal text
    current = df["Close"].iloc[-1]
    chg_pct = ((price_forecast - current) / current * 100) if price_forecast else 0
    signal  = (f"{'📈 Bullish' if direction=='Up' else '📉 Bearish' if direction=='Down' else '↔️ Sideways'} — "
               f"Models predict {direction} with {confidence:.0f}% consensus. "
               f"Forecast close: ₹{price_forecast:,.2f} ({chg_pct:+.2f}%)" if price_forecast
               else f"Models predict {direction}.")

    return {
        "direction":          direction,
        "confidence":         round(confidence, 1),
        "probabilities":      avg_proba,
        "price_forecast":     price_forecast,
        "model_votes":        votes,
        "feature_importance": feat_imp,
        "signal":             signal,
    }


def predict_today(df: pd.DataFrame, vix_series: pd.Series = None) -> dict:
    """
    Same as predict_tomorrow but uses the most recent complete bar
    (useful during market hours when today's bar is incomplete).
    Trains on all bars except the last 2, predicts on bar[-2].
    """
    df_shifted = df.iloc[:-1].copy() if len(df) > 2 else df.copy()
    result = predict_tomorrow(df_shifted, vix_series)
    result["note"] = "Prediction for current session (trained on all prior history)."
    return result


# ─────────────────────────────────────────────────────────
# Full Training Report  (run once, cache result)
# ─────────────────────────────────────────────────────────

def full_training_report(df: pd.DataFrame,
                          vix_series: pd.Series = None,
                          n_splits: int = 5) -> dict:
    """
    Runs both classification and regression evaluation,
    plus tomorrow's prediction.

    Returns:
      {
        "classification": {model: metrics},
        "regression":     {model: metrics},
        "prediction":     {...},
        "feature_cols":   [...],
        "n_samples":      int,
      }
    """
    logger.info("[ML] Starting full training report ...")
    X, y_cls, y_reg, feat_cols = build_feature_matrix(df, vix_series)

    if len(X) < 30:
        return {"error": f"Need ≥30 rows, got {len(X)}."}

    cls_results = evaluate_classification(X, y_cls, n_splits)
    reg_results = evaluate_regression(X, y_reg, n_splits)
    pred_result = predict_tomorrow(df, vix_series)

    logger.info("[ML] Report complete.")
    return {
        "classification": cls_results,
        "regression":     reg_results,
        "prediction":     pred_result,
        "feature_cols":   feat_cols,
        "n_samples":      len(X),
    }