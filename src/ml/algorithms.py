"""
ML Algorithms — Multiple classifiers for market direction prediction.

Implements 8 algorithms beyond XGBoost V2:
  1. LightGBM (dart boosting)
  2. Random Forest
  3. Extra Trees
  4. Gradient Boosting (sklearn)
  5. Stacking Ensemble (meta-model)
  6. Voting Ensemble (soft voting)
  7. AdaBoost
  8. CatBoost (optional, requires catboost package)

All share the same interface: get_algorithm(name) -> configured estimator.
"""

from typing import Optional

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression


def _xgboost_classifier(scale_pos_weight: float = 1.0):
    """XGBoost V2 baseline — binary logistic with strong regularization."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )


def _lightgbm_classifier(scale_pos_weight: float = 1.0):
    """LightGBM with dart boosting — often beats XGBoost on financial data."""
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        boosting_type="dart",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )


def _random_forest_classifier():
    """Random Forest — good baseline, less prone to overfitting."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def _extra_trees_classifier():
    """Extra Trees — more random than RF, often better on noisy financial data."""
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def _gradient_boosting_classifier():
    """sklearn Gradient Boosting — deterministic, good for comparison."""
    return GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.7,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
    )


def _adaboost_classifier():
    """AdaBoost — focuses on hard-to-classify samples."""
    return AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
    )


def _catboost_classifier(scale_pos_weight: float = 1.0):
    """CatBoost — handles categorical features natively, often best for tabular."""
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=200,
        depth=5,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        subsample=0.7,
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )


def _stacking_ensemble(
    base_estimators: Optional[list] = None,
    scale_pos_weight: float = 1.0,
):
    """
    Stacking Ensemble — stack multiple models with Logistic Regression meta-learner.
    This is how quant funds combine models.
    """
    if base_estimators is None:
        base_estimators = [
            ("xgb", _xgboost_classifier(scale_pos_weight)),
            ("lgbm", _lightgbm_classifier(scale_pos_weight)),
            ("rf", _random_forest_classifier()),
            ("et", _extra_trees_classifier()),
        ]

    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        ),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )


def _voting_ensemble(
    base_estimators: Optional[list] = None,
    scale_pos_weight: float = 1.0,
):
    """
    Voting Ensemble — soft voting across all models.
    Majority vote = more robust signals.
    """
    if base_estimators is None:
        base_estimators = [
            ("xgb", _xgboost_classifier(scale_pos_weight)),
            ("lgbm", _lightgbm_classifier(scale_pos_weight)),
            ("rf", _random_forest_classifier()),
            ("et", _extra_trees_classifier()),
            ("gb", _gradient_boosting_classifier()),
        ]

    return VotingClassifier(
        estimators=base_estimators,
        voting="soft",
        n_jobs=-1,
    )


# ── Public registry ──────────────────────────────────────────────

ALGORITHM_REGISTRY = {
    "xgboost_v2": {
        "factory": _xgboost_classifier,
        "needs_scale": True,
        "needs_early_stop": True,
        "description": "XGBoost V2 (baseline)",
    },
    "lightgbm": {
        "factory": _lightgbm_classifier,
        "needs_scale": True,
        "needs_early_stop": False,
        "description": "LightGBM (dart boosting)",
    },
    "random_forest": {
        "factory": _random_forest_classifier,
        "needs_scale": False,
        "needs_early_stop": False,
        "description": "Random Forest",
    },
    "extra_trees": {
        "factory": _extra_trees_classifier,
        "needs_scale": False,
        "needs_early_stop": False,
        "description": "Extra Trees",
    },
    "gradient_boosting": {
        "factory": _gradient_boosting_classifier,
        "needs_scale": False,
        "needs_early_stop": False,
        "description": "Gradient Boosting (sklearn)",
    },
    "adaboost": {
        "factory": _adaboost_classifier,
        "needs_scale": False,
        "needs_early_stop": False,
        "description": "AdaBoost",
    },
    "stacking": {
        "factory": _stacking_ensemble,
        "needs_scale": True,
        "needs_early_stop": False,
        "description": "Stacking Ensemble (meta-model)",
    },
    "voting": {
        "factory": _voting_ensemble,
        "needs_scale": True,
        "needs_early_stop": False,
        "description": "Voting Ensemble (soft)",
    },
}


def get_algorithm(name: str, scale_pos_weight: float = 1.0):
    """
    Get a configured classifier by name.

    Args:
        name: Algorithm key from ALGORITHM_REGISTRY.
        scale_pos_weight: Class imbalance ratio (neg/pos count).

    Returns:
        Configured sklearn-compatible estimator.

    Raises:
        ValueError: If name not in registry.
    """
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{name}'. "
            f"Available: {list(ALGORITHM_REGISTRY.keys())}"
        )

    entry = ALGORITHM_REGISTRY[name]
    factory = entry["factory"]

    if entry["needs_scale"]:
        return factory(scale_pos_weight=scale_pos_weight)
    return factory()


def get_individual_algorithms() -> list[str]:
    """Return names of individual (non-ensemble) algorithms."""
    return [
        "xgboost_v2", "lightgbm", "random_forest", "extra_trees",
        "gradient_boosting", "adaboost",
    ]


def get_ensemble_algorithms() -> list[str]:
    """Return names of ensemble algorithms."""
    return ["stacking", "voting"]


def try_add_catboost(scale_pos_weight: float = 1.0):
    """
    Attempt to import CatBoost. Returns (estimator, True) or (None, False).
    Does not crash if catboost is not installed.
    """
    try:
        estimator = _catboost_classifier(scale_pos_weight)
        return estimator, True
    except ImportError:
        return None, False
