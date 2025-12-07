import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src.utils.log_utils import get_logger

LOGGER = get_logger("explain_utils")


class _PredictOnlyWrapper:
    """
    Lightweight wrapper to adapt already-fitted models (e.g., joblib-loaded
    stacked ensembles) so they satisfy scikit-learn's expectation that an
    estimator implement `fit`, while we only ever use `predict`.

    This lets us:
      - keep saving/loading models with joblib exactly as you do now
      - run permutation_importance on the loaded model
      - still use the *original* model object for SHAP
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y=None):
        # No-op: model is already trained; permutation_importance only checks
        # that `fit` exists, it does not re-train the model.
        return self

    def predict(self, X):
        return self.base_estimator.predict(X)


def permutation_importance_scores(
        model,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        n_repeats: int = 5,
) -> pd.DataFrame:
    """
    Compute permutation importance for a joblib-loaded (already-fitted) model.

    We wrap the loaded model in _PredictOnlyWrapper so that scikit-learn's
    parameter validation sees an estimator with a `fit` method, but we still
    use the original model's `predict` under the hood.
    """
    # Always wrap to be robust to custom classes that may not satisfy
    # sklearn's HasMethods('fit') constraint after serialization.
    estimator = _PredictOnlyWrapper(model)

    r = permutation_importance(
        estimator,
        X_valid,
        y_valid,
        n_repeats=n_repeats,
        random_state=42,
    )

    return (
        pd.DataFrame(
            {
                "feature": X_valid.columns,
                "importance": r.importances_mean,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def shap_summary_df(model, X_sample: pd.DataFrame) -> pd.DataFrame:
    """
    Returns mean |SHAP| per feature for tree models (XGBoost/LightGBM) and
    generic models via KernelExplainer fallback.

    IMPORTANT:
    - We use the *original* model here (NOT the wrapper) so that tree models
      still expose their native boosters to TreeExplainer.
    - For your stacked/other models, we fall back to KernelExplainer on
      model.predict, which works fine on joblib-loaded fitted models.
    """
    try:
        # Tree-specific fast path: XGBoost / LightGBM
        if hasattr(model, "get_booster") or model.__class__.__name__.startswith("LGBM"):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_sample)
        else:
            # Generic / stacked / linear models: KernelExplainer on predict
            X_background = shap.sample(X_sample, 200, random_state=42)
            explainer = shap.KernelExplainer(model.predict, X_background)
            shap_vals = explainer.shap_values(
                shap.sample(X_sample, 200, random_state=42),
                nsamples=200,
            )

        sv = np.abs(shap_vals).mean(axis=0)
        return (
            pd.DataFrame(
                {
                    "feature": X_sample.columns,
                    "mean_abs_shap": sv,
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    except Exception:
        LOGGER.exception("shap_summary_df")
        # In reports we don't want to hard-fail if SHAP blows up; just
        # return an empty frame and let the UI show "no SHAP available".
        return pd.DataFrame({"feature": [], "mean_abs_shap": []})
