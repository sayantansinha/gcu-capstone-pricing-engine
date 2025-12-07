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
        scoring: str = "r2",
) -> pd.DataFrame:
    """
    Compute permutation importance for a joblib-loaded (already-fitted) model.

    We wrap the loaded model in _PredictOnlyWrapper so that scikit-learn's
    parameter validation sees an estimator with a `fit` method, but we still
    use the original model's `predict` under the hood.

    Parameters
    ----------
    model : fitted model (e.g., joblib-loaded stacked ensemble)
    X_valid : validation features
    y_valid : validation target
    n_repeats : number of permutation repeats
    scoring : sklearn scoring string, default "r2" for regression

    Returns
    -------
    DataFrame with columns ["feature", "importance"], sorted descending.
    """
    # Wrap to satisfy sklearn's "has fit" requirement; we rely on predict()
    estimator = _PredictOnlyWrapper(model)

    r = permutation_importance(
        estimator,
        X_valid,
        y_valid,
        n_repeats=n_repeats,
        random_state=42,
        scoring=scoring,
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
    Compute SHAP summary values for a joblib-loaded, already-fitted model.

    Works for:
      - Tree models (XGBoost / LightGBM) via TreeExplainer
      - Stacked, linear, MLP, or general models via KernelExplainer

    Avoids:
      - Infinite runtimes (limits background/eval/nsamples)
      - sklearn warnings about missing feature names
    """
    try:
        if X_sample is None or len(X_sample) == 0:
            return pd.DataFrame({"feature": [], "mean_abs_shap": []})

        feature_names = list(X_sample.columns)

        # -----------------------------------------------------------
        # Fast path: tree-based models → TreeExplainer
        # -----------------------------------------------------------
        if hasattr(model, "get_booster") or model.__class__.__name__.startswith("LGBM"):
            eval_data = X_sample.sample(
                n=min(200, len(X_sample)), random_state=42
            )
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(eval_data)

            sv = np.abs(shap_vals).mean(axis=0)
            return (
                pd.DataFrame(
                    {
                        "feature": eval_data.columns,
                        "mean_abs_shap": sv,
                    }
                )
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )

        # -----------------------------------------------------------
        # Generic / Stacked / Linear / MLP → KernelExplainer
        # -----------------------------------------------------------

        # Limit sample sizes heavily for runtime safety
        background = X_sample.sample(
            n=min(50, len(X_sample)), random_state=42
        )
        eval_data = X_sample.sample(
            n=min(100, len(X_sample)), random_state=123
        )

        def predict_with_feature_names(X):
            """
            SHAP often passes numpy arrays; convert them back to DataFrame
            so sklearn models requiring feature names behave correctly.
            """
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=feature_names)
            else:
                X = X[feature_names]  # enforce ordering + subset
            return model.predict(X)

        explainer = shap.KernelExplainer(
            predict_with_feature_names,
            background,
        )

        shap_vals = explainer.shap_values(
            eval_data,
            nsamples=100,  # keeps runtime under control
        )

        sv = np.abs(shap_vals).mean(axis=0)
        return (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "mean_abs_shap": sv,
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    except Exception:
        # Silent failure to keep reports UI stable
        return pd.DataFrame({"feature": [], "mean_abs_shap": []})
