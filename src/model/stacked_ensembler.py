from typing import Dict, Any, List

import numpy as np


class StackedEnsembleRegressor:
    """
    Simple stacked ensemble:
    - Uses base models to generate predictions as meta-features
    - Uses a meta-model (e.g., Ridge) on top of those predictions
    """

    def __init__(
            self,
            base_models: Dict[str, Any],
            meta_model: Any,
            base_model_order: List[str],
    ) -> None:
        self.base_models = base_models
        self.meta_model = meta_model
        self.base_model_order = base_model_order

    def predict(self, X) -> np.ndarray:
        """
        X: array-like or DataFrame of original features.
        Returns meta_model(base_model_preds(X)).
        """
        # Each base model does X -> y_hat
        preds = []
        for name in self.base_model_order:
            mdl = self.base_models[name]
            preds.append(mdl.predict(X))
        M = np.column_stack(preds)
        return self.meta_model.predict(M)
