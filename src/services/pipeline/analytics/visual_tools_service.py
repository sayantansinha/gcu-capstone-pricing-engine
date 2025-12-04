import io, base64
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.gofplots import qqplot


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    out = "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return out


def chart_actual_vs_pred(y_true: pd.Series, y_pred: pd.Series) -> str:
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    mn, mx = float(y_true.min()), float(y_true.max())
    ax.plot([mn, mx], [mn, mx], linestyle="--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    return _to_b64(fig)


def chart_residuals(y_true: pd.Series, y_pred: pd.Series) -> str:
    resid = y_true - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, resid)
    ax.axhline(0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Predicted")
    return _to_b64(fig)


def chart_residuals_qq(y_true: pd.Series, y_pred: pd.Series) -> str:
    resid = y_true - y_pred
    fig = qqplot(resid, line="s")
    fig.suptitle("Residuals Q–Q Plot")
    return _to_b64(fig)
