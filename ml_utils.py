# ml_utils.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


# ===================== Utilidades generales =====================
def infer_task(y: pd.Series) -> str:
    """Detecta si la tarea es clasificación (pocas categorías) o regresión."""
    if y.dtype.kind in "ifu":
        return "clasificación" if y.nunique(dropna=True) <= 10 else "regresión"
    return "clasificación"


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Preprocesador: numérico (imputación+escalado) y categórico (imputación+onehot)."""
    num_cols = list(X.select_dtypes(include=np.number).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="mean")),
                         ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, num_cols),
                              ("cat", cat_pipe, cat_cols)])


def compute_perm_importance(pipeline: Pipeline,
                            X_te: pd.DataFrame,
                            y_te: pd.Series,
                            tarea: str,
                            topk: int = 8) -> pd.DataFrame:
    """Importancias por permutación (en espacio original X_te)."""
    scoring = "f1_macro" if tarea == "clasificación" else "r2"
    pi = permutation_importance(
        pipeline, X_te, y_te,
        n_repeats=5, random_state=42, n_jobs=-1, scoring=scoring
    )
    importances = pd.DataFrame({
        "feature": X_te.columns,
        "importance": pi.importances_mean,
        "std": pi.importances_std
    }).sort_values("importance", ascending=False)
    return importances.head(topk)


# ===================== Corrección RMSE (compatibilidad sklearn) =====================
def _rmse(y_true, y_pred) -> float:
    """
    RMSE compatible con versiones de scikit-learn que NO aceptan squared=False.
    """
    yt = np.asarray(pd.Series(y_true), dtype=float)
    yp = np.asarray(pd.Series(y_pred), dtype=float)
    return float(np.sqrt(mean_squared_error(yt, yp)))


# ===================== Split robusto =====================
def safe_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    tarea: str,
    test_size: float = 0.2,
    random_state: int = 42,
    min_per_class: int = 2,
    auto_increase_test_size: bool = False,
    max_test_size: float = 0.4,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
    """
    Split robusto ante NaN en y y clases raras.

    - Elimina NaN en y (alineando X).
    - Si clasificación, estratifica solo si TODAS las clases tienen >= min_per_class
      y cada clase aporta ≥1 muestra a test y ≥1 a train.
    - Si auto_increase_test_size=True, intenta aumentar test_size (pasos de 0.05)
      hasta max_test_size para permitir estratificación.

    Devuelve: X_tr, X_te, y_tr, y_te, info (dict con flags y mensajes).
    """
    info: Dict = {
        "stratified": False,
        "reason": "",
        "n_classes": None,
        "class_counts": None,
        "used_test_size": float(test_size),
        "auto_adjusted": False,
        "rare_classes": [],
    }

    # 1) Sanitizar y (quitar NaN)
    y_series = pd.Series(y)
    mask = ~y_series.isna()
    if mask.sum() < len(y_series):
        X = X.loc[mask]
        y_series = y_series.loc[mask]
        X = X.reset_index(drop=True)
        y_series = y_series.reset_index(drop=True)
        info["reason"] += "Se eliminaron filas con etiqueta NaN. "

    stratify = None
    if tarea == "clasificación":
        vc = y_series.value_counts()
        info["n_classes"] = int(vc.shape[0])
        info["class_counts"] = vc.to_dict()
        info["rare_classes"] = [str(k) for k, v in vc.items() if v < min_per_class]

        def can_stratify(ts: float) -> bool:
            return (
                (vc >= min_per_class).all()
                and (vc * ts >= 1).all()
                and (vc * (1.0 - ts) >= 1).all()
            )

        # ¿Podemos estratificar con el test_size actual?
        if can_stratify(test_size):
            stratify = y_series
            info["stratified"] = True
        else:
            if auto_increase_test_size:
                ts = float(test_size)
                while ts + 1e-9 < float(max_test_size) and not can_stratify(ts):
                    ts = round(ts + 0.05, 2)
                if can_stratify(ts):
                    stratify = y_series
                    info["stratified"] = True
                    info["auto_adjusted"] = True
                    info["used_test_size"] = ts
                else:
                    info["reason"] += (
                        "No fue posible estratificar ni aumentando test_size hasta el máximo permitido. "
                    )
            else:
                info["reason"] += (
                    "Estratificación desactivada: clases con muy pocos ejemplos o test_size insuficiente. "
                )

    # 2) Split final
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y_series,
        test_size=info["used_test_size"],
        random_state=random_state,
        stratify=stratify,
    )
    return X_tr, X_te, y_tr, y_te, info


# ===================== Métricas, artefactos e insights ML =====================
def ml_metrics_and_artifacts(pipe: Pipeline,
                             X_tr, X_te, y_tr, y_te,
                             tarea: str) -> Dict:
    """Entrena el pipeline y devuelve métricas + artefactos auxiliares."""
    pipe.fit(X_tr, y_tr)
    pred_tr = pipe.predict(X_tr)
    pred_te = pipe.predict(X_te)

    out: Dict = {"tarea": tarea}

    if tarea == "clasificación":
        out["acc_tr"] = accuracy_score(y_tr, pred_tr)
        out["acc_te"] = accuracy_score(y_te, pred_te)
        out["f1_tr"] = f1_score(y_tr, pred_tr, average="macro")
        out["f1_te"] = f1_score(y_te, pred_te, average="macro")

        labels = sorted(pd.Series(y_te).astype(str).unique())
        cm = confusion_matrix(
            pd.Series(y_te).astype(str),
            pd.Series(pred_te).astype(str),
            labels=labels
        )
        out["labels"] = labels
        out["cm"] = cm
        out["y_te"] = y_te
        out["pred_te"] = pred_te

        # Errores frecuentes (real→pred)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        errs = []
        for r in labels:
            for p in labels:
                if r != p:
                    val = int(cm_df.loc[r, p])
                    if val > 0:
                        errs.append((r, p, val))
        out["top_errors"] = sorted(errs, key=lambda x: x[2], reverse=True)[:5]

    else:
        # —— Regresión (RMSE corregido) ——
        out["mae"] = mean_absolute_error(y_te, pred_te)
        out["rmse"] = _rmse(y_te, pred_te)               # <- sin squared=False
        out["r2"] = r2_score(y_te, pred_te)

        # Baseline ingenuo: predecir la media de train
        y_tr_mean = float(np.mean(pd.Series(y_tr, dtype=float)))
        baseline = np.full(shape=len(pd.Series(y_te)), fill_value=y_tr_mean, dtype=float)
        out["rmse_baseline"] = _rmse(y_te, baseline)     # <- idem

        out["y_te"] = y_te
        out["pred_te"] = pred_te

        resid = pd.Series(y_te, dtype=float) - pd.Series(pred_te, dtype=float)
        out["resid_mean"] = float(np.mean(resid))
        out["resid_p95"] = float(np.percentile(np.abs(resid), 95))

    # Importancias por permutación (en X original)
    out["perm_importance"] = compute_perm_importance(pipe, X_te, y_te, tarea, topk=8)
    return out


def ml_key_insights(ml: Dict, X_te: pd.DataFrame) -> list[str]:
    """Bullets compactos con hallazgos clave de ML (métricas, errores, importancias)."""
    bullets = []
    ntest = len(ml["y_te"])
    bullets.append(f"Evaluación en test con {ntest} muestras.")

    if ml["tarea"] == "clasificación":
        bullets.append(
            f"Accuracy={ml['acc_te']:.3f}, F1-macro={ml['f1_te']:.3f}. "
            f"(train acc={ml['acc_tr']:.3f}, f1={ml['f1_tr']:.3f})"
        )
        dist = pd.Series(ml["y_te"]).astype(str).value_counts(normalize=True)
        maj = dist.idxmax()
        bullets.append(f"Clase mayoritaria en test: {maj} ({dist.max()*100:.1f}%).")
        if ml.get("top_errors"):
            top = ", ".join([f"{r}→{p}:{n}" for r, p, n in ml["top_errors"][:3]])
            bullets.append(f"Errores f
