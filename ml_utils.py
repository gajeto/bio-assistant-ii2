# ml_utils.py
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix
)

def infer_task(y: pd.Series) -> str:
    if y.dtype.kind in "ifu":
        return "clasificación" if y.nunique(dropna=True) <= 10 else "regresión"
    return "clasificación"

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = list(X.select_dtypes(include=np.number).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

def compute_perm_importance(pipeline: Pipeline, X_te: pd.DataFrame, y_te: pd.Series, tarea: str, topk: int = 8) -> pd.DataFrame:
    scoring = "f1_macro" if tarea == "clasificación" else "r2"
    pi = permutation_importance(pipeline, X_te, y_te, n_repeats=5, random_state=42, n_jobs=-1, scoring=scoring)
    importances = pd.DataFrame({
        "feature": X_te.columns,
        "importance": pi.importances_mean,
        "std": pi.importances_std
    }).sort_values("importance", ascending=False)
    return importances.head(topk)

def ml_metrics_and_artifacts(pipe: Pipeline, X_tr, X_te, y_tr, y_te, tarea: str) -> Dict:
    pipe.fit(X_tr, y_tr)
    pred_tr = pipe.predict(X_tr)
    pred_te = pipe.predict(X_te)

    out = {"tarea": tarea}

    if tarea == "clasificación":
        out["acc_tr"] = accuracy_score(y_tr, pred_tr)
        out["acc_te"] = accuracy_score(y_te, pred_te)
        out["f1_tr"] = f1_score(y_tr, pred_tr, average="macro")
        out["f1_te"] = f1_score(y_te, pred_te, average="macro")
        labels = sorted(pd.Series(y_te).astype(str).unique())
        cm = confusion_matrix(y_te.astype(str), pd.Series(pred_te).astype(str), labels=labels)
        out["labels"] = labels
        out["cm"] = cm
        out["y_te"] = y_te
        out["pred_te"] = pred_te
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
        out["mae"] = mean_absolute_error(y_te, pred_te)
        out["rmse"] = mean_squared_error(y_te, pred_te, squared=False)
        out["r2"] = r2_score(y_te, pred_te)
        baseline = np.full_like(y_te, fill_value=np.mean(y_tr), dtype=float)
        out["rmse_baseline"] = mean_squared_error(y_te, baseline, squared=False)
        out["y_te"] = y_te
        out["pred_te"] = pred_te
        resid = y_te - pred_te
        out["resid_mean"] = float(np.mean(resid))
        out["resid_p95"] = float(np.percentile(np.abs(resid), 95))

    out["perm_importance"] = compute_perm_importance(pipe, X_te, y_te, tarea, topk=8)
    return out

def ml_key_insights(ml: Dict, X_te: pd.DataFrame) -> list[str]:
    bullets = []
    ntest = len(ml["y_te"])
    bullets.append(f"Evaluación en test con {ntest} muestras.")
    if ml["tarea"] == "clasificación":
        bullets.append(f"Accuracy={ml['acc_te']:.3f}, F1-macro={ml['f1_te']:.3f}. "
                       f"(train acc={ml['acc_tr']:.3f}, f1={ml['f1_tr']:.3f})")
        dist = pd.Series(ml["y_te"]).astype(str).value_counts(normalize=True)
        maj = dist.idxmax()
        bullets.append(f"Clase mayoritaria en test: {maj} ({dist.max()*100:.1f}%).")
        if ml.get("top_errors"):
            top = ", ".join([f"{r}→{p}:{n}" for r,p,n in ml["top_errors"][:3]])
            bullets.append(f"Errores frecuentes (real→pred): {top}.")
    else:
        bullets.append(f"MAE={ml['mae']:.3f}, RMSE={ml['rmse']:.3f}, R²={ml['r2']:.3f}. "
                       f"RMSE baseline(media)={ml['rmse_baseline']:.3f}.")
        bullets.append(f"Residuo medio ≈ {ml['resid_mean']:.3f}; p95(|resid|) ≈ {ml['resid_p95']:.3f}.")
    imp = ml["perm_importance"]
    if not imp.empty:
        top_feats = ", ".join(f"{row.feature}({row.importance:.3f})" for _, row in imp.head(5).iterrows())
        bullets.append("Features más influyentes (perm.): " + top_feats)
    return bullets

def ml_objective_interpretation(ml: Dict) -> str:
    if ml["tarea"] == "clasificación":
        acc, f1 = ml["acc_te"], ml["f1_te"]
        acc_tr, f1_tr = ml["acc_tr"], ml["f1_tr"]
        gap = (acc_tr - acc) + (f1_tr - f1)
        msg = [f"El baseline de clasificación muestra Accuracy={acc:.3f} y F1-macro={f1:.3f} en test."]
        if gap > 0.15:
            msg.append("Existe indicio de sobreajuste (gap notable entre train y test).")
        elif gap < -0.05:
            msg.append("Rendimiento en test mejor que en train (posible subajuste o split favorable).")
        else:
            msg.append("Generalización razonable: métricas similares entre train y test.")
        if ml.get("top_errors"):
            r,p,n = ml["top_errors"][0]
            msg.append(f"Error más frecuente: {r}→{p} (conteo={n}). Revisar separación entre estas clases.")
        return " ".join(msg)
    else:
        rmse, r2, base = ml["rmse"], ml["r2"], ml["rmse_baseline"]
        msg = [f"El baseline de regresión obtiene RMSE={rmse:.3f} y R²={r2:.3f}."]
        if rmse < base:
            msg.append(f"Mejora respecto al baseline ingenuo (RMSE media={base:.3f}).")
        else:
            msg.append(f"No mejora al baseline ingenuo (RMSE media={base:.3f}); conviene ajustar features/modelo.")
        if abs(ml["resid_mean"]) > 0.1*rmse:
            msg.append("Sesgo en residuales (media distinta de 0): revisar especificación del modelo.")
        return " ".join(msg)
