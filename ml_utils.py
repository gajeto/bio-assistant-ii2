# ml_utils.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import precision_recall_fscore_support




# -------- Tarea, RMSE, Split robusto (como antes) --------
def infer_task(y: pd.Series) -> str:
    if y.dtype.kind in "ifu":
        return "clasificación" if y.nunique(dropna=True) <= 10 else "regresión"
    return "clasificación"

def _rmse(y_true, y_pred) -> float:
    yt = np.asarray(pd.Series(y_true), dtype=float)
    yp = np.asarray(pd.Series(y_pred), dtype=float)
    return float(np.sqrt(mean_squared_error(yt, yp)))

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
    info: Dict = {
        "stratified": False,
        "reason": "",
        "n_classes": None,
        "class_counts": None,
        "used_test_size": float(test_size),
        "auto_adjusted": False,
        "rare_classes": [],
    }
    y_series = pd.Series(y)
    mask = ~y_series.isna()
    if mask.sum() < len(y_series):
        X = X.loc[mask].reset_index(drop=True)
        y_series = y_series.loc[mask].reset_index(drop=True)
        info["reason"] += "Se eliminaron filas con etiqueta NaN. "
    stratify = None
    if tarea == "clasificación":
        vc = y_series.value_counts()
        info["n_classes"] = int(vc.shape[0])
        info["class_counts"] = vc.to_dict()
        info["rare_classes"] = [str(k) for k, v in vc.items() if v < min_per_class]
        def can_stratify(ts: float) -> bool:
            return (vc >= min_per_class).all() and (vc * ts >= 1).all() and (vc * (1 - ts) >= 1).all()
        if can_stratify(test_size):
            stratify = y_series; info["stratified"] = True
        else:
            if auto_increase_test_size:
                ts = float(test_size)
                while ts + 1e-9 < float(max_test_size) and not can_stratify(ts):
                    ts = round(ts + 0.05, 2)
                if can_stratify(ts):
                    stratify = y_series; info["stratified"] = True
                    info["auto_adjusted"] = True; info["used_test_size"] = ts
                else:
                    info["reason"] += "No fue posible estratificar ni aumentando test_size. "
            else:
                info["reason"] += "Estratificación desactivada por clases raras/test_size insuficiente. "
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_series, test_size=info["used_test_size"], random_state=random_state, stratify=stratify
    )
    return X_tr, X_te, y_tr, y_te, info

# -------- Preprocesadores --------
def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Básico: imputación media/moda + StandardScaler + OneHot."""
    num_cols = list(X.select_dtypes(include=np.number).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

def make_preprocessor_advanced(
    X: pd.DataFrame,
    robust: bool = True,
    yeo_johnson: bool = True,
    onehot_min_freq: Optional[float] = 0.01
) -> ColumnTransformer:
    """
    Avanzado: imputación mediana/moda, Yeo-Johnson (opcional), RobustScaler (opcional),
    OneHot con min_frequency (opcional) para reducir cardinalidad rara.
    """
    num_cols = list(X.select_dtypes(include=np.number).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_steps = [("imputer", SimpleImputer(strategy="median" if robust else "mean"))]
    if yeo_johnson and len(num_cols) > 0:
        num_steps.append(("yeoj", PowerTransformer(method="yeo-johnson", standardize=False)))
    num_steps.append(("scaler", RobustScaler() if robust else StandardScaler()))
    if onehot_min_freq is not None:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=onehot_min_freq))
        ])
    else:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
    return ColumnTransformer([("num", Pipeline(num_steps), num_cols),
                              ("cat", cat_pipe, cat_cols)])


# -------- Métricas e insights --------
def compute_perm_importance(pipeline: Pipeline, X_te: pd.DataFrame, y_te: pd.Series, tarea: str, topk: int = 8) -> pd.DataFrame:
    scoring = "f1_macro" if tarea == "clasificación" else "r2"
    pi = permutation_importance(pipeline, X_te, y_te, n_repeats=5, random_state=42, n_jobs=-1, scoring=scoring)
    importances = pd.DataFrame({"feature": X_te.columns, "importance": pi.importances_mean, "std": pi.importances_std}).sort_values("importance", ascending=False)
    return importances.head(topk)

def ml_metrics_and_artifacts(pipe: Pipeline, X_tr, X_te, y_tr, y_te, tarea: str) -> Dict:
    pipe.fit(X_tr, y_tr)
    pred_tr = pipe.predict(X_tr); pred_te = pipe.predict(X_te)
    out: Dict = {"tarea": tarea}
    if tarea == "clasificación":
        out["acc_tr"] = accuracy_score(y_tr, pred_tr)
        out["acc_te"] = accuracy_score(y_te, pred_te)
        out["f1_tr"] = f1_score(y_tr, pred_tr, average="macro")
        out["f1_te"] = f1_score(y_te, pred_te, average="macro")
        labels = sorted(pd.Series(y_te).astype(str).unique())
        cm = confusion_matrix(pd.Series(y_te).astype(str), pd.Series(pred_te).astype(str), labels=labels)
        out["labels"] = labels; out["cm"] = cm
        out["y_te"] = y_te; out["pred_te"] = pred_te
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        errs = [(r,p,int(cm_df.loc[r,p])) for r in labels for p in labels if r!=p and int(cm_df.loc[r,p])>0]
        out["top_errors"] = sorted(errs, key=lambda x: x[2], reverse=True)[:5]
    else:
        out["mae"] = mean_absolute_error(y_te, pred_te)
        out["rmse"] = _rmse(y_te, pred_te)
        out["r2"] = r2_score(y_te, pred_te)
        y_tr_mean = float(np.mean(pd.Series(y_tr, dtype=float)))
        baseline = np.full(shape=len(pd.Series(y_te)), fill_value=y_tr_mean, dtype=float)
        out["rmse_baseline"] = _rmse(y_te, baseline)
        out["y_te"] = y_te; out["pred_te"] = pred_te
        resid = pd.Series(y_te, dtype=float) - pd.Series(pred_te, dtype=float)
        out["resid_mean"] = float(np.mean(resid)); out["resid_p95"] = float(np.percentile(np.abs(resid), 95))
    out["perm_importance"] = compute_perm_importance(pipe, X_te, y_te, tarea, topk=8)
    return out

def ml_key_insights(ml: Dict, X_te: pd.DataFrame) -> list[str]:
    """
    Resumen enriquecido del baseline ML con la mayor información útil para LLM.
    Incluye:
    - Métricas globales (y gap train-test).
    - Desbalance y distribución por clase (clasificación).
    - Métricas por clase (top peores por F1) y confusiones más frecuentes.
    - Mejoras respecto a baseline ingenuo (regresión) y diagnóstico de residuales.
    - Importancias por permutación (top) y variables candidatas a ingeniería.
    - Señales de sobreajuste/subajuste.
    """
    bullets = []

    # 0) Modelo utilizado (si está guardado por la app)
    if "modelo_name" in ml:
        bullets.append(f"Modelo: **{ml['modelo_name']}**.")

    # 1) Tamaño de test
    ntest = len(ml["y_te"])
    bullets.append(f"Evaluación en test: **{ntest}** muestras.")

    if ml["tarea"] == "clasificación":
        # 2) Métricas globales y gap
        acc_te, f1_te = ml["acc_te"], ml["f1_te"]
        acc_tr, f1_tr = ml["acc_tr"], ml["f1_tr"]
        bullets.append(
            f"Rendimiento global (test): Accuracy=**{acc_te:.3f}** ({acc_te*100:.1f}%), "
            f"F1-macro=**{f1_te:.3f}** ({f1_te*100:.1f}%). "
            f"(train: acc={acc_tr:.3f}/{acc_tr*100:.1f}%, f1={f1_tr:.3f}/{f1_tr*100:.1f}%)."
        )
        gap = (acc_tr - acc_te) + (f1_tr - f1_te)
        if gap > 0.15:
            bullets.append("Señal de **sobreajuste** (gap notable train-test).")
        elif gap < -0.05:
            bullets.append("Test > Train (posible **subajuste** o split favorable).")
        else:
            bullets.append("Generalización **razonable** (métricas similares entre train y test).")

        # 3) Distribución de clases (desbalance)
        yte = pd.Series(ml["y_te"]).astype(str)
        dist = yte.value_counts(normalize=True)
        maj = dist.idxmax(); pmaj = float(dist.max())
        bullets.append(
            f"Distribución de la etiqueta (test): mayoritaria **{maj}** ({pmaj*100:.1f}%)."
        )
        if pmaj >= 0.7:
            bullets.append("**Desbalance alto**; considerar estratificación, ponderación o muestreo.")

        # 4) Métricas por clase (top peores F1)
        y_pred = pd.Series(ml["pred_te"]).astype(str).values
        labels = sorted(yte.unique().tolist())
        pr, rc, f1, _ = precision_recall_fscore_support(yte.values, y_pred, labels=labels, zero_division=0)
        per_class = pd.DataFrame({"clase": labels, "precision": pr, "recall": rc, "f1": f1})
        worst = per_class.sort_values("f1").head(min(3, len(per_class)))
        bullets.append(
            "Clases con **peor F1**: " +
            ", ".join([f"{r.clase} (P={r.precision:.2f}, R={r.recall:.2f}, F1={r.f1:.2f})" for _, r in worst.iterrows()])
            + "."
        )

        # 5) Errores más frecuentes (ya calculados)
        if ml.get("top_errors"):
            top = ", ".join([f"{r}→{p}:{n}" for r, p, n in ml["top_errors"][:3]])
            bullets.append(f"Confusiones dominantes (real→pred): {top}.")

        # 6) Importancias
        imp = ml["perm_importance"]
        if hasattr(imp, "empty") and not imp.empty:
            top_feats = ", ".join(f"{row.feature}({row.importance:.3f})" for _, row in imp.head(5).iterrows())
            bullets.append("Variables más influyentes (perm.): " + top_feats + ".")
            # Sugerencias de FE a partir de importancias
            bullets.append("Sugerencia: explorar interacciones y transformaciones de las variables más influyentes.")

    else:
        # 2) Métricas globales y comparación con baseline ingenuo
        rmse, mae, r2 = ml["rmse"], ml["mae"], ml["r2"]
        base = ml["rmse_baseline"]
        impv = (1.0 - (rmse / base)) * 100.0 if base and base > 0 else np.nan
        bullets.append(
            f"Rendimiento global (test): RMSE=**{rmse:.3f}**, MAE=**{mae:.3f}**, R²=**{r2:.3f}** ({r2*100:.1f}%). "
            f"Mejora vs baseline ingenuo (media): **{impv:.1f}%**."
        )

        # 3) Diagnóstico de residuales
        resid = pd.Series(ml["y_te"], dtype=float) - pd.Series(ml["pred_te"], dtype=float)
        r_mean = float(np.mean(resid))
        r_p95 = float(np.percentile(np.abs(resid), 95))
        r_skew = float(resid.skew(skipna=True))
        r_kurt = float(resid.kurtosis(skipna=True))
        # Heteroscedasticidad (aprox): correlación |resid| vs pred
        try:
            r_abs = np.abs(resid)
            r_pred = pd.Series(ml["pred_te"], dtype=float)
            corr_hr = float(np.corrcoef(r_abs.fillna(0), r_pred.fillna(0))[0, 1])
        except Exception:
            corr_hr = np.nan
        bullets.append(
            f"Residuales: media≈{r_mean:.3f}, p95(|resid|)≈{r_p95:.3f}, skew≈{r_skew:.2f}, kurtosis≈{r_kurt:.2f}, "
            f"corr(|resid|, pred)≈{corr_hr:.2f}."
        )
        if abs(r_mean) > 0.1 * rmse:
            bullets.append("Posible **sesgo** (residuo medio alejado de 0).")
        if abs(corr_hr) >= 0.3 and np.isfinite(corr_hr):
            bullets.append("Señal de **heteroscedasticidad** (|resid| correlaciona con la predicción).")

        # 4) Importancias
        imp = ml["perm_importance"]
        if hasattr(imp, "empty") and not imp.empty:
            top_feats = ", ".join(f"{row.feature}({row.importance:.3f})" for _, row in imp.head(5).iterrows())
            bullets.append("Variables más influyentes (perm.): " + top_feats + ".")
            bullets.append("Sugerencia: revisar transformaciones de las variables más influyentes (p. ej., Yeo-Johnson).")

        # 5) Interpretación objetiva (existente)
        bullets.append(ml_objective_interpretation(ml))

    return bullets


def ml_objective_interpretation(ml: Dict) -> str:
    if ml["tarea"] == "clasificación":
        acc, f1 = ml["acc_te"], ml["f1_te"]; acc_tr, f1_tr = ml["acc_tr"], ml["f1_tr"]
        gap = (acc_tr - acc) + (f1_tr - f1)
        msg = [f"El baseline de clasificación muestra Accuracy={acc:.3f} y F1-macro={f1:.3f} en test."]
        if gap > 0.15: msg.append("Indicio de sobreajuste (gap notable train-test).")
        elif gap < -0.05: msg.append("Test > Train (posible subajuste/split favorable).")
        else: msg.append("Generalización razonable (métricas similares).")
        if ml.get("top_errors"):
            r,p,n = ml["top_errors"][0]; msg.append(f"Error más frecuente: {r}→{p} (n={n}).")
        return " ".join(msg)
    rmse, r2, base = ml["rmse"], ml["r2"], ml["rmse_baseline"]
    msg = [f"Regresión: RMSE={rmse:.3f}, R²={r2:.3f}."]
    msg.append("Mejora vs baseline." if rmse < base else "No mejora al baseline; ajustar features/modelo.")
    if abs(ml["resid_mean"]) > 0.1*rmse: msg.append("Sesgo en residuales; revisar especificación.")
    return " ".join(msg)

# ml_utils.py (añadir en cualquier parte, p. ej. debajo de make_preprocessor_advanced)
CLASSIFIERS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=600),
    "RandomForest (Clas.)": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting (Clas.)": lambda: GradientBoostingClassifier(random_state=42),
    "LinearSVC": lambda: LinearSVC(),
    "KNN (Clas.)": lambda: KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": lambda: GaussianNB(),
}

REGRESSORS = {
    "Linear Regression": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(alpha=1.0),
    "Lasso": lambda: Lasso(alpha=0.001),
    "RandomForest (Regr.)": lambda: RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting (Regr.)": lambda: GradientBoostingRegressor(random_state=42),
    "SVR (RBF)": lambda: SVR(C=1.0, epsilon=0.1),
    "KNN (Regr.)": lambda: KNeighborsRegressor(n_neighbors=5),
}

def model_choices(tarea: str):
    return list(CLASSIFIERS.keys()) if tarea == "clasificación" else list(REGRESSORS.keys())

def build_model(tarea: str, name: str):
    if tarea == "clasificación":
        return CLASSIFIERS.get(name, CLASSIFIERS["Logistic Regression"])()
    return REGRESSORS.get(name, REGRESSORS["Linear Regression"])()

def ml_bullet_summary(ml: Dict) -> list[str]:
    """
    Resumen compacto en viñetas: rendimiento, variables influyentes, errores/clases/confianza.
    """
    bullets = []
    if ml["tarea"] == "clasificación":
        bullets.append(f"Rendimiento (test): Accuracy={ml['acc_te']:.3f}, F1-macro={ml['f1_te']:.3f} "
                       f"(train acc={ml['acc_tr']:.3f}, f1={ml['f1_tr']:.3f}).")
        if ml.get("top_errors"):
            top = ", ".join([f"{r}→{p}:{n}" for r, p, n in ml["top_errors"][:3]])
            bullets.append(f"Errores más frecuentes (real→pred): {top}.")
        if hasattr(ml.get("perm_importance"), "empty") and not ml["perm_importance"].empty:
            feats = ", ".join(ml["perm_importance"].head(5)["feature"].astype(str).tolist())
            bullets.append(f"Variables más influyentes (perm.): {feats}.")
    else:
        bullets.append(f"Rendimiento (test): RMSE={ml['rmse']:.3f}, MAE={ml['mae']:.3f}, R²={ml['r2']:.3f} "
                       f"(RMSE baseline={ml['rmse_baseline']:.3f}).")
        bullets.append(f"Diagnóstico de residuales: media≈{ml['resid_mean']:.3f}, p95(|resid|)≈{ml['resid_p95']:.3f}.")
        if hasattr(ml.get("perm_importance"), "empty") and not ml["perm_importance"].empty:
            feats = ", ".join(ml["perm_importance"].head(5)["feature"].astype(str).tolist())
            bullets.append(f"Variables más influyentes (perm.): {feats}.")
    # Cierra con interpretación objetiva
    bullets.append(ml_objective_interpretation(ml))
    return bullets
