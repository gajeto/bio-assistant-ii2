# eda_plus.py
from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression

# -------------------- utilidades base --------------------
def numeric_cols(df: pd.DataFrame):
    return list(df.select_dtypes(include=np.number).columns)

# -------------------- outliers / sesgo --------------------
def outlier_report(df: pd.DataFrame, fence: float = 1.5) -> tuple[pd.DataFrame, pd.Series]:
    nums = numeric_cols(df)
    rows = []
    for c in nums:
        s = pd.to_numeric(df[c], errors="coerce")
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - fence*iqr, q3 + fence*iqr
        iqr_mask = (s < low) | (s > high)
        z = (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1)
        z3_mask = z.abs() > 3
        rows.append({
            "columna": c,
            "q1": q1, "q3": q3, "iqr": iqr,
            "fence_low": low, "fence_high": high,
            "outliers_iqr": int(iqr_mask.sum()),
            "outliers_z3": int(z3_mask.sum()),
            "pct_outliers": round(100 * iqr_mask.sum() / len(s.dropna()) if len(s.dropna()) else 0, 2),
            "skew": s.skew(skipna=True),
            "kurtosis": s.kurtosis(skipna=True),
        })
    rep = pd.DataFrame(rows).sort_values("pct_outliers", ascending=False)
    any_outlier_mask = pd.Series(False, index=df.index)
    for c in nums:
        s = pd.to_numeric(df[c], errors="coerce")
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
        any_outlier_mask = any_outlier_mask | ((s < low) | (s > high))
    return rep, any_outlier_mask

def skew_kurtosis_table(df: pd.DataFrame) -> pd.DataFrame:
    nums = numeric_cols(df)
    data = []
    for c in nums:
        s = pd.to_numeric(df[c], errors="coerce")
        data.append({"columna": c, "skew": s.skew(skipna=True), "kurtosis": s.kurtosis(skipna=True)})
    return pd.DataFrame(data).sort_values("skew", ascending=False)

def suggest_transforms(df: pd.DataFrame, skew_thr: float = 1.0) -> list[str]:
    sug = []
    nums = numeric_cols(df)
    if not nums:
        return ["No hay columnas numéricas para sugerir transformaciones."]
    sk = skew_kurtosis_table(df)
    skewed = sk[sk["skew"].abs() >= skew_thr]["columna"].tolist()
    if skewed:
        sug.append("Aplicar Yeo-Johnson/Log1p a columnas sesgadas: " + ", ".join(skewed[:8]))
    nulls = df.isna().mean().sort_values(ascending=False)
    high_null = nulls[nulls > 0.05]
    if len(high_null) > 0:
        sug.append("Imputación robusta (mediana/moda) en: " + ", ".join([f"{c} ({p*100:.1f}%)" for c, p in high_null.head(8).items()]))
    cols = df.columns
    if {"cobertura", "edad"}.issubset(cols):
        sug.append("Nueva feature ratio: cobertura_por_edad = cobertura / (edad + 1).")
    if {"frecuencia_alelo", "cobertura"}.issubset(cols):
        sug.append("Nueva feature interacción: fa_x_cobertura = frecuencia_alelo * cobertura.")
    return sug

def find_top_corr_pair(df: pd.DataFrame) -> tuple[str, str] | None:
    nums = numeric_cols(df)
    if len(nums) < 2:
        return None
    corr = df[nums].corr(numeric_only=True).abs()
    np.fill_diagonal(corr.values, np.nan)
    idx = np.nanargmax(corr.values)
    r, c = np.unravel_index(idx, corr.shape)
    return nums[r], nums[c]

# -------------------- PCA / DBSCAN --------------------
def _prepare_numeric_matrix(df: pd.DataFrame):
    X_num = df.select_dtypes(include=np.number)
    if X_num.shape[1] == 0:
        return None, None
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    M = pipe.fit_transform(X_num)
    return M, pipe

def pca_2d(df: pd.DataFrame):
    M, prep = _prepare_numeric_matrix(df)
    if M is None or M.shape[1] < 2:
        return None, None, None
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(M)
    comp = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1]})
    return comp, pca, prep

def dbscan_outliers(df: pd.DataFrame, eps: float = 0.7, min_samples: int = 10):
    M, prep = _prepare_numeric_matrix(df)
    if M is None:
        return None, None
    lab = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(M)
    out_mask = (lab == -1)
    return lab, out_mask

# -------------------- Multicolinealidad (VIF + Nº condición) --------------------
def compute_vif_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    VIF por columna numérica usando regresión lineal con sklearn.
    VIF = 1 / (1 - R^2) regrediendo cada variable contra el resto.
    """
    nums = numeric_cols(df)
    if len(nums) < 2:
        return pd.DataFrame(columns=["feature", "vif", "r2_aux"])
    X = df[nums].copy()
    # imputación y estandarización
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_imp = imp.fit_transform(X)
    X_std = sc.fit_transform(X_imp)
    res = []
    for i, name in enumerate(nums):
        y = X_std[:, i]
        Xr = np.delete(X_std, i, axis=1)
        if Xr.shape[1] == 0:
            vif = 1.0; r2 = 0.0
        else:
            try:
                lr = LinearRegression()
                lr.fit(Xr, y)
                r2 = float(lr.score(Xr, y))
                r2 = min(max(r2, 0.0), 0.999999999)  # clamp para evitar división por 0
                vif = float(1.0 / (1.0 - r2))
            except Exception:
                r2 = np.nan; vif = np.inf
        res.append({"feature": name, "vif": vif, "r2_aux": r2})
    vif_df = pd.DataFrame(res).sort_values("vif", ascending=False)
    return vif_df

def condition_number(df: pd.DataFrame) -> float | None:
    """
    Número de condición de la matriz estandarizada (mayor → más multicolinealidad).
    """
    nums = numeric_cols(df)
    if len(nums) < 2:
        return None
    X = df[nums].copy()
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_std = sc.fit_transform(imp.fit_transform(X))
    try:
        cn = float(np.linalg.cond(X_std))
    except Exception:
        cn = None
    return cn
