# eda_plus.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN

def numeric_cols(df: pd.DataFrame):
    return list(df.select_dtypes(include=np.number).columns)

def outlier_report(df: pd.DataFrame, fence: float = 1.5) -> pd.DataFrame:
    """
    Reporte IQR + conteo outliers por columna numérica y % filas afectadas.
    """
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
    # filas con outlier en cualquier numérico (útil para samplear)
    any_outlier_mask = pd.Series(False, index=df.index)
    for c in nums:
        s = pd.to_numeric(df[c], errors="coerce")
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - fence*iqr, q3 + fence*iqr
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
    """
    Sugerencias simples según skew/kurtosis y nulos.
    """
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
    # ratios específicos si existen estas columnas
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
    """
    Devuelve (df_pca, pca, prep) o (None, None, None) si no hay suficientes numéricas.
    """
    M, prep = _prepare_numeric_matrix(df)
    if M is None or M.shape[1] < 2:
        return None, None, None
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(M)
    comp = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1]})
    return comp, pca, prep

def dbscan_outliers(df: pd.DataFrame, eps: float = 0.7, min_samples: int = 10):
    """
    Devuelve (labels, out_mask) o (None, None) si no hay numéricas.
    labels = -1 son outliers.
    """
    M, prep = _prepare_numeric_matrix(df)
    if M is None:
        return None, None
    lab = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(M)
    out_mask = (lab == -1)
    return lab, out_mask
