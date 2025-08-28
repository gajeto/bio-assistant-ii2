# utils_data.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
# importa utilidades avanzadas del EDA
from eda import (
    outlier_report, skew_kurtosis_table, compute_vif_table,
    condition_number, find_top_corr_pair
)


def load_csv(file) -> pd.DataFrame:
    try:
        return pd.read_csv(file)
    except Exception:
        file.seek(0)
        return pd.read_csv(file, sep=";")

def cap_rows(df: pd.DataFrame, max_rows: int = 1000) -> pd.DataFrame:
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return df.reset_index(drop=True)

def demo_data(n=1000, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genes = ["BRCA1","BRCA2","TP53","APOE","CFTR","HBB"]
    impacts = ["sinónima","missense","nonsense","frameshift"]
    zyg = ["het","hom"]
    df = pd.DataFrame({
        "id_muestra": [f"S{10000+i}" for i in range(n)],
        "gen": rng.choice(genes, size=n),
        "impacto": rng.choice(impacts, size=n, p=[0.45,0.4,0.1,0.05]),
        "cigocidad": rng.choice(zyg, size=n, p=[0.8,0.2]),
        "frecuencia_alelo": np.clip(rng.normal(0.02, 0.015, size=n), 0, 0.3),
        "cobertura": np.clip(rng.normal(120, 25, size=n), 20, 300).round(0),
        "edad": np.clip(rng.normal(45, 16, size=n), 0, 95).round(0),
        "sexo": rng.choice(["F","M"], size=n, p=[0.52,0.48]),
    })
    df["etiqueta"] = np.where(
        (df["impacto"].isin(["missense","nonsense","frameshift"])) &
        (df["frecuencia_alelo"] < 0.01) &
        (df["cobertura"] >= 60),
        "prob_patogénico", "otro"
    )
    miss_idx = rng.choice(n, size=int(0.02*n), replace=False)
    df.loc[miss_idx, "frecuencia_alelo"] = np.nan
    return df

def basic_eda(df: pd.DataFrame) -> Dict:
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isna().sum().to_dict(),
        "null_pct": (df.isna().mean()*100).round(2).to_dict(),
        "nunique": df.nunique(dropna=False).to_dict(),
        "desc_num": df.select_dtypes(include=np.number).describe().T,
        "desc_cat": df.select_dtypes(exclude=np.number).describe(include="all").T
    }

def eda_summary_text(eda: dict, df: pd.DataFrame, target_hint: Optional[str] = None) -> str:
    """
    Resumen rico y compacto del EDA para inyectar en el prompt del LLM.
    Incluye: tamaño, tipos, faltantes (top), duplicados, cardinalidades,
    variables near-constant, outliers (IQR), skew/kurtosis, correlaciones fuertes,
    multicolinealidad (VIF/condición), distribución de posible target y
    recomendaciones de preprocesamiento/feature engineering.
    """
    if df is None or df.empty:
        return "EDA: dataset vacío."

    lines: list[str] = []

    # 1) Tamaño y tipos
    n_rows, n_cols = df.shape
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    lines.append(f"Dataset: **{n_rows} filas × {n_cols} columnas**.")
    lines.append(f"Tipos: numéricas={len(num_cols)}, categóricas={len(cat_cols)}, fecha/hora={len(dt_cols)}, booleanas={len(bool_cols)}.")

    # 2) Faltantes y duplicados
    miss_pct = df.isna().mean().mul(100).sort_values(ascending=False)
    total_miss = float(miss_pct.mean())
    top_miss = miss_pct[miss_pct > 0].head(5)
    lines.append(f"Faltantes promedio: **{total_miss:.1f}%**.")
    if not top_miss.empty:
        lines.append("Top faltantes: " + ", ".join([f"{c}({p:.1f}%)" for c, p in top_miss.items()]) + ".")
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        lines.append(f"Filas duplicadas: **{dup_count}**.")

    # 3) Cardinalidad y near-constant
    nun = df.nunique(dropna=False)
    high_card = [c for c in cat_cols if nun[c] > min(50, 0.05 * n_rows)]
    low_var = []
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.nunique(dropna=True) <= 2 or (s.std(skipna=True) == 0):
            low_var.append(c)
    if high_card:
        lines.append("Alta cardinalidad (categ.): " + ", ".join(high_card[:8]) + ".")
    if low_var:
        lines.append("Variables casi constantes: " + ", ".join(low_var[:8]) + ".")

    # 4) Outliers y sesgo
    try:
        rep_out, any_out_mask = outlier_report(df)
        if not rep_out.empty and rep_out["pct_outliers"].max() > 0:
            top_out = rep_out.head(5)[["columna", "pct_outliers"]].values
            lines.append("Outliers (IQR) top: " + ", ".join([f"{c}({pct:.1f}%)" for c, pct in top_out]) + ".")
            any_rows = int(any_out_mask.sum())
            if any_rows > 0:
                lines.append(f"Filas con ≥1 outlier: **{any_rows}** ({any_rows/n_rows*100:.1f}%).")
    except Exception:
        pass

    try:
        sk = skew_kurtosis_table(df)
        skewed = sk.loc[sk["skew"].abs() >= 1.0, "columna"].tolist()
        if skewed:
            lines.append("Sesgo alto (|skew|≥1): " + ", ".join(skewed[:8]) + ".")
    except Exception:
        pass

    # 5) Correlaciones fuertes
    strong_corrs = []
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True).abs()
        np.fill_diagonal(corr.values, np.nan)
        # top 3 pares
        flat = [
            (num_cols[i], num_cols[j], corr.iloc[i, j])
            for i in range(len(num_cols)) for j in range(len(num_cols))
            if i < j
        ]
        flat = [t for t in flat if pd.notna(t[2])]
        flat.sort(key=lambda x: x[2], reverse=True)
        strong_corrs = flat[:3]
        if strong_corrs:
            lines.append("Correlaciones fuertes: " + ", ".join([f"{a}~{b}(|r|≈{r:.2f})" for a, b, r in strong_corrs]) + ".")

    # 6) Multicolinealidad (VIF + nº de condición)
    try:
        vif_df = compute_vif_table(df)
        if not vif_df.empty and (vif_df["vif"] > 10).any():
            high_vif = vif_df[vif_df["vif"] > 10].head(5)
            lines.append("VIF alto (>10): " + ", ".join([f"{r.feature}({r.vif:.1f})" for _, r in high_vif.iterrows()]) + ".")
        cn = condition_number(df)
        if cn is not None:
            lines.append(f"Número de condición: **{cn:.1f}** (>30 sugiere multicolinealidad).")
    except Exception:
        pass

    # 7) Posible target: distribución/resumen (heurística)
    tgt = None
    if target_hint and target_hint in df.columns:
        tgt = target_hint
    else:
        candidates = [c for c in df.columns if c.lower() in ("label", "etiqueta", "target")]
        if candidates:
            tgt = candidates[0]
    if tgt:
        y = df[tgt]
        if (y.dtype.kind in "ifu" and y.nunique(dropna=True) <= 10) or (y.dtype.kind not in "ifu"):
            # tratar como clasificación
            dist = y.astype(str).value_counts(normalize=True)
            maj = dist.idxmax(); p = float(dist.max())
            lines.append(f"Target detectado: **{tgt}** (clasificación). Mayoritaria={maj} ({p*100:.1f}%).")
            if p >= 0.7:
                lines.append("Desbalance alto; usar estratificación y métricas macro (p. ej., F1-macro).")
        else:
            # tratar como regresión
            s = pd.to_numeric(y, errors="coerce")
            lines.append(
                f"Target detectado: **{tgt}** (regresión). Rango≈[{np.nanmin(s):.3f}, {np.nanmax(s):.3f}], "
                f"media={np.nanmean(s):.3f}, desv={np.nanstd(s):.3f}."
            )

    # 8) Recomendaciones de preprocesamiento/FE
    recs = []
    # Faltantes
    hi_miss = miss_pct[miss_pct >= 5].sort_values(ascending=False).head(8)
    if not hi_miss.empty:
        recs.append("Imputación robusta (mediana/moda) en: " + ", ".join([f"{c}({p:.1f}%)" for c, p in hi_miss.items()]) + ".")
    # Skew / transformaciones
    try:
        if skewed:
            recs.append("Aplicar transformaciones para sesgo (p. ej., **Yeo-Johnson/Log1p**) en variables sesgadas.")
    except Exception:
        pass
    # OneHot con min_frequency para alta cardinalidad
    if high_card:
        recs.append("Codificación **OneHot** con `min_frequency` para categorías raras en variables de alta cardinalidad.")
    # Escalado
    if len(num_cols) > 0:
        recs.append("Estandarizar numéricas (**RobustScaler/StandardScaler**) antes de modelos sensibles a escala.")
    # Interacciones/ratios a partir de correlaciones
    if strong_corrs:
        a, b, _ = strong_corrs[0]
        recs.append(f"Probar interacción **{a}×{b}** o ratios relacionados (si tienen sentido).")
    # Near-constant
    if low_var:
        recs.append("Eliminar o consolidar variables casi constantes (sin poder predictivo).")

    if recs:
        lines.append("Recomendaciones: " + " ".join(recs))

    # 9) Cierre
    return "\n".join(lines)


def eda_summary_markdown(eda: Dict) -> str:
    top_missing = sorted(eda["null_pct"].items(), key=lambda x: x[1], reverse=True)[:10]
    md = []
    md.append("# Resumen EDA\n")
    md.append(f"- **Filas**: {eda['shape'][0]}  \n- **Columnas**: {eda['shape'][1]}")
    md.append("## Tipos de datos")
    md.append(", ".join(f"`{k}`: {v}" for k,v in eda["dtypes"].items()))
    md.append("## % Faltantes (Top 10)")
    for k,v in top_missing:
        md.append(f"- {k}: {v}%")
    if not eda["desc_num"].empty:
        md.append("\n## Descriptivos numéricos (primeras 5 variables)")
        for c in list(eda["desc_num"].index)[:5]:
            d = eda["desc_num"].loc[c]
            md.append(f"- **{c}** → media {d['mean']:.4f} | std {d['std']:.4f} | "
                      f"min {d['min']:.4f} | 50% {d['50%']:.4f} | max {d['max']:.4f}")
    return "\n".join(md)

def eda_key_insights(eda: Dict, df: pd.DataFrame, target_guess: Optional[str]=None, max_corr_pairs: int = 3) -> list[str]:
    bullets = []
    nrows, ncols = eda["shape"]
    bullets.append(f"Dataset con {nrows} filas y {ncols} columnas.")
    missing = pd.Series(eda["null_pct"]).sort_values(ascending=False)
    top_missing = missing[missing > 0].head(3)
    if not top_missing.empty:
        bullets.append("Mayor % de faltantes: " + ", ".join(f"{k} ({v}%)" for k, v in top_missing.items()))
    if target_guess and target_guess in df.columns:
        vc = df[target_guess].value_counts(dropna=False)
        if len(vc) > 0:
            total = len(df)
            parts = [f"{k}: {v} ({(v/total)*100:.1f}%)" for k, v in vc.head(4).items()]
            bullets.append(f"Distribución de {target_guess}: " + ", ".join(parts))
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True).abs()
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        corr_upper = corr.where(mask)
        pairs = corr_upper.unstack().dropna().sort_values(ascending=False).head(max_corr_pairs)
        if not pairs.empty:
            bullets.append("Correlaciones fuertes: " + ", ".join(f"{a}~{b}={v:.2f}" for (a, b), v in pairs.items()))
    return bullets
