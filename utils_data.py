# utils_data.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

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

def eda_summary_text(eda: Dict, df: pd.DataFrame, max_lines=80) -> str:
    lines = []
    lines.append(f"Forma: {eda['shape'][0]} filas x {eda['shape'][1]} columnas.")
    lines.append("Tipos: " + ", ".join(f"{k}:{v}" for k,v in eda["dtypes"].items()))
    lines.append("Faltantes %: " + ", ".join(f"{k}:{v}%" for k,v in eda["null_pct"].items()))
    if not eda["desc_num"].empty:
        lines.append("Resumen numérico (5 primeras):")
        for c in list(eda["desc_num"].index)[:5]:
            d = eda["desc_num"].loc[c]
            lines.append(f"- {c}: media={d['mean']:.4f}, std={d['std']:.4f}, "
                         f"min={d['min']:.4f}, mediana={d['50%']:.4f}, max={d['max']:.4f}")
    return "\n".join(lines[:max_lines])

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
