# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.inspection import permutation_importance

# Plotly (gráficas interactivas)
try:
    from plotly import express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception as e:
    PLOTLY_OK = False
    PX_ERR = e

# GROQ SDK
try:
    from groq import Groq
    GROQ_OK = True
except Exception:
    GROQ_OK = False

# ===================== Configuración de página =====================
st.set_page_config(page_title="Asistente Genético (EDA + ML + Groq)", page_icon="🧬", layout="wide")
st.title("🧬 Asistente Genético: EDA + ML + Chat (Groq)")
st.caption("Sube un CSV (hasta ~1.000 filas) → EDA → Baseline ML → Chat en español con Groq. "
           "Herramienta educativa; **no** es consejo médico.")

# ===================== Catálogo de modelos (Groq) =====================
GROQ_MODEL_CATALOG = {
    "Llama 3.1 8B (rápido)": "llama-3.1-8b-instant",
    "Llama 3.1 70B (calidad)": "llama-3.1-70b-versatile",
    "Llama 3 (8B) 8192 ctx": "llama3-8b-8192",
    "Mixtral 8x7B 32k": "mixtral-8x7b-32768",
}

# ===================== Utilidades de datos =====================
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

def eda_summary_text(eda: Dict, df: pd.DataFrame, max_lines=80) -> str:
    lines = []
    lines.append(f"Forma: {eda['shape'][0]} filas x {eda['shape'][1]} columnas.")
    lines.append("Tipos: " + ", ".join(f"{k}:{v}" for k,v in eda["dtypes"].items()))
    lines.append("Faltantes %: " + ", ".join(f"{k}:{v}%" for k,v in eda["null_pct"].items()))
    if not eda["desc_num"].empty:
        lines.append("Resumen numérico (5 primeras):")
        for c in list(eda["desc_num"].index)[:5]:
            d = eda["desc_num"].loc[c]
            lines.append(f"- {c}: media={d['mean']:.4f}, std={d['std']:.4f}, min={d['min']:.4f}, mediana={d['50%']:.4f}, max={d['max']:.4f}")
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
            md.append(f"- **{c}** → media {d['mean']:.4f} | std {d['std']:.4f} | min {d['min']:.4f} | 50% {d['50%']:.4f} | max {d['max']:.4f}")
    return "\n".join(md)

def eda_key_insights(eda: Dict, df: pd.DataFrame, target_guess: Optional[str]=None, max_corr_pairs: int = 3) -> List[str]:
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

# ===================== ML: métricas, importancias, interpretación =====================
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
        # Errores frecuentes (pares real→pred)
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
        # baseline ingenuo (media entreno)
        baseline = np.full_like(y_te, fill_value=np.mean(y_tr), dtype=float)
        out["rmse_baseline"] = mean_squared_error(y_te, baseline, squared=False)
        out["y_te"] = y_te
        out["pred_te"] = pred_te
        resid = y_te - pred_te
        out["resid_mean"] = float(np.mean(resid))
        out["resid_p95"] = float(np.percentile(np.abs(resid), 95))

    # Importancias por permutación (en espacio original)
    out["perm_importance"] = compute_perm_importance(pipe, X_te, y_te, tarea, topk=8)
    return out

def ml_key_insights(ml: Dict, X_te: pd.DataFrame) -> List[str]:
    bullets = []
    ntest = len(ml["y_te"])
    bullets.append(f"Evaluación en test con {ntest} muestras.")

    if ml["tarea"] == "clasificación":
        bullets.append(f"Accuracy={ml['acc_te']:.3f}, F1-macro={ml['f1_te']:.3f}. "
                       f"(train acc={ml['acc_tr']:.3f}, f1={ml['f1_tr']:.3f})")
        # Balance de clases reales
        dist = pd.Series(ml["y_te"]).astype(str).value_counts(normalize=True)
        maj = dist.idxmax()
        bullets.append(f"Clase mayoritaria en test: {maj} ({dist.max()*100:.1f}%).")
        # Errores frecuentes
        if ml.get("top_errors"):
            top = ", ".join([f"{r}→{p}:{n}" for r,p,n in ml["top_errors"][:3]])
            bullets.append(f"Errores frecuentes (real→pred): {top}.")
    else:
        bullets.append(f"MAE={ml['mae']:.3f}, RMSE={ml['rmse']:.3f}, R²={ml['r2']:.3f}. "
                       f"RMSE baseline(media)={ml['rmse_baseline']:.3f}.")

        # Resumen de residuales
        bullets.append(f"Residuo medio ≈ {ml['resid_mean']:.3f}; p95(|resid|) ≈ {ml['resid_p95']:.3f}.")

    # Importancias
    imp = ml["perm_importance"]
    if not imp.empty:
        top_feats = ", ".join(f"{row.feature}({row.importance:.3f})" for _, row in imp.head(5).iterrows())
        bullets.append("Features más influyentes (perm.): " + top_feats)
    return bullets

def ml_objective_interpretation(ml: Dict) -> str:
    """Texto breve, determinístico, en español, basado solo en métricas/artefactos."""
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
            msg.append(f"Error más frecuente: {r}→{p} (conteo={n}). Conviene revisar separación entre estas clases.")
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

# ===================== LLM Groq =====================
def llm_status_badge(model_id: str, temp: float, max_tokens: int, key_ok: bool) -> str:
    if key_ok and GROQ_OK:
        return f"🟢 **Groq conectado** — modelo: `{model_id}` · temp={temp} · máx_tokens={max_tokens}"
    elif not GROQ_OK:
        return "🔴 Groq SDK no instalado. Agrega `groq` a requirements.txt."
    return "🟠 Falta `GROQ_API_KEY` en *Secrets*."

class GroqLLM:
    """Cliente Groq con modo normal y streaming."""
    def __init__(self, model_id: str, api_key: Optional[str], temp: float = 0.2, max_tokens: int = 350):
        self.model = model_id
        self.key = api_key
        self.temp = float(temp)
        self.max_tokens = int(max_tokens)

    def available(self) -> bool:
        return bool(self.key) and GROQ_OK

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        if not self.available():
            return "⚠️ Groq: configura `GROQ_API_KEY` en Secrets."
        try:
            client = Groq(api_key=self.key)
            rsp = client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":system_prompt},
                          {"role":"user","content":user_prompt}],
                temperature=self.temp,
                max_completion_tokens=self.max_tokens,
            )
            return (rsp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"⚠️ Error Groq API: {e}"

    def ask_stream(self, system_prompt: str, user_prompt: str):
        """Genera texto incremental (streaming)."""
        if not self.available():
            yield "⚠️ Groq: configura `GROQ_API_KEY` en Secrets."
            return
        try:
            client = Groq(api_key=self.key)
            stream = client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":system_prompt},
                          {"role":"user","content":user_prompt}],
                temperature=self.temp,
                max_completion_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    yield delta
        except Exception as e:
            yield f"⚠️ Error Groq API (stream): {e}"

# ===================== Sidebar (carga, apariencia, Groq) =====================
with st.sidebar:
    st.header("1) Cargar datos")
    up = st.file_uploader("Sube un CSV", type=["csv"])
    usar_demo = st.checkbox("Usar datos demo (sintéticos)", value=not bool(up))

    st.header("2) Apariencia")
    tema_oscuro = st.toggle("Modo oscuro (gráficas)", value=False)
    PLOTLY_TEMPLATE = "plotly_dark" if tema_oscuro else "plotly"

    st.header("3) LLM (Groq)")
    modelo_key = st.selectbox("Modelo (Groq):", options=list(GROQ_MODEL_CATALOG.keys()), index=0)
    modelo_id = GROQ_MODEL_CATALOG[modelo_key]
    custom_id = st.text_input("Modelo personalizado (opcional)", value="")
    if custom_id.strip():
        modelo_id = custom_id.strip()
    st.session_state["llm_model_id"] = modelo_id
    st.caption(f"Usando: `{modelo_id}`")

    st.markdown("**Parámetros**")
    st.session_state["llm_temp"] = st.slider("Temperatura", 0.0, 1.0, st.session_state.get("llm_temp", 0.2), 0.05)
    st.session_state["llm_max_tokens"] = st.slider("Máx. tokens de salida", 50, 800, st.session_state.get("llm_max_tokens", 350), 10)

# ===================== Cargar datos =====================
if up is not None:
    df = load_csv(up)
elif usar_demo:
    df = demo_data(1000)
else:
    st.info("Sube un CSV o activa el demo en la barra lateral.")
    st.stop()

df = cap_rows(df, 1000)
st.success(f"Datos cargados: {df.shape[0]} filas × {df.shape[1]} columnas")
st.dataframe(df.head(), use_container_width=True)

# ===================== EDA base =====================
eda = basic_eda(df)

# ===================== Pestañas =====================
tab_eda, tab_ml, tab_chat, tab_demo, tab_export = st.tabs(["📊 EDA", "🤖 ML", "💬 Chat", "🔬 Demo EDA→ML→LLM", "📥 Exportar"])

# ------------------------------- EDA TAB -------------------------------
with tab_eda:
    st.subheader("Exploración de datos")
    c1, c2 = st.columns(2)
    miss_tbl = pd.DataFrame({
        "columna": list(eda["null_counts"].keys()),
        "faltantes": list(eda["null_counts"].values()),
        "faltantes_%": list(eda["null_pct"].values())
    }).sort_values("faltantes_%", ascending=False)
    dtype_tbl = pd.DataFrame({
        "columna": list(eda["dtypes"].keys()),
        "tipo": list(eda["dtypes"].values()),
        "nunique": list(eda["nunique"].values())
    }).sort_values("nunique", ascending=False)

    with c1:
        st.write("**Faltantes por columna**")
        st.dataframe(miss_tbl, use_container_width=True, height=260)
        if PLOTLY_OK:
            st.plotly_chart(px.bar(miss_tbl, x="columna", y="faltantes_%", title="% faltantes", template=PLOTLY_TEMPLATE), use_container_width=True)
        else:
            st.warning(f"Plotly no disponible: {PX_ERR}")

    with c2:
        st.write("**Tipos y cardinalidades**")
        st.dataframe(dtype_tbl, use_container_width=True, height=260)

    st.write("**Descriptivos numéricos**")
    num_desc = eda["desc_num"]
    if not num_desc.empty:
        st.dataframe(num_desc, use_container_width=True, height=260)
    else:
        st.write("No hay columnas numéricas.")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    with st.expander("Distribuciones numéricas (hasta 6) con box marginal"):
        if PLOTLY_OK:
            for c in num_cols[:6]:
                st.plotly_chart(px.histogram(df, x=c, nbins=40, title=c, marginal="box", template=PLOTLY_TEMPLATE), use_container_width=True)

    with st.expander("Frecuencias categóricas (hasta 6)"):
        if PLOTLY_OK:
            for c in cat_cols[:6]:
                vc = df[c].astype(str).value_counts(dropna=False).head(20).reset_index()
                vc.columns = [c, "conteo"]
                st.plotly_chart(px.bar(vc, x=c, y="conteo", title=c, template=PLOTLY_TEMPLATE), use_container_width=True)

    if len(num_cols) >= 2 and PLOTLY_OK:
        st.subheader("Matriz de correlación (subset numérico)")
        corr = df[num_cols[:12]].corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr, text_auto=False, aspect="auto", title="Correlaciones", template=PLOTLY_TEMPLATE), use_container_width=True)

    posibles_targets = [c for c in df.columns if c.lower() in ("etiqueta","label","target")]
    target_sugerido = posibles_targets[0] if posibles_targets else None
    st.subheader("Distribuciones por clase (opcional)")
    col_tar = st.selectbox("Columna de etiqueta (opcional)", options=["(ninguna)"] + list(df.columns),
                           index=(list(df.columns).index(target_sugerido)+1 if target_sugerido in df.columns else 0))
    if col_tar != "(ninguna)" and PLOTLY_OK:
        bal = df[col_tar].astype(str).value_counts(dropna=False).reset_index()
        bal.columns = [col_tar, "conteo"]
        st.plotly_chart(px.bar(bal, x=col_tar, y="conteo", title="Balance de clases", template=PLOTLY_TEMPLATE), use_container_width=True)
        for c in num_cols[:3]:
            st.plotly_chart(px.violin(df, x=col_tar, y=c, box=True, points="outliers", title=f"{c} por {col_tar}", template=PLOTLY_TEMPLATE), use_container_width=True)
        if len(num_cols) >= 2:
            st.plotly_chart(px.scatter(df, x=num_cols[0], y=num_cols[1], color=col_tar, title=f"{num_cols[0]} vs {num_cols[1]} por {col_tar}", template=PLOTLY_TEMPLATE), use_container_width=True)

# ------------------------------- ML TAB -------------------------------
with tab_ml:
    st.subheader("Baseline ML mínimo")
    posibles_targets = [c for c in df.columns if c.lower() in ("etiqueta","label","target")]
    target_sugerido = posibles_targets[0] if posibles_targets else df.columns[0]
    target = st.selectbox("Variable objetivo", options=df.columns,
                          index=(list(df.columns).index(target_sugerido) if target_sugerido in df.columns else 0))
    y = df[target]
    X = df.drop(columns=[target])
    tarea = infer_task(y)
    st.write(f"Tarea detectada: **{tarea}**")

    pre = make_preprocessor(X)
    modelo = LogisticRegression(max_iter=600) if tarea == "clasificación" else LinearRegression()
    pipe = Pipeline([("prep", pre), ("model", modelo)])

    estrat = y if (tarea == "clasificación" and y.nunique() >= 2) else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=estrat)

    if st.button("Entrenar baseline"):
        ml_art = ml_metrics_and_artifacts(pipe, X_tr, X_te, y_tr, y_te, tarea)
        st.session_state["ml"] = {
            "pipeline": pipe,
            "X_te": X_te,
            "y_te": y_te,
            **ml_art
        }

    # Mostrar resultados si existen
    if "ml" in st.session_state and st.session_state["ml"]["tarea"] == tarea:
        ml = st.session_state["ml"]
        if tarea == "clasificación":
            st.write(f"**Accuracy (test):** {ml['acc_te']:.3f} | **F1-macro (test):** {ml['f1_te']:.3f} "
                     f"(train: acc={ml['acc_tr']:.3f}, f1={ml['f1_tr']:.3f})")

            if PLOTLY_OK:
                etiquetas = ml["labels"]
                cm = ml["cm"]
                fig_cm = px.imshow(cm, x=etiquetas, y=etiquetas, text_auto=True, color_continuous_scale="Blues",
                                   title="Matriz de confusión", template=PLOTLY_TEMPLATE)
                fig_cm.update_layout(xaxis_title="Predicha", yaxis_title="Real")
                st.plotly_chart(fig_cm, use_container_width=True)

        else:
            st.write(f"**MAE:** {ml['mae']:.3f} | **RMSE:** {ml['rmse']:.3f} | **R²:** {ml['r2']:.3f} "
                     f"| **RMSE baseline(media):** {ml['rmse_baseline']:.3f}")

            if PLOTLY_OK:
                resid = ml["y_te"] - ml["pred_te"]
                st.plotly_chart(px.scatter(x=ml["pred_te"], y=resid, labels={"x":"Predicción","y":"Residual"},
                                           title="Predicción vs Residual", template=PLOTLY_TEMPLATE), use_container_width=True)
                st.plotly_chart(px.histogram(resid, nbins=40, title="Histograma de residuales",
                                             template=PLOTLY_TEMPLATE), use_container_width=True)

        # Importancias (perm)
        imp = ml["perm_importance"]
        st.write("**Importancias por permutación (top):**")
        st.dataframe(imp, use_container_width=True, height=240)
        if PLOTLY_OK and not imp.empty:
            st.plotly_chart(px.bar(imp, x="feature", y="importance", error_y="std", title="Importancia (perm)",
                                   template=PLOTLY_TEMPLATE), use_container_width=True)

        st.info(ml_objective_interpretation(ml))

# ------------------------------- CHAT TAB (Groq con streaming) -------------------------------
with tab_chat:
    st.subheader("Chat en español (Groq)")

    model_id = st.session_state.get("llm_model_id") or "llama-3.1-8b-instant"
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    temp = float(st.session_state.get("llm_temp", 0.2))
    max_tokens = int(st.session_state.get("llm_max_tokens", 350))
    groq = GroqLLM(model_id=model_id, api_key=api_key, temp=temp, max_tokens=max_tokens)

    st.info(llm_status_badge(model_id, temp, max_tokens, key_ok=bool(api_key)))

    if "mensajes" not in st.session_state:
        st.session_state.mensajes = [{"role":"assistant",
                                      "content":"Hola 👋 Puedo responder en español sobre tu dataset, el EDA y el baseline ML. ¿Qué quieres saber?"}]
    for m in st.session_state.mensajes:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Construir contexto con EDA + (si existe) resumen ML
    eda_resumen = eda_summary_text(eda, df)
    ml_text = ""
    if "ml" in st.session_state:
        ml = st.session_state["ml"]
        ml_ins = ml_key_insights(ml, st.session_state["ml"]["X_te"])
        ml_text = "INSIGHTS DE ML:\n" + "\n".join(f"- {b}" for b in ml_ins) + "\n"

    sistema = (
        "Eres un asistente de datos genéticos. Responde en **español**, breve y conservador. "
        "No des consejo médico. Cita cifras solo si están en el contexto.\n\n"
        f"RESUMEN EDA:\n{eda_resumen}\n\n{ml_text}"
    )

    pregunta = st.chat_input("Escribe tu pregunta…")
    if pregunta:
        st.session_state.mensajes.append({"role":"user", "content":pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        # Streaming
        full = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for delta in groq.ask_stream(sistema, pregunta):
                full += delta
                placeholder.markdown(full)
        st.session_state.mensajes.append({"role":"assistant", "content":full})

# ------------------------------- DEMO TAB (EDA→ML→LLM) -------------------------------
with tab_demo:
    st.subheader("Demostración: ¿el LLM usa los insights del EDA y del ML?")

    # 1) Insights EDA
    posibles_targets = [c for c in df.columns if c.lower() in ("etiqueta","label","target")]
    target_sugerido = posibles_targets[0] if posibles_targets else None
    target_for_insights = st.selectbox(
        "Columna de etiqueta (opcional, para enriquecer insights EDA):",
        options=["(ninguna)"] + list(df.columns),
        index=(list(df.columns).index(target_sugerido)+1 if target_sugerido in df.columns else 0)
    )
    if target_for_insights == "(ninguna)":
        target_for_insights = None
    insights_eda = eda_key_insights(eda, df, target_guess=target_for_insights)
    st.markdown("**Insights del EDA:**")
    st.markdown("\n".join(f"- {b}" for b in insights_eda))

    # 2) Insights ML (si ya entrenaste)
    ml_insights = []
    ml_interp_txt = ""
    if "ml" in st.session_state:
        ml = st.session_state["ml"]
        ml_insights = ml_key_insights(ml, ml["X_te"])
        ml_interp_txt = ml_objective_interpretation(ml)
        st.markdown("**Insights del baseline ML:**")
        st.markdown("\n".join(f"- {b}" for b in ml_insights))
        st.success(f"**Interpretación objetiva (automática):** {ml_interp_txt}")
    else:
        st.warning("Entrena el baseline en la pestaña **ML** para habilitar insights de ML en esta demo.")

    st.divider()

    # 3) Pregunta de prueba y prompts
    pregunta_demo = st.text_input(
        "Pregunta para comparar respuestas (ideal: algo que se beneficie de EDA y ML):",
        value="¿Qué problemas de calidad ves, qué variables parecen influyentes y cómo interpretarías el rendimiento del modelo?"
    )

    sistema_sin = (
        "Eres un asistente de datos genéticos. Responde en español, breve y conservador. "
        "Si te falta contexto, dilo explícitamente."
    )
    sistema_con_eda = (
        "Eres un asistente de datos genéticos. Responde en español, breve y conservador. "
        "Usa EXCLUSIVAMENTE los INSIGHTS EDA para cifras del dataset. Si falta, dilo.\n\n"
        "INSIGHTS EDA:\n" + "\n".join(f"- {b}" for b in insights_eda)
    )
    sistema_eda_ml = (
        "Eres un asistente de datos genéticos. Responde en español, breve y conservador. "
        "Cuando cites cifras del dataset o del modelo, usa EXCLUSIVAMENTE los bloques de INSIGHTS. "
        "Incluye una interpretación del rendimiento del modelo basada en dichas métricas.\n\n"
        "INSIGHTS EDA:\n" + "\n".join(f"- {b}" for b in insights_eda) + "\n\n" +
        ("INSIGHTS ML:\n" + "\n".join(f"- {b}" for b in ml_insights) + "\nInterpretación ML (objetiva): " + ml_interp_txt
         if ml_insights else "INSIGHTS ML: (no disponibles)")
    )

    colA, colB, colC = st.columns(3)

    model_id = st.session_state.get("llm_model_id") or "llama-3.1-8b-instant"
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    temp = float(st.session_state.get("llm_temp", 0.2))
    max_tokens = int(st.session_state.get("llm_max_tokens", 350))
    groq_demo = GroqLLM(model_id=model_id, api_key=api_key, temp=temp, max_tokens=max_tokens)

    with colA:
        st.write("### 🔵 Sin contexto")
        if st.button("Generar SIN contexto"):
            resp = groq_demo.ask(sistema_sin, pregunta_demo)
            st.markdown(resp)

    with colB:
        st.write("### 🟢 Con EDA")
        if st.button("Generar con EDA"):
            resp = groq_demo.ask(sistema_con_eda, pregunta_demo)
            st.markdown(resp)

    with colC:
        st.write("### 🟣 Con EDA + ML")
        if st.button("Generar con EDA + ML"):
            resp = groq_demo.ask(sistema_eda_ml, pregunta_demo)
            st.markdown(resp)

    st.divider()
    st.write("### ✅ Evidencia de uso del contexto (heurística textual)")
    resp_eval = st.text_area("Pega aquí una respuesta del LLM (idealmente la de EDA+ML) para evaluarla:")
    if resp_eval:
        checks = {}
        nrows, ncols = eda["shape"]
        checks["Menciona #filas"] = str(nrows) in resp_eval
        missing = pd.Series(eda["null_pct"]).sort_values(ascending=False)
        top_missing_names = [str(k).lower() for k in missing[missing > 0].head(3).index]
        checks["Cita columnas con más nulos (EDA)"] = any(name in resp_eval.lower() for name in top_missing_names)
        # Señales ML
        if "ml" in st.session_state:
            ml = st.session_state["ml"]
            if ml["tarea"] == "clasificación":
                checks["Cita métricas ML (acc/f1)"] = ("accuracy" in resp_eval.lower() or "f1" in resp_eval.lower())
                if ml.get("top_errors"):
                    r,p,_ = ml["top_errors"][0]
                    checks["Menciona error frecuente real→pred"] = (f"{r}" in resp_eval and f"{p}" in resp_eval)
            else:
                checks["Cita métricas ML (RMSE/R²/MAE)"] = ("rmse" in resp_eval.lower() or "r²" in resp_eval.lower() or "mae" in resp_eval.lower())
            imp = ml["perm_importance"]
            if not imp.empty:
                any_feat = any(str(f).lower() in resp_eval.lower() for f in imp["feature"].head(5))
                checks["Menciona features influyentes (perm)"] = any_feat

        for k, v in checks.items():
            st.write(("✅ " if v else "⚠️ ") + k)

# ------------------------------- EXPORT TAB -------------------------------
with tab_export:
    st.subheader("Exportar resumen EDA")
    md = eda_summary_markdown(eda)
    st.write("**Vista previa (Markdown):**")
    st.code(md, language="markdown")

    st.download_button("⬇️ Descargar resumen_eda.md", data=md.encode("utf-8"),
                       file_name="resumen_eda.md", mime="text/markdown")

    st.write("**Tablas clave:**")
    miss_tbl = pd.DataFrame({
        "columna": list(eda["null_counts"].keys()),
        "faltantes": list(eda["null_counts"].values()),
        "faltantes_%": list(eda["null_pct"].values())
    }).sort_values("faltantes_%", ascending=False)
    st.download_button("⬇️ faltantes.csv", data=miss_tbl.to_csv(index=False).encode("utf-8"),
                       file_name="faltantes.csv", mime="text/csv")

    num_desc = eda["desc_num"]
    if not num_desc.empty:
        st.download_button("⬇️ descriptivos_numericos.csv", data=num_desc.to_csv(index=True).encode("utf-8"),
                           file_name="descriptivos_numericos.csv", mime="text/csv")
