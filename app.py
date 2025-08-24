import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from typing import Dict, List
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
import requests

# ===================== Configuración de página =====================
st.set_page_config(page_title="Asistente Genético (EDA + Chat)", page_icon="🧬", layout="wide")
st.title("🧬 Asistente Genético: EDA + Chat LLM (ligero)")
st.caption("Sube un CSV (hasta ~1.000 filas) → EDA enriquecido → Baseline ML → Chat en español sobre tu dataset/genética. (Educativo; no es consejo médico).")

# ===================== Utilidades =====================
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
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

# =============== Cliente LLM (español, ligero, chat) ===============
class TinyLLM:
    """
    Modelo pequeño en HuggingFace Inference (por defecto TinyLlama 1.1B Chat).
    Responde en español si lo pides en el prompt. Define HF_TOKEN en secretos/env.
    """
    def __init__(self, model: str = None, timeout: int = 45):
        self.model = model or os.environ.get("HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.hf_token = os.environ.get("HF_TOKEN")
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.hf_token)

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        if not self.available():
            # Modo offline: devuelve trozos del contexto (español)
            lines = [ln for ln in system_prompt.splitlines()
                     if any(tok in ln.lower() for tok in user_prompt.lower().split()[:5])]
            if not lines:
                lines = system_prompt.splitlines()[:15]
            return "Modo sin LLM (no hay HF_TOKEN). Contexto EDA:\n\n" + "\n".join(lines[:40])

        headers = {"Authorization": f"Bearer {self.hf_token}"}
        prompt = f"Sistema: {system_prompt}\n\nUsuario: {user_prompt}\n\nAsistente:"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 350, "temperature": 0.2}}
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model}",
            json=payload, headers=headers, timeout=self.timeout
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"]
            return text.split("Asistente:", 1)[-1].strip()
        if isinstance(data, dict) and "error" in data:
            return f"[HF Inference Error] {data['error']}"
        return str(data)

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

# ===================== Sidebar (carga y opciones) =====================
with st.sidebar:
    st.header("1) Cargar datos")
    up = st.file_uploader("Sube un CSV", type=["csv"])
    usar_demo = st.checkbox("Usar datos demo (sintéticos)", value=not bool(up))

    st.header("2) LLM (español, pequeño)")
    st.write("Proveedor: HuggingFace Inference. Modelo por defecto: TinyLlama 1.1B Chat.")
    st.caption("Configura `HF_TOKEN` (secrets/env). Puedes cambiar `HF_MODEL` si quieres otro LLM pequeñito.")

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

# ===================== EDA enriquecido =====================
st.header("2) EDA enriquecido")

eda = basic_eda(df)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Faltantes por columna")
    miss_tbl = pd.DataFrame({
        "columna": list(eda["null_counts"].keys()),
        "faltantes": list(eda["null_counts"].values()),
        "faltantes_%": list(eda["null_pct"].values())
    }).sort_values("faltantes_%", ascending=False)
    st.dataframe(miss_tbl, use_container_width=True, height=240)
    st.plotly_chart(px.bar(miss_tbl, x="columna", y="faltantes_%", title="% faltantes"), use_container_width=True)

with c2:
    st.subheader("Tipos y cardinalidades")
    dtype_tbl = pd.DataFrame({
        "columna": list(eda["dtypes"].keys()),
        "tipo": list(eda["dtypes"].values()),
        "nunique": list(eda["nunique"].values())
    }).sort_values("nunique", ascending=False)
    st.dataframe(dtype_tbl, use_container_width=True, height=240)

st.subheader("Descriptivos numéricos")
num_desc = eda["desc_num"]
if not num_desc.empty:
    st.dataframe(num_desc, use_container_width=True, height=260)
else:
    st.write("No hay columnas numéricas.")

# Distribuciones enriquecidas
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

with st.expander("Distribuciones numéricas (hasta 6) con box marginal"):
    for c in num_cols[:6]:
        st.plotly_chart(px.histogram(df, x=c, nbins=40, title=c, marginal="box"), use_container_width=True)

with st.expander("Frecuencias categóricas (hasta 6)"):
    for c in cat_cols[:6]:
        vc = df[c].astype(str).value_counts(dropna=False).head(20).reset_index()
        vc.columns = [c, "conteo"]
        st.plotly_chart(px.bar(vc, x=c, y="conteo", title=c), use_container_width=True)

# Correlación (subconjunto numérico)
if len(num_cols) >= 2:
    st.subheader("Matriz de correlación (subset numérico)")
    corr = df[num_cols[:12]].corr(numeric_only=True)
    st.plotly_chart(px.imshow(corr, text_auto=False, aspect="auto", title="Correlaciones"), use_container_width=True)

# Violin/box por clase (si hay etiqueta)
posibles_targets = [c for c in df.columns if c.lower() in ("etiqueta","label","target")]
target_sugerido = posibles_targets[0] if posibles_targets else None
st.subheader("Distribuciones por clase (si eliges una etiqueta)")
col_tar = st.selectbox("Columna de etiqueta (opcional)", options=["(ninguna)"] + list(df.columns),
                       index=(list(df.columns).index(target_sugerido)+1 if target_sugerido in df.columns else 0))
if col_tar != "(ninguna)":
    if col_tar in df.columns:
        # Balance de clases
        bal = df[col_tar].astype(str).value_counts(dropna=False).reset_index()
        bal.columns = [col_tar, "conteo"]
        st.plotly_chart(px.bar(bal, x=col_tar, y="conteo", title="Balance de clases"), use_container_width=True)

        # Violin para 1-3 num cols
        for c in num_cols[:3]:
            st.plotly_chart(px.violin(df, x=col_tar, y=c, box=True, points="outliers",
                                      title=f"Distribución de {c} por {col_tar}"), use_container_width=True)

        # Dispersión entre dos num cols (si hay)
        if len(num_cols) >= 2:
            st.plotly_chart(
                px.scatter(df, x=num_cols[0], y=num_cols[1], color=col_tar,
                           title=f"{num_cols[0]} vs {num_cols[1]} coloreado por {col_tar}"),
                use_container_width=True
            )

# ===================== Baseline ML mínimo =====================
st.header("3) Baseline ML mínimo")

target = st.selectbox("Selecciona la variable objetivo", options=df.columns,
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
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    if tarea == "clasificación":
        acc = accuracy_score(y_te, pred)
        f1m = f1_score(y_te, pred, average="macro")
        st.write(f"**Accuracy:** {acc:.3f} | **F1-macro:** {f1m:.3f}")

        # Matriz de confusión
        etiquetas = sorted(pd.Series(y_te).astype(str).unique())
        cm = confusion_matrix(y_te, pred, labels=etiquetas)
        fig_cm = px.imshow(cm, x=etiquetas, y=etiquetas, text_auto=True, color_continuous_scale="Blues",
                           title="Matriz de confusión")
        fig_cm.update_layout(xaxis_title="Predicha", yaxis_title="Real")
        st.plotly_chart(fig_cm, use_container_width=True)

    else:
        mae = mean_absolute_error(y_te, pred)
        rmse = mean_squared_error(y_te, pred, squared=False)
        r2 = r2_score(y_te, pred)
        st.write(f"**MAE:** {mae:.3f} | **RMSE:** {rmse:.3f} | **R²:** {r2:.3f}")

        # Gráficos de residuales simples
        resid = y_te - pred
        st.plotly_chart(px.scatter(x=pred, y=resid, labels={"x":"Predicción", "y":"Residual"},
                                   title="Predicción vs Residual"), use_container_width=True)
        st.plotly_chart(px.histogram(resid, nbins=40, title="Histograma de residuales"),
                        use_container_width=True)

# ===================== Chat LLM en español (historial) =====================
st.header("4) Chat en español sobre tu EDA/datos")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = [{"role":"assistant",
                                  "content":"Hola 👋 Soy tu asistente. Puedo responder en español sobre tu dataset y temas básicos de genética. ¿Qué quieres saber?"}]

# Render del historial
for m in st.session_state.mensajes:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Construir contexto EDA
eda_resumen = eda_summary_text(eda, df)
llm = TinyLLM()
sistema = (
    "Eres un asistente de datos genéticos. Responde en **español**, de forma breve y conservadora. "
    "No des consejo médico. Cuando sea posible, referencia columnas y métricas simples del EDA.\n\n"
    f"CONTEXTO EDA:\n{eda_resumen}\n"
)

# Entrada de chat
pregunta = st.chat_input("Escribe tu pregunta (en español)...")
if pregunta:
    # Mostrar mensaje del usuario
    st.session_state.mensajes.append({"role":"user", "content":pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Responder con LLM (o fallback)
    respuesta = llm.ask(sistema, pregunta)
    st.session_state.mensajes.append({"role":"assistant", "content":respuesta})
    with st.chat_message("assistant"):
        st.markdown(respuesta)
