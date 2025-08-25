# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict
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

# —— Plotly (robusto) ——
try:
    from plotly import express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception as e:
    PLOTLY_OK = False
    PX_ERR = e

# —— BACKEND LOCAL (transformers/tokenizers) ——
# Si no vas a usar backend local, puedes comentar estas importaciones.
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_OK = True
except Exception as e:
    TRANSFORMERS_OK = False

# ===================== Configuración de página =====================
st.set_page_config(page_title="Asistente Genético (EDA + Chat)", page_icon="🧬", layout="wide")
st.title("🧬 Asistente Genético: EDA + Chat LLM (ligero)")
st.caption("Sube un CSV (hasta ~1.000 filas) → EDA enriquecido → Baseline ML → Chat en español. "
           "Herramienta educativa; **no** es consejo médico.")

# ===================== Catálogo de modelos LLM (ligeros) =====================
MODEL_CATALOG = {
    "TinyLlama 1.1B Chat (por defecto)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Llama 3.2 1B Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen2.5 0.5B Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Falcon 1B Instruct": "tiiuae/falcon-1b-instruct",
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
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
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

# ===================== Backend local: carga cacheada =====================
@st.cache_resource(show_spinner=True)
def load_local_model(model_id: str, hf_token: str):
    """
    Descarga y cachea tokenizer y modelo para uso local (CPU).
    Requiere transformers/tokenizers/torch.
    """
    if not TRANSFORMERS_OK:
        raise RuntimeError("transformers/torch no están instalados. Agrega a requirements.txt.")

    tok = AutoTokenizer.from_pretrained(
        model_id, token=hf_token, use_fast=True, trust_remote_code=True
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, token=hf_token,
        torch_dtype=torch.float32,   # CPU; si tienes GPU, ajusta dtype y device_map
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    mdl.eval()
    return tok, mdl

def local_generate(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float) -> str:
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=float(temperature) > 0.0,
            temperature=max(0.01, float(temperature)),
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=pad_id
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

# ===================== LLM: badge de estado =====================
def llm_status_badge(llm: "TinyLLM") -> str:
    which = "Local (transformers)" if llm.backend == "local" else "API (HF Inference)"
    if llm.available():
        return (f"🟢 **LLM conectado** — {which} — modelo: `{llm.model}` · "
                f"temp={st.session_state.get('llm_temp',0.2)} · "
                f"max_tokens={st.session_state.get('llm_max_tokens',350)}")
    return f"🟠 **LLM parcialmente configurado** — {which}. Revisa token/permisos si usas API."

# =============== Cliente LLM con DOS backends (API y Local) ===============
class TinyLLM:
    """
    LLM con dos backends:
      - API (HF Inference): usa requests + HF_TOKEN
      - Local (transformers/tokenizers): descarga y ejecuta el modelo aquí (CPU)
    Lee modelo de st.session_state['hf_model_override'] y backend de st.session_state['llm_backend'].
    Maneja errores HTTP en claro. Respeta temperatura y max tokens desde la UI.
    """
    def __init__(self, timeout: int = 60):
        # modelo activo (selector del sidebar o secreto/env)
        def _get_secret(key, default=None):
            val = os.environ.get(key)
            if not val:
                try:
                    val = st.secrets.get(key, default)
                except Exception:
                    val = default
            return val

        self.model = st.session_state.get("hf_model_override") or _get_secret(
            "HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        self.hf_token = _get_secret("HF_TOKEN", None)
        self.timeout = timeout
        self.backend = st.session_state.get("llm_backend", "api")  # "api" | "local"

    def available(self) -> bool:
        if self.backend == "api":
            # Con API pública algunos modelos responden sin token, pero lo normal es necesitarlo
            return bool(self.hf_token)
        else:  # local
            return TRANSFORMERS_OK  # requiere libs instaladas

    def _human_error(self, status: int, body_text: str) -> str:
        msg = (body_text or "")[:400]
        if status == 401:
            return "⚠️ 401 No autorizado: revisa `HF_TOKEN`."
        if status == 403:
            return (
                "⚠️ 403 Prohibido: el modelo es privado o requiere aceptar sus términos.\n"
                "Soluciones:\n"
                "1) Usa un `HF_TOKEN` con acceso.\n"
                "2) En la página del modelo, pulsa 'Request access'/'Agree and access repository'.\n"
                "3) Selecciona un modelo público en el sidebar y reintenta.\n"
                f"Detalle: {msg}"
            )
        if status == 404:
            return "⚠️ 404 No encontrado: revisa `HF_MODEL`/selector."
        if status == 429:
            return "⚠️ 429 Rate limit: intenta de nuevo en breve."
        if status == 503:
            return "⚠️ 503 Inicializando el modelo en HF; reintenta."
        return f"⚠️ Error HTTP {status}: {msg}"

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        temp = float(st.session_state.get("llm_temp", 0.2))
        max_tokens = int(st.session_state.get("llm_max_tokens", 350))
        prompt = f"Sistema: {system_prompt}\n\nUsuario: {user_prompt}\n\nAsistente:"

        # Backend LOCAL
        if self.backend == "local":
            if not TRANSFORMERS_OK:
                return "⚠️ Backend local no disponible: instala transformers/tokenizers/torch."
            try:
                tok, mdl = load_local_model(self.model, self.hf_token)
                out = local_generate(tok, mdl, prompt, max_new_tokens=max_tokens, temperature=temp)
                return out.strip()
            except Exception as e:
                return f"⚠️ Error en backend local (transformers/tokenizers): {e}"

        # Backend API (HF Inference)
        if not self.hf_token:
            # Modo offline si no hay token para API
            lines = [ln for ln in system_prompt.splitlines()
                     if any(tok in ln.lower() for tok in user_prompt.lower().split()[:5])]
            if not lines:
                lines = system_prompt.splitlines()[:15]
            return "Modo sin token para API. Activa 'Local' o configura HF_TOKEN. Contexto EDA:\n\n" + "\n".join(lines[:40])

        headers = {"Authorization": f"Bearer {self.hf_token}", "Accept": "application/json"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_tokens, "temperature": temp},
            "options": {"wait_for_model": True, "use_cache": True}
        }
        url = f"https://api-inference.huggingface.co/models/{self.model}"

        try:
            r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            return f"⚠️ Error de red al llamar a HF Inference: {e}"

        if r.status_code != 200:
            try:
                err = r.json().get("error", r.text)
            except Exception:
                err = r.text
            return self._human_error(r.status_code, err)

        try:
            data = r.json()
        except ValueError:
            return "⚠️ La respuesta de HF no es JSON válido."

        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"]
            return text.split("Asistente:", 1)[-1].strip()
        if isinstance(data, dict) and "error" in data:
            return self._human_error(500, data["error"])
        return str(data)[:1000]

# ===================== Sidebar (carga, apariencia, LLM) =====================
with st.sidebar:
    st.header("1) Cargar datos")
    up = st.file_uploader("Sube un CSV", type=["csv"])
    usar_demo = st.checkbox("Usar datos demo (sintéticos)", value=not bool(up))

    st.header("2) Apariencia")
    tema_oscuro = st.toggle("Modo oscuro (gráficas)", value=False)
    PLOTLY_TEMPLATE = "plotly_dark" if tema_oscuro else "plotly"

    st.header("3) LLM (español, pequeño)")
    backend = st.radio(
        "Backend de LLM",
        ["API (HF Inference)", "Local (transformers/tokenizers)"],
        index=0,
        help="API = peticiones HTTP a Hugging Face; Local = descarga y ejecuta el modelo aquí."
    )
    st.session_state["llm_backend"] = "local" if backend.startswith("Local") else "api"

    modelo_key = st.selectbox(
        "Modelo:",
        options=list(MODEL_CATALOG.keys()),
        index=0,
        help="Elige un modelo liviano."
    )
    modelo_id = MODEL_CATALOG[modelo_key]
    custom_id = st.text_input("Modelo personalizado (opcional)", value="")
    if custom_id.strip():
        modelo_id = custom_id.strip()
    st.session_state["hf_model_override"] = modelo_id
    st.caption(f"Usando: `{modelo_id}`")

    st.markdown("**Parámetros del modelo**")
    st.session_state["llm_temp"] = st.slider("Temperatura", 0.0, 1.0, st.session_state.get("llm_temp", 0.2), 0.05,
                                             help="Menor = más determinista.")
    st.session_state["llm_max_tokens"] = st.slider("Máx. tokens de salida", 50, 800,
                                                   st.session_state.get("llm_max_tokens", 350), 10,
                                                   help="Respuesta más corta o más larga.")

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
tab_eda, tab_ml, tab_chat, tab_export = st.tabs(["📊 EDA", "🤦 ML", "💬 Chat", "📥 Exportar"])

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
            st.plotly_chart(
                px.bar(miss_tbl, x="columna", y="faltantes_%", title="% faltantes", template=PLOTLY_TEMPLATE),
                use_container_width=True
            )
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
                st.plotly_chart(px.histogram(df, x=c, nbins=40, title=c, marginal="box",
                                             template=PLOTLY_TEMPLATE), use_container_width=True)

    with st.expander("Frecuencias categóricas (hasta 6)"):
        if PLOTLY_OK:
            for c in cat_cols[:6]:
                vc = df[c].astype(str).value_counts(dropna=False).head(20).reset_index()
                vc.columns = [c, "conteo"]
                st.plotly_chart(px.bar(vc, x=c, y="conteo", title=c, template=PLOTLY_TEMPLATE),
                                use_container_width=True)

    if len(num_cols) >= 2 and PLOTLY_OK:
        st.subheader("Matriz de correlación (subset numérico)")
        corr = df[num_cols[:12]].corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr, text_auto=False, aspect="auto", title="Correlaciones",
                                  template=PLOTLY_TEMPLATE), use_container_width=True)

    posibles_targets = [c for c in df.columns if c.lower() in ("etiqueta","label","target")]
    target_sugerido = posibles_targets[0] if posibles_targets else None
    st.subheader("Distribuciones por clase (opcional)")
    col_tar = st.selectbox("Columna de etiqueta (opcional)", options=["(ninguna)"] + list(df.columns),
                           index=(list(df.columns).index(target_sugerido)+1 if target_sugerido in df.columns else 0))
    if col_tar != "(ninguna)" and PLOTLY_OK:
        bal = df[col_tar].astype(str).value_counts(dropna=False).reset_index()
        bal.columns = [col_tar, "conteo"]
        st.plotly_chart(px.bar(bal, x=col_tar, y="conteo", title="Balance de clases",
                               template=PLOTLY_TEMPLATE), use_container_width=True)
        for c in num_cols[:3]:
            st.plotly_chart(px.violin(df, x=col_tar, y=c, box=True, points="outliers",
                                      title=f"{c} por {col_tar}", template=PLOTLY_TEMPLATE), use_container_width=True)
        if len(num_cols) >= 2:
            st.plotly_chart(px.scatter(df, x=num_cols[0], y=num_cols[1], color=col_tar,
                                       title=f"{num_cols[0]} vs {num_cols[1]} por {col_tar}",
                                       template=PLOTLY_TEMPLATE), use_container_width=True)

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
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        if tarea == "clasificación":
            acc = accuracy_score(y_te, pred)
            f1m = f1_score(y_te, pred, average="macro")
            st.write(f"**Accuracy:** {acc:.3f} | **F1-macro:** {f1m:.3f}")

            if PLOTLY_OK:
                etiquetas = sorted(pd.Series(y_te).astype(str).unique())
                cm = confusion_matrix(y_te, pred, labels=etiquetas)
                fig_cm = px.imshow(cm, x=etiquetas, y=etiquetas, text_auto=True,
                                   color_continuous_scale="Blues", title="Matriz de confusión",
                                   template=PLOTLY_TEMPLATE)
                fig_cm.update_layout(xaxis_title="Predicha", yaxis_title="Real")
                st.plotly_chart(fig_cm, use_container_width=True)

        else:
            mae = mean_absolute_error(y_te, pred)
            rmse = mean_squared_error(y_te, pred, squared=False)
            r2 = r2_score(y_te, pred)
            st.write(f"**MAE:** {mae:.3f} | **RMSE:** {rmse:.3f} | **R²:** {r2:.3f}")

            if PLOTLY_OK:
                resid = y_te - pred
                st.plotly_chart(px.scatter(x=pred, y=resid, labels={"x":"Predicción", "y":"Residual"},
                                           title="Predicción vs Residual", template=PLOTLY_TEMPLATE),
                                use_container_width=True)
                st.plotly_chart(px.histogram(resid, nbins=40, title="Histograma de residuales",
                                             template=PLOTLY_TEMPLATE), use_container_width=True)

# ------------------------------- CHAT TAB -------------------------------
with tab_chat:
    st.subheader("Chat en español")

    # Estado del LLM y diagnóstico
    llm = TinyLLM()
    st.info(llm_status_badge(llm))

    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("Probar conexión LLM"):
            test_resp = llm.ask("Eres un sistema de prueba.", "Di 'ok' en una palabra.")
            st.write("**Respuesta de prueba:**", test_resp[:500])
    with cols[1]:
        st.write(f"**Backend:** {'Local (transformers)' if llm.backend=='local' else 'API (HF Inference)'}")
        st.write(f"**Modelo:** `{llm.model}`")

    # Historial de chat
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = [{"role":"assistant",
                                      "content":"Hola 👋 Puedo responder en español sobre tu dataset y genética básica. ¿Qué quieres saber?"}]
    for m in st.session_state.mensajes:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Contexto EDA -> sistema
    eda_resumen = eda_summary_text(eda, df)
    sistema = (
        "Eres un asistente de datos genéticos. Responde en **español**, breve y conservador. "
        "No des consejo médico. Usa el EDA cuando aplique.\n\n"
        f"CONTEXTO EDA:\n{eda_resumen}\n"
    )

    pregunta = st.chat_input("Escribe tu pregunta…")
    if pregunta:
        st.session_state.mensajes.append({"role":"user", "content":pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)
        respuesta = llm.ask(sistema, pregunta)

        # Sugerencia rápida si 403
        if isinstance(respuesta, str) and respuesta.startswith("⚠️ 403"):
            st.warning("El modelo actual no está accesible con tu token.")
            cA, cB = st.columns(2)
            with cA:
                if st.button("Cambiar a modelo público (TinyLlama)"):
                    st.session_state["hf_model_override"] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                    st.experimental_rerun()
            with cB:
                if st.button("Cambiar a Qwen 0.5B"):
                    st.session_state["hf_model_override"] = "Qwen/Qwen2.5-0.5B-Instruct"
                    st.experimental_rerun()

        st.session_state.mensajes.append({"role":"assistant", "content":respuesta})
        with st.chat_message("assistant"):
            st.markdown(respuesta)

# ------------------------------- EXPORT TAB -------------------------------
with tab_export:
    st.subheader("Exportar resumen EDA")
    md = eda_summary_markdown(eda)
    st.write("**Vista previa (Markdown):**")
    st.code(md, language="markdown")

    st.download_button("⬇️ Descargar resumen_eda.md", data=md.encode("utf-8"),
                       file_name="resumen_eda.md", mime="text/markdown")

    # Exportables en CSV
    st.write("**Tablas clave:**")
    if 'miss_tbl' not in locals():
        miss_tbl = pd.DataFrame({
            "columna": list(eda["null_counts"].keys()),
            "faltantes": list(eda["null_counts"].values()),
            "faltantes_%": list(eda["null_pct"].values())
        }).sort_values("faltantes_%", ascending=False)

    st.download_button("⬇️ faltantes.csv", data=miss_tbl.to_csv(index=False).encode("utf-8"),
                       file_name="faltantes.csv", mime="text/csv")

    num_desc = eda["desc_num"]
    if not num_desc.empty:
        st.download_button("⬇️ descriptivos_numericos.csv",
                           data=num_desc.to_csv(index=True).encode("utf-8"),
                           file_name="descriptivos_numericos.csv", mime="text/csv")
