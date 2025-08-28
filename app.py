# app.py
import os
import pandas as pd
import numpy as np
import streamlit as st

from utils_data import (
    load_csv, cap_rows, demo_data,
    basic_eda, eda_summary_text, eda_summary_markdown
)
from ml_utils import (
    infer_task, make_preprocessor,
    ml_metrics_and_artifacts, ml_key_insights, ml_objective_interpretation,
    safe_train_test_split
)
from llm_groq import GroqLLM, build_system_prompt, GROQ_MODEL_CATALOG
from ui_theme import apply_bio_theme, render_kpi_cards   # ← import KPI cards
from demo_tab import render_demo_tab

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline

# Plotly (gráficas)
try:
    from plotly import express as px
    PLOTLY_OK = True
except Exception as e:
    PLOTLY_OK = False
    PX_ERR = e

# ===================== Config & Tema =====================
st.set_page_config(page_title="Asistente BIOGEN", page_icon="🧬", layout="wide")
apply_bio_theme()

st.title("🧬 Asistente BIOGEN Insights")
st.caption("Carga un dataset de contenido relacionado a temas biológicos (o genéticos) e interactua en lenguaje natural para explorar sus posibilidades"
           "Herramienta educativa; **no** es consejo médico.")

# ===================== Sidebar =====================
with st.sidebar:
    st.header("1) Cargar datos")
    up = st.file_uploader("Sube un CSV", type=["csv"])
    usar_demo = st.checkbox("Usar datos demo (sintéticos)", value=not bool(up))

    st.header("2) Apariencia")
    tema_oscuro = st.toggle("Modo oscuro (gráficas)", value=False)
    PLOTLY_TEMPLATE = "plotly_dark" if tema_oscuro else "plotly"

    st.header("3) Groq (LLM)")
    modelo_key = st.selectbox("Modelo (Groq):", options=list(GROQ_MODEL_CATALOG.keys()), index=0)
    modelo_id = GROQ_MODEL_CATALOG[modelo_key]
    custom_id = st.text_input("Modelo personalizado (opcional)", value="")
    if custom_id.strip():
        modelo_id = custom_id.strip()
    st.session_state["llm_model_id"] = modelo_id
    st.caption(f"Usando: `{modelo_id}`")

    st.markdown("**Parámetros**")
    st.session_state["llm_temp"] = st.slider("Temperatura", 0.0, 1.0, st.session_state.get("llm_temp", 0.2), 0.05)
    st.session_state["llm_max_tokens"] = st.slider("Máx. tokens salida", 50, 800, st.session_state.get("llm_max_tokens", 350), 10)

# ===================== Cargar datos =====================
if up is not None:
    df = load_csv(up)
elif usar_demo:
    df = demo_data(1000)
    df = cap_rows(df, 1000)
else:
    st.info("Sube un CSV o activa el demo en la barra lateral.")
    st.stop()

st.success(f"Datos cargados: {df.shape[0]} filas × {df.shape[1]} columnas")
st.dataframe(df.head(), use_container_width=True)

# ===================== EDA base =====================
eda = basic_eda(df)

# ===================== Tabs =====================
tab_chat, tab_eda, tab_ml, tab_demo, tab_export = st.tabs(
    ["💬 Asistente", "📊 EDA", "🤖 ML", "🔬 Integración LLM", "📥 Exportar"]
)

# ------------------------------- CHAT TAB (con scroll) -------------------------------
with tab_chat:
    st.subheader("Chat assistant (Groq)")
    model_id = st.session_state.get("llm_model_id") or "llama-3.1-8b-instant"
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    temp = float(st.session_state.get("llm_temp", 0.2))
    max_tokens = int(st.session_state.get("llm_max_tokens", 350))
    llm = GroqLLM(model_id=model_id, api_key=api_key, temp=temp, max_tokens=max_tokens)

    st.info(llm.status_badge())

    if "mensajes" not in st.session_state:
        st.session_state.mensajes = [{"role":"assistant",
                                      "content":"Hola 👋 Puedo responder en español sobre tu dataset, el EDA, un ML baseline o temas generales de biología. ¿Qué quieres saber?"}]
    # Render histórico simple
    for m in st.session_state.mensajes:
        with st.chat_message(m["role"]):
            # Para mensajes largos antiguos, también usamos scroll
            html = f'<div class="scrollbox">{m["content"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\\n","<br/>")}</div>'
            st.markdown(html, unsafe_allow_html=True)

    # Contexto: EDA + (si existe) resumen ML
    eda_resumen = eda_summary_text(eda, df)
    ml_text = ""
    if "ml" in st.session_state:
        ml = st.session_state["ml"]
        ml_ins = ml_key_insights(ml, st.session_state["ml"]["X_te"])
        ml_text = "INSIGHTS DE ML:\n" + "\n".join(f"- {b}" for b in ml_ins) + "\n"
    sistema = build_system_prompt(eda_resumen, ml_text)

    pregunta = st.chat_input("Escribe tu pregunta…")
    if pregunta:
        st.session_state.mensajes.append({"role":"user", "content":pregunta})
        with st.chat_message("user"):
            st.markdown(f'<div class="scrollbox">{pregunta}</div>', unsafe_allow_html=True)

        # Streaming con scroll
        full = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()

            def render_scrolling_md(text: str):
                safe = (text.replace("&","&amp;")
                            .replace("<","&lt;")
                            .replace(">","&gt;")
                            .replace("\n","<br/>"))
                placeholder.markdown(f'<div class="scrollbox">{safe}</div>', unsafe_allow_html=True)

            for delta in llm.ask_stream(sistema, pregunta):
                full += delta or ""
                render_scrolling_md(full)

        st.session_state.mensajes.append({"role":"assistant", "content":full})

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

    st.write("**Análisis descriptivo**")
    num_desc = eda["desc_num"]
    if not num_desc.empty:
        st.dataframe(num_desc, use_container_width=True, height=260)
    else:
        st.write("No hay columnas numéricas.")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    with st.expander("Distribuciones numéricas (Top 5)"):
        if PLOTLY_OK:
            for c in num_cols[:6]:
                st.plotly_chart(px.histogram(df, x=c, nbins=40, title=c, marginal="box",
                                             template=PLOTLY_TEMPLATE), use_container_width=True)

    with st.expander("Frecuencias categóricas (Top 5)"):
        if PLOTLY_OK:
            for c in cat_cols[:6]:
                vc = df[c].astype(str).value_counts(dropna=False).head(20).reset_index()
                vc.columns = [c, "conteo"]
                st.plotly_chart(px.bar(vc, x=c, y="conteo", title=c, template=PLOTLY_TEMPLATE),
                                use_container_width=True)

    if len(num_cols) >= 2 and PLOTLY_OK:
        st.subheader("Matriz de correlación (variables numéricas)")
        corr = df[num_cols[:12]].corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr, text_auto=False, aspect="auto", title="Correlaciones",
                                  template=PLOTLY_TEMPLATE), use_container_width=True)

# ------------------------------- ML TAB -------------------------------
with tab_ml:
    st.subheader("Baseline ML")
    posibles_targets = [c for c in df.columns if c.lower() in ("etiqueta","label","target")]
    target_sugerido = posibles_targets[0] if posibles_targets else df.columns[0]
    target = st.selectbox("Variable objetivo", options=df.columns,
                          index=(list(df.columns).index(target_sugerido) if target_sugerido in df.columns else 0))
    y = df[target]
    X = df.drop(columns=[target])
    tarea = infer_task(y)
    st.write(f"Tarea de Machine Learning detectada: **{tarea}**")

    # Config de split
    st.markdown("**Configuración de particiones train/test**")
    csplit = st.columns(3)
    with csplit[0]:
        test_size = st.slider("Proporción de test", 0.10, 0.50, 0.20, 0.05)
    with csplit[1]:
        force_strat = st.checkbox("Forzar estratificación auto ajustando test_size", value=True)
    with csplit[2]:
        max_ts = st.slider("Máximo test_size permitido", 0.20, 0.50, max(0.35, test_size), 0.05,
                           help="Se usa solo si está activado 'Forzar estratificación'.")

    # Distribución global de la etiqueta
    if tarea == "clasificación":
        st.markdown("**Distribución de la variable de respuesta:**")
        vc_abs = y.astype(str).value_counts()
        vc_pct = (y.astype(str).value_counts(normalize=True) * 100).round(1)
        st.dataframe(pd.DataFrame({"conteo": vc_abs, "porcentaje_%": vc_pct}),
                     use_container_width=True, height=220)

    # Pipeline
    pre = make_preprocessor(X)
    modelo = LogisticRegression(max_iter=600) if tarea == "clasificación" else LinearRegression()
    pipe = Pipeline([("prep", pre), ("model", modelo)])

    if st.button("Entrenar modelo baseline"):
        # Split robusto
        X_tr, X_te, y_tr, y_te, split_info = safe_train_test_split(
            X, y, tarea=tarea,
            test_size=float(test_size), random_state=42,
            min_per_class=2,
            auto_increase_test_size=bool(force_strat),
            max_test_size=float(max_ts),
        )

        # Métricas del split
        colm = st.columns(3)
        with colm[0]:
            st.metric("Clases", split_info.get("n_classes") or (y.nunique() if tarea=="clasificación" else "-"))
        with colm[1]:
            st.metric("test_size usado", f"{split_info['used_test_size']:.2f}")
        with colm[2]:
            st.metric("Estratificado", "Sí" if split_info["stratified"] else "No")
        if split_info.get("rare_classes"):
            st.warning("Clases raras (<2 ejemplos): " + ", ".join(split_info["rare_classes"]))
        if not split_info["stratified"] and tarea == "clasificación":
            st.warning(split_info.get("reason") or "Estratificación desactivada.")

        # Entrenar & guardar
        ml_art = ml_metrics_and_artifacts(pipe, X_tr, X_te, y_tr, y_te, tarea)
        st.session_state["ml"] = {"pipeline": pipe, "X_te": X_te, "y_te": y_te, **ml_art}

        # ===== KPI GRANDES =====
        ml = st.session_state["ml"]
        if tarea == "clasificación":
            items = [
                ("Accuracy (test)", f"{ml['acc_te']:.3f}", f"Train: {ml['acc_tr']:.3f}"),
                ("F1-macro (test)", f"{ml['f1_te']:.3f}", f"Train: {ml['f1_tr']:.3f}"),
                ("Clases", f"{len(ml['labels'])}", "en test"),
            ]
            render_kpi_cards(items, caption="Resultados del entrenamiento")
        else:
            items = [
                ("RMSE (test)", f"{ml['rmse']:.3f}", f"Baseline media: {ml['rmse_baseline']:.3f}"),
                ("MAE (test)", f"{ml['mae']:.3f}", ""),
                ("R² (test)", f"{ml['r2']:.3f}", ""),
            ]
            render_kpi_cards(items, caption="Resultados del entrenamiento")

    # Visualizaciones + importancias
    if "ml" in st.session_state and st.session_state["ml"]["tarea"] == tarea:
        ml = st.session_state["ml"]
        if tarea == "clasificación":
            st.write(f"**Accuracy (test):** {ml['acc_te']:.3f} | **F1-macro (test):** {ml['f1_te']:.3f} "
                     f"(train: acc={ml['acc_tr']:.3f}, f1={ml['f1_tr']:.3f})")
            if PLOTLY_OK:
                etiquetas = ml["labels"]; cm = ml["cm"]
                fig_cm = px.imshow(cm, x=etiquetas, y=etiquetas, text_auto=True, color_continuous_scale="Greens",
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

        imp = ml["perm_importance"]
        st.write("**Importancias por permutación (top):**")
        st.dataframe(imp, use_container_width=True, height=240)
        if PLOTLY_OK and not imp.empty:
            st.plotly_chart(px.bar(imp, x="feature", y="importance", error_y="std",
                                   title="Importancia (perm)", template=PLOTLY_TEMPLATE),
                            use_container_width=True)
        st.info(ml_objective_interpretation(ml))


# ------------------------------- DEMO TAB -------------------------------
with tab_demo:
    render_demo_tab(df=df, eda=eda, PLOTLY_TEMPLATE=PLOTLY_TEMPLATE)

# ------------------------------- EXPORT TAB -------------------------------
with tab_export:
    st.subheader("Exportar resumen del EDA")
    md = eda_summary_markdown(eda)
    st.write("**Vista previa (Markdown):**")
    st.code(md, language="markdown")

    st.download_button("⬇️ Descargar resumen_eda.md", data=md.encode("utf-8"),
                       file_name="resumen_eda.md", mime="text/markdown")

    miss_tbl = pd.DataFrame({
        "columna": list(eda["null_counts"].keys()),
        "faltantes": list(eda["null_counts"].values()),
        "faltantes_%": list(eda["null_pct"].values())
    }).sort_values("faltantes_%", ascending=False)
    st.download_button("⬇️ faltantes.csv", data=miss_tbl.to_csv(index=False).encode("utf-8"),
                       file_name="faltantes.csv", mime="text/csv")

    num_desc = eda["desc_num"]
    if not num_desc.empty:
        st.download_button("⬇️ analisis_descriptivo.csv",
                           data=num_desc.to_csv(index=True).encode("utf-8"),
                           file_name="analisis_descriptivo.csv", mime="text/csv")
