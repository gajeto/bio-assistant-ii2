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
    infer_task, make_preprocessor, make_preprocessor_advanced,
    ml_metrics_and_artifacts, ml_key_insights, ml_objective_interpretation,
    safe_train_test_split, model_choices, build_model, ml_bullet_summary
)
from eda import (
    outlier_report, skew_kurtosis_table, suggest_transforms, find_top_corr_pair,
    pca_2d, dbscan_outliers, compute_vif_table, condition_number
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
st.caption("Carga un dataset de contenido relacionado a temas biológicos (o genéticos) e interactua en lenguaje natural para explorar sus posibilidades")

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
    st.session_state["ctx_limit"] = st.slider(
    "Límite de contexto para el LLM (tokens aprox.)",
    300, 3000, st.session_state.get("ctx_limit", 1200), 50,
    help="El contexto EDA/ML se truncará adecuadamente a este límite"
    )


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
            st.plotly_chart(px.bar(miss_tbl, x="columna", y="faltantes_%", title="% faltantes", template=PLOTLY_TEMPLATE),
                            use_container_width=True)
        else:
            st.warning(f"Plotly no disponible: {PX_ERR}")

    with c2:
        st.write("**Tipos y cardinalidades**")
        st.dataframe(dtype_tbl, use_container_width=True, height=260)

    st.write("**Descriptivos numéricos**")
    num_desc = eda["desc_num"]
    if not num_desc.empty:
        st.dataframe(num_desc, use_container_width=True, height=260)

    # ==== NUEVO: Outliers & Sesgo ====
    st.subheader("Calidad avanzada: Outliers y Sesgo")
    rep_out, any_out_mask = outlier_report(df)
    st.dataframe(rep_out, use_container_width=True, height=260)
    if PLOTLY_OK and not rep_out.empty:
        st.plotly_chart(px.bar(rep_out.head(12), x="columna", y="pct_outliers",
                               title="% de outliers (IQR) – Top 12", template=PLOTLY_TEMPLATE),
                        use_container_width=True)
    sk_tbl = skew_kurtosis_table(df)
    st.write("**Skew / Kurtosis (ordenado por sesgo):**")
    st.dataframe(sk_tbl, use_container_width=True, height=220)

    # Boxplots de las 3 columnas con más outliers
    if PLOTLY_OK and not rep_out.empty:
        st.write("**Boxplots (Top 3 columnas con más outliers)**")
        for c in rep_out["columna"].head(3):
            st.plotly_chart(px.box(df, y=c, points="suspectedoutliers", title=f"Boxplot: {c}",
                                   template=PLOTLY_TEMPLATE), use_container_width=True)

    # Sugerencias de FE/Prepro
    st.subheader("Sugerencias de Feature Engineering / Preprocesamiento")
    for s in suggest_transforms(df):
        st.markdown(f"- {s}")

    # Heatmap de faltantes (muestra 200 filas para legibilidad)
    if PLOTLY_OK:
        st.subheader("Patrones de faltantes (muestra)")
        sample = df.sample(min(200, len(df)), random_state=42)
        heat = sample.isna().astype(int)
        st.plotly_chart(px.imshow(heat.T, aspect="auto", color_continuous_scale="Greens",
                                  title="Heatmap de faltantes (1=NaN)", template=PLOTLY_TEMPLATE),
                        use_container_width=True)
    
        # ==== Descriptivos numéricos con formato visible ====
    st.subheader("Descriptivos numéricos (formato destacado)")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        desc = df[num_cols].describe().T
        desc = desc.rename(columns={
            "count": "conteo", "mean": "media", "std": "desv",
            "min": "min", "25%": "p25", "50%": "mediana", "75%": "p75", "max": "max"
        })
        desc = desc[["conteo","media","desv","min","p25","mediana","p75","max"]].round(3)
        st.dataframe(desc, use_container_width=True, height=300)
    else:
        st.info("No hay columnas numéricas.")

    # ==== Matriz de correlaciones (si aplica) ====
    if PLOTLY_OK and len(num_cols) >= 2:
        st.subheader("Matriz de correlación (subset numérico)")
        corr = df[num_cols[:12]].corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr, text_auto=False, aspect="auto",
                                title="Correlaciones", template=PLOTLY_TEMPLATE),
                        use_container_width=True)

    # ==== Multicolinealidad: VIF y número de condición ====
    st.subheader("Multicolinealidad (VIF y número de condición)")
    try:
        vif_df = compute_vif_table(df)
    except Exception as e:
        vif_df = pd.DataFrame(columns=["feature", "vif", "r2_aux"])
        st.warning("No se pudo calcular VIF de forma robusta en este dataset.")

    if not vif_df.empty:
        st.dataframe(vif_df.round({"vif": 2, "r2_aux": 3}), use_container_width=True, height=260)
        max_vif = np.nanmax(vif_df["vif"].values)
        st.caption(f"VIF alto (>10) sugiere multicolinealidad fuerte. VIF máximo observado: **{max_vif:.2f}**.")
    else:
        st.info("No fue posible calcular VIF (se requieren ≥2 columnas numéricas no constantes).")


    cn = condition_number(df)
    if cn is not None:
        st.caption(f"Número de condición (matriz estandarizada): **{cn:.1f}** "
                f"(>30 indica posible multicolinealidad problemática).")

    # ==== Resumen final de insights (bullets) ====
    st.subheader("Resumen de insights del EDA")
    res_bullets = []
    # Tamaño
    res_bullets.append(f"- Filas: **{df.shape[0]}**, Columnas: **{df.shape[1]}**.")
    # Faltantes
    miss_pct = df.isna().mean().sort_values(ascending=False)
    top_miss = miss_pct.head(3)[miss_pct.head(3) > 0]
    if len(top_miss) > 0:
        res_bullets.append("- Columnas con más faltantes: " + ", ".join([f"**{c}** ({p*100:.1f}%)" for c,p in top_miss.items()]) + ".")
    # Outliers
    try:
        rep_out, _mask = outlier_report(df)
        if not rep_out.empty and rep_out["pct_outliers"].max() > 0:
            r = rep_out.head(3)[["columna","pct_outliers"]].values
            res_bullets.append("- Outliers (IQR) más afectados: " + ", ".join([f"**{c}** ({pct:.1f}%)" for c,pct in r]) + ".")
    except Exception:
        pass
    # Correlación fuerte
    try:
        if len(num_cols) >= 2:
            cmat = df[num_cols].corr(numeric_only=True).abs()
            np.fill_diagonal(cmat.values, np.nan)
            max_corr = np.nanmax(cmat.values)
            if np.isfinite(max_corr) and max_corr >= 0.8:
                i, j = np.unravel_index(np.nanargmax(cmat.values), cmat.shape)
                res_bullets.append(f"- Posible colinealidad: **{num_cols[i]} ~ {num_cols[j]}** (|r|≈{max_corr:.2f}).")
    except Exception:
        pass
    # VIF alto
    if not vif_df.empty and (vif_df["vif"] > 10).any():
        high = vif_df[vif_df["vif"] > 10].head(5)
        res_bullets.append("- VIF alto (>10): " + ", ".join([f"**{f}** ({v:.1f})" for f,v in zip(high['feature'], high['vif'])]) + ".")
    # Sugerencias rápidas
    for s in suggest_transforms(df):
        res_bullets.append(f"- {s}")

    st.markdown("\n".join(res_bullets))


# ------------------------------- ML TAB -------------------------------
with tab_ml:
    st.subheader("Baseline ML enriquecido")

    # ======= Selección de target y tarea =======
    posibles_targets = [c for c in df.columns if c.lower() in ("etiqueta","label","target")]
    target_sugerido = posibles_targets[0] if posibles_targets else df.columns[0]
    target = st.selectbox("Variable objetivo", options=df.columns,
                          index=(list(df.columns).index(target_sugerido) if target_sugerido in df.columns else 0))
    y_all = df[target]
    X_all = df.drop(columns=[target])
    tarea = infer_task(y_all)
    st.write(f"Tarea detectada: **{tarea}**")

    # ======= Config de split =======
    st.markdown("**Configuración de partición train/test**")
    csplit = st.columns(3)
    with csplit[0]:
        test_size = st.slider("Proporción de test", 0.10, 0.50, 0.20, 0.05)
    with csplit[1]:
        force_strat = st.checkbox("Forzar estratificación auto-ajustando test_size", value=True)
    with csplit[2]:
        max_ts = st.slider("Máximo test_size permitido", 0.20, 0.50, max(0.35, test_size), 0.05)

    # ======= Selección de modelo =======
    st.markdown("**Modelo**")
    modelos = model_choices(tarea)
    modelo_name = st.selectbox("Tipo de modelo", options=modelos,
                               index=(modelos.index("Logistic Regression") if tarea=="clasificación" else modelos.index("Linear Regression")))
    modelo = build_model(tarea, modelo_name)

    # ======= Preprocesamiento =======
    st.markdown("**Preprocesamiento y FE**")
    cprep = st.columns(3)
    with cprep[0]:
        use_advanced = st.checkbox("Preprocesamiento avanzado (robusto)", value=True)
    with cprep[1]:
        yeoj = st.checkbox("Yeo-Johnson", value=True)
    with cprep[2]:
        minfreq = st.slider("OneHot min_frequency", 0.0, 0.10, 0.01, 0.01)

    add_interaction = st.checkbox("Interacción (producto) del par numérico más correlacionado", value=True)

    # ======= Exploración (opcional): PCA + DBSCAN =======
    with st.expander("Exploración PCA/DBSCAN (opcional)"):
        can_pca = X_all.select_dtypes(include=np.number).shape[1] >= 2
        if can_pca and PLOTLY_OK:
            pca_df, pca_obj, _ = pca_2d(X_all)
            if pca_df is not None:
                st.plotly_chart(px.scatter(pca_df, x="PC1", y="PC2", title="PCA (2D)",
                                           template=PLOTLY_TEMPLATE), use_container_width=True)
        else:
            st.caption("PCA no disponible (se requieren ≥2 columnas numéricas).")

        # DBSCAN
        eps = st.slider("DBSCAN eps", 0.1, 3.0, 0.7, 0.1)
        min_s = st.slider("DBSCAN min_samples", 3, 50, 10, 1)
        excluir_outliers = st.checkbox("Excluir outliers DBSCAN antes de entrenar (no supervisado)", value=False)
        lab, out_mask = dbscan_outliers(X_all, eps=eps, min_samples=min_s)
        if lab is not None:
            out_count = int(out_mask.sum())
            st.write(f"Etiquetas DBSCAN: clusters={len(set(lab)) - (1 if (-1 in lab) else 0)}, outliers={out_count}")
            if can_pca and PLOTLY_OK:
                # colorear por cluster; resaltar -1
                pca_df, _, _ = pca_2d(X_all)
                color = pd.Series(lab, dtype="int").astype(str).replace({"-1": "outlier"})
                fig = px.scatter(pca_df, x="PC1", y="PC2", color=color,
                                 title="PCA coloreado por DBSCAN (outlier=-1)", template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("DBSCAN no disponible (sin columnas numéricas).")

    # ======= Distribución global de la etiqueta =======
    if tarea == "clasificación":
        st.markdown("**Distribución de la etiqueta (global):**")
        vc_abs = y_all.astype(str).value_counts()
        vc_pct = (y_all.astype(str).value_counts(normalize=True) * 100).round(1)
        st.dataframe(pd.DataFrame({"conteo": vc_abs, "porcentaje_%": vc_pct}),
                     use_container_width=True, height=220)

    # ======= Entrenamiento =======
    if st.button("Entrenar baseline"):
        # Opción: excluir outliers antes de split (aviso de que es no supervisado)
        X_use, y_use = X_all, y_all
        if 'excluir_outliers' in locals() and excluir_outliers and (out_mask is not None):
            keep = ~out_mask
            removed = int(out_mask.sum())
            X_use = X_all.loc[keep].reset_index(drop=True)
            y_use = y_all.loc[keep].reset_index(drop=True)
            st.warning(f"Se excluyeron {removed} filas marcadas como outliers por DBSCAN (no supervisado).")

        # Split robusto
        X_tr, X_te, y_tr, y_te, split_info = safe_train_test_split(
            X_use, y_use, tarea=tarea,
            test_size=float(test_size), random_state=42,
            min_per_class=2, auto_increase_test_size=bool(force_strat), max_test_size=float(max_ts),
        )

        # Interacción (en TRAIN/TEST para evitar fugas)
        from eda import find_top_corr_pair
        if add_interaction:
            pair = find_top_corr_pair(X_tr)
            if pair:
                a, b = pair
                X_tr = X_tr.copy(); X_te = X_te.copy()
                X_tr[f"{a}_x_{b}"] = pd.to_numeric(X_tr[a], errors="coerce") * pd.to_numeric(X_tr[b], errors="coerce")
                X_te[f"{a}_x_{b}"] = pd.to_numeric(X_te[a], errors="coerce") * pd.to_numeric(X_te[b], errors="coerce")
                st.info(f"Se añadió interacción: **{a} × {b}**")

        # Preprocesador
        if use_advanced:
            pre = make_preprocessor_advanced(X_tr, robust=True, yeo_johnson=bool(yeoj),
                                             onehot_min_freq=(float(minfreq) if minfreq>0 else None))
        else:
            pre = make_preprocessor(X_tr)

        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("prep", pre), ("model", modelo)])

        # Feedback split
        colm = st.columns(3)
        with colm[0]:
            st.metric("Clases", split_info.get("n_classes") or (y_use.nunique() if tarea=="clasificación" else "-"))
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
        st.session_state["ml"] = {"pipeline": pipe, "X_te": X_te, "y_te": y_te, **ml_art, "modelo_name": modelo_name}

        # KPIs grandes
        from ui_theme import render_kpi_cards
        # ===== KPI GRANDES (con % cuando aplica) =====
        ml = st.session_state["ml"]
        if tarea == "clasificación":
            items = [
                ("Accuracy (test)", f"{ml['acc_te']*100:.1f}%", f"Train: {ml['acc_tr']*100:.1f}%"),
                ("F1-macro (test)", f"{ml['f1_te']*100:.1f}%", f"Train: {ml['f1_tr']*100:.1f}%"),
                ("Clases", f"{len(ml['labels'])}", "en test"),
            ]
        else:
            items = [
                ("RMSE (test)", f"{ml['rmse']:.3f}", f"Baseline media: {ml['rmse_baseline']:.3f}"),
                ("MAE (test)", f"{ml['mae']:.3f}", ""),
                ("R² (test)", f"{ml['r2']*100:.1f}%", ""),
            ]
        render_kpi_cards(items, caption="Resultados del entrenamiento")

    # ======= Visualizaciones y resumen final =======
    if "ml" in st.session_state and st.session_state["ml"]["tarea"] == tarea:
        ml = st.session_state["ml"]
        if tarea == "clasificación" and PLOTLY_OK:
            etiquetas = ml["labels"]; cm = ml["cm"]
            fig_cm = px.imshow(cm, x=etiquetas, y=etiquetas, text_auto=True, color_continuous_scale="Greens",
                               title=f"Matriz de confusión – {ml.get('modelo_name','')}", template=PLOTLY_TEMPLATE)
            fig_cm.update_layout(xaxis_title="Predicha", yaxis_title="Real")
            st.plotly_chart(fig_cm, use_container_width=True)
        elif tarea != "clasificación" and PLOTLY_OK:
            resid = ml["y_te"] - ml["pred_te"]
            st.plotly_chart(px.scatter(x=ml["pred_te"], y=resid, labels={"x":"Predicción","y":"Residual"},
                                       title=f"Predicción vs Residual – {ml.get('modelo_name','')}", template=PLOTLY_TEMPLATE), use_container_width=True)
            st.plotly_chart(px.histogram(resid, nbins=40, title="Histograma de residuales",
                                         template=PLOTLY_TEMPLATE), use_container_width=True)

        imp = ml["perm_importance"]
        st.write("**Importancias por permutación (top):**")
        st.dataframe(imp, use_container_width=True, height=240)
        if PLOTLY_OK and not imp.empty:
            st.plotly_chart(px.bar(imp, x="feature", y="importance", error_y="std",
                                   title="Importancia (perm)", template=PLOTLY_TEMPLATE),
                            use_container_width=True)

        # ——— Resumen en bullets de los insights de modelado ———
        st.subheader("📌 Resumen de modelado (insights)")
        bullets = ml_bullet_summary(ml)
        st.markdown("\n".join(f"- {b}" for b in bullets))



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
