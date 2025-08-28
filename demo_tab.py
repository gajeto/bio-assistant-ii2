# demo_tab.py
from __future__ import annotations
import os
import math
import pandas as pd
import streamlit as st

from llm_groq import GroqLLM, build_system_prompt
from ml_utils import ml_key_insights
from utils_data import eda_summary_text
try:
    from plotly import express as px
    PLOTLY_OK = True
except Exception as e:
    PLOTLY_OK = False
    PX_ERR = e


def _scrollbox(md_text: str, height: int = 360):
    safe = (md_text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace("\n", "<br/>"))
    st.markdown(
        f"""<div style="
                max-height:{int(height)}px;
                overflow:auto;
                padding:.85rem 1rem;
                background: rgba(255,255,255,.92);
                border-left:4px solid #16a34a; border-radius: 8px;
                white-space: pre-wrap; line-height:1.45;">
                {safe}
            </div>""",
        unsafe_allow_html=True
    )


def _estimate_tokens(s: str) -> int:
    """
    Estimación sencilla de tokens (aprox.): 1 token ≈ 4 caracteres.
    Esto NO es exacto; sirve para dar porcentajes orientativos.
    """
    return max(1, math.ceil(len(s) / 4))


def _context_breakdown(context_text: str, answer_text: str) -> tuple[float, float, dict]:
    """
    Devuelve (pct_context, pct_modelo, detalle).
    pct_context ≈ context_tokens / (context_tokens + answer_tokens)
    pct_modelo  ≈ answer_tokens / (context_tokens + answer_tokens)
    """
    c = _estimate_tokens(context_text)
    a = _estimate_tokens(answer_text)
    total = c + a
    pct_c = round(100 * c / total, 1)
    pct_a = round(100 * a / total, 1)
    return pct_c, pct_a, {"context_tokens_est": c, "answer_tokens_est": a, "total_est": total}


def _eda_support_charts(df: pd.DataFrame, PLOTLY_TEMPLATE: str):
    """Pequeño set de gráficas EDA para respaldar respuestas."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    miss = df.isna().mean().mul(100).sort_values(ascending=False)
    miss_tbl = miss[miss > 0].reset_index()
    miss_tbl.columns = ["columna", "faltantes_%"]

    if PLOTLY_OK and not miss_tbl.empty:
        st.plotly_chart(
            px.bar(miss_tbl.head(20), x="columna", y="faltantes_%",
                   title="% faltantes (Top 20)", template=PLOTLY_TEMPLATE),
            use_container_width=True
        )

    if PLOTLY_OK and len(num_cols) >= 2:
        corr = df[num_cols[:12]].corr(numeric_only=True)
        st.plotly_chart(
            px.imshow(corr, text_auto=False, aspect="auto",
                      title="Matriz de correlación (subset numérico)", template=PLOTLY_TEMPLATE),
            use_container_width=True
        )


def _ml_support_charts(ml: dict, PLOTLY_TEMPLATE: str):
    """Gráficas/artefactos del baseline ML si existen."""
    if not ml:
        st.info("Aún no hay resultados de ML para mostrar.")
        return
    if ml["tarea"] == "clasificación":
        if PLOTLY_OK:
            etiquetas = ml["labels"]
            cm = ml["cm"]
            fig_cm = px.imshow(cm, x=etiquetas, y=etiquetas, text_auto=True,
                               color_continuous_scale="Greens",
                               title="Matriz de confusión (test)", template=PLOTLY_TEMPLATE)
            fig_cm.update_layout(xaxis_title="Predicha", yaxis_title="Real")
            st.plotly_chart(fig_cm, use_container_width=True)
    else:
        if PLOTLY_OK:
            resid = ml["y_te"] - ml["pred_te"]
            st.plotly_chart(
                px.scatter(x=ml["pred_te"], y=resid,
                           labels={"x": "Predicción", "y": "Residual"},
                           title="Predicción vs Residual", template=PLOTLY_TEMPLATE),
                use_container_width=True
            )
            st.plotly_chart(
                px.histogram(resid, nbins=40, title="Histograma de residuales",
                             template=PLOTLY_TEMPLATE),
                use_container_width=True
            )


def render_demo_tab(df: pd.DataFrame, eda: dict, PLOTLY_TEMPLATE: str = "plotly"):
    """
    Demo comparativa:
      - 3 columnas: Sin contexto | Con EDA | EDA + ML
      - Chat en cada columna (no streaming) para evitar scroll con texto incompleto
      - Cuantificación (aprox.) del peso del contexto vs. modelo
      - Gráficas de soporte EDA/ML si aplica
    """
    st.subheader("Comparador: Sin contexto vs. EDA vs. EDA + ML")

    # ===== Preparar LLM =====
    model_id = st.session_state.get("llm_model_id") or "llama-3.1-8b-instant"
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    temp = float(st.session_state.get("llm_temp", 0.2))
    max_tokens = int(st.session_state.get("llm_max_tokens", 350))
    llm = GroqLLM(model_id=model_id, api_key=api_key, temp=temp, max_tokens=max_tokens)

    # ===== Mensaje del usuario para la comparación =====
    pregunta = st.text_input(
        "Escribe una pregunta para comparar cómo responde el LLM en los tres escenarios",
        value="¿Qué problemas de calidad de datos ves y qué pasos recomiendas antes de entrenar un modelo?"
    )
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        ejecutar = st.button("Comparar respuestas")

    # ===== Preparar contextos =====
    eda_text = eda_summary_text(eda, df) if eda else ""
    ml_text = ""
    if "ml" in st.session_state:
        ml = st.session_state["ml"]
        try:
            ml_ins = ml_key_insights(ml, st.session_state["ml"]["X_te"])
            ml_text = "INSIGHTS DE ML:\n" + "\n".join(f"- {b}" for b in ml_ins) + "\n"
        except Exception:
            ml_text = ""

    # ===== Tres columnas lado a lado =====
    c_sin, c_eda, c_full = st.columns(3)

    # ----------------- SIN CONTEXTO -----------------
    with c_sin:
        st.markdown("### 🟢 Sin contexto")
        if ejecutar and pregunta.strip():
            sistema_sin = build_system_prompt("", "")
            respuesta = llm.ask(sistema_sin, pregunta)  # NO streaming → evita scroll parcial
            _scrollbox(respuesta, height=360)

            # Cuantificación aproximada
            pct_ctx, pct_model, det = _context_breakdown("", respuesta)
            st.caption(f"**Uso de contexto (aprox.):** {pct_ctx}% contexto / {pct_model}% modelo "
                       f"(tokens ~ ctx:{det['context_tokens_est']}, resp:{det['answer_tokens_est']})")

        with st.expander("Ver soporte EDA/ML (si aplica)"):
            st.info("Este panel no incluye contexto EDA/ML.")

    # ----------------- CON EDA -----------------
    with c_eda:
        st.markdown("### 🟢 Con EDA")
        if ejecutar and pregunta.strip():
            sistema_eda = build_system_prompt(eda_text, "")
            respuesta = llm.ask(sistema_eda, pregunta)
            _scrollbox(respuesta, height=360)

            # Cuantificación aproximada
            pct_ctx, pct_model, det = _context_breakdown(eda_text, respuesta)
            st.caption(f"**Uso de contexto (aprox.):** {pct_ctx}% contexto / {pct_model}% modelo "
                       f"(tokens ~ ctx:{det['context_tokens_est']}, resp:{det['answer_tokens_est']})")

        with st.expander("Gráficas EDA usadas como soporte"):
            _eda_support_charts(df, PLOTLY_TEMPLATE)

    # ----------------- EDA + ML -----------------
    with c_full:
        st.markdown("### 🟢 EDA + ML")
        if ejecutar and pregunta.strip():
            sistema_full = build_system_prompt(eda_text, ml_text)
            respuesta = llm.ask(sistema_full, pregunta)
            _scrollbox(respuesta, height=360)

            # Cuantificación aproximada
            pct_ctx, pct_model, det = _context_breakdown(eda_text + "\n" + ml_text, respuesta)
            st.caption(f"**Uso de contexto (aprox.):** {pct_ctx}% contexto / {pct_model}% modelo "
                       f"(tokens ~ ctx:{det['context_tokens_est']}, resp:{det['answer_tokens_est']})")

        with st.expander("Gráficas/artefactos del baseline ML"):
            _ml_support_charts(st.session_state.get("ml", {}), PLOTLY_TEMPLATE)

    # Nota de transparencia metodológica
    st.caption(
        "ℹ️ Los porcentajes de “uso de contexto” son estimaciones basadas en longitud de texto; "
        "no equivalen a métricas de atención internas del modelo ni a contadores reales de tokens."
    )
