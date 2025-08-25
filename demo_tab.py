# demo_tab.py
import os
import streamlit as st
import pandas as pd
from utils_data import eda_key_insights
from ml_utils import ml_key_insights, ml_objective_interpretation
from llm_groq import GroqLLM, build_system_prompt
from ui_theme import scrollable_md

def render_demo_tab(df: pd.DataFrame, eda: dict, PLOTLY_TEMPLATE: str):
    st.subheader("Demostración: ¿el LLM usa los insights del **EDA** y del **ML**?")

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

    # 2) Insights ML (si entrenaste en la pestaña ML)
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

    # 3) Pregunta y prompts
    pregunta_demo = st.text_input(
        "Pregunta para comparar respuestas (ideal: algo que use EDA y ML):",
        value="¿Qué problemas de calidad ves, qué variables son influyentes y cómo interpretarías el rendimiento del modelo?"
    )

    sistema_sin = (
        "Eres un asistente de datos genéticos. Responde en español, breve y conservador. "
        "Si te falta contexto, dilo explícitamente. Al final, agrega una sección **Recursos adicionales (curados)** "
        "con 2–4 enlaces solo de dominios: ncbi.nlm.nih.gov, ensembl.org, gnomad.broadinstitute.org, ebi.ac.uk, "
        "who.int, cdc.gov, nature.com, wikipedia.org, clinvar. Si no aplica, omite la sección."
    )

    # Reutiliza build_system_prompt para que el LLM siempre añada links curados
    from utils_data import eda_summary_text
    eda_resumen = eda_summary_text(eda, df)
    ml_block = ""
    if ml_insights:
        ml_block = "INSIGHTS DE ML:\n" + "\n".join(f"- {b}" for b in ml_insights) + "\nInterpretación ML: " + ml_interp_txt

    sistema_con_eda = build_system_prompt(eda_resumen, "")
    sistema_eda_ml = build_system_prompt(eda_resumen, ml_block)

    colA, colB, colC = st.columns(3)

    # Cliente Groq
    model_id = st.session_state.get("llm_model_id") or "llama-3.1-8b-instant"
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    temp = float(st.session_state.get("llm_temp", 0.2))
    max_tokens = int(st.session_state.get("llm_max_tokens", 350))
    llm = GroqLLM(model_id=model_id, api_key=api_key, temp=temp, max_tokens=max_tokens)

    with colA:
        st.write("### 🔵 Sin contexto")
        if st.button("Generar SIN contexto"):
            resp = llm.ask(sistema_sin, pregunta_demo)
            scrollable_md(resp, dark=False, height=360)

    with colB:
        st.write("### 🟢 Con EDA")
        if st.button("Generar con EDA"):
            resp = llm.ask(sistema_con_eda, pregunta_demo)
            scrollable_md(resp, dark=False, height=360)

    with colC:
        st.write("### 🟣 Con EDA + ML")
        if st.button("Generar con EDA + ML"):
            resp = llm.ask(sistema_eda_ml, pregunta_demo)
            scrollable_md(resp, dark=False, height=360)

    st.divider()
    st.write("### ✅ Evidencia de uso del contexto (heurística textual)")
    resp_eval = st.text_area("Pega aquí una respuesta (mejor la de EDA+ML) para evaluarla:")
    if resp_eval:
        # Heurísticas simples
        checks = {}
        nrows, ncols = eda["shape"]
        checks["Menciona #filas"] = str(nrows) in resp_eval
        missing = pd.Series(eda["null_pct"]).sort_values(ascending=False)
        top_missing_names = [str(k).lower() for k in missing[missing > 0].head(3).index]
        checks["Cita columnas con más nulos (EDA)"] = any(name in resp_eval.lower() for name in top_missing_names)
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
