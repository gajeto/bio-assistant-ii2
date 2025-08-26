# llm_groq.py
import os
import streamlit as st
from typing import Optional

try:
    from groq import Groq
    GROQ_OK = True
except Exception:
    GROQ_OK = False

# llm_groq.py
GROQ_MODEL_CATALOG = {
    "Llama 3.1 8B (rápido)": "llama-3.1-8b-instant",
    "Llama 3.3 70B (mejor calidad)": "llama-3.3-70b-versatile",
    "GPT-OSS 20B (open-weight)": "openai/gpt-oss-20b",
    "GPT-OSS 120B (open-weight)": "openai/gpt-oss-120b",
    "Mixtral 8x7B 32k": "mixtral-8x7b-32768",
}

ALLOWED_LINK_DOMAINS = [
    "ncbi.nlm.nih.gov", "ensembl.org", "gnomad.broadinstitute.org", "ebi.ac.uk",
    "who.int", "cdc.gov", "nature.com", "wikipedia.org", "clinvar"
]

def build_system_prompt(eda_summary: str, ml_block: str = "") -> str:
    """
    Pide enlaces adicionales al final, restringidos a dominios curados,
    para minimizar 'links inventados'.
    """
    return (
        "Eres un asistente de datos genéticos, y también de biología general que identifica especies por su descripción general y características biológicas. "
        "Además, puedes explicar conceptos biológicos o géneticos complejos"
        "Responde en **español**, breve y conservador. Puedes dar ejemplos de análogias si aplica. "
        "No des consejo médico. Usa el contexto provisto para cifras.\n\n"
        f"RESUMEN EDA:\n{eda_summary}\n\n{ml_block}\n"
        "Al final, agrega una sección Recursos adicionales (curados) con 2 a 4 enlaces "
        "de dominios web relacionados con biología o genética "
        "Si no hay un recurso pertinente, omite la sección."
    )

class GroqLLM:
    """Cliente Groq con modo normal y streaming."""
    def __init__(self, model_id: str, api_key: Optional[str], temp: float = 0.2, max_tokens: int = 350):
        self.model = model_id
        self.key = api_key
        self.temp = float(temp)
        self.max_tokens = int(max_tokens)

    def available(self) -> bool:
        return bool(self.key) and GROQ_OK

    def status_badge(self) -> str:
        if self.available():
            return f"🟢 **Groq conectado** — modelo: `{self.model}` · temp={self.temp} · máx_tokens={self.max_tokens}"
        elif not GROQ_OK:
            return "🔴 Groq SDK no instalado. Agrega `groq` a requirements.txt."
        return "🟠 Falta `GROQ_API_KEY` en *Secrets*."

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
