# ui_theme.py
import streamlit as st

MAIN_BG = "https://images.unsplash.com/photo-1559757175-08ea0eac0e3f?q=80&w=1600&auto=format&fit=crop"
SIDE_BG = "https://images.unsplash.com/photo-1583912268180-6ec9a24b8301?q=80&w=1200&auto=format&fit=crop"

def apply_bio_theme():
    """
    Inyecta CSS: tema verde y fondos con overlay.
    También define una clase .scrollbox para textos largos con scroll.
    """
    css = f"""
    <style>
    /* Fondo principal con overlay verdoso */
    .stApp {{
        background: linear-gradient(rgba(22,163,74,0.10), rgba(6,95,70,0.15)),
                    url('{MAIN_BG}');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    /* Sidebar con fondo suavizado */
    section[data-testid="stSidebar"] > div:first-child {{
        background: linear-gradient(rgba(16,185,129,0.20), rgba(6,78,59,0.35)),
                    url('{SIDE_BG}');
        background-size: cover;
        background-position: center;
    }}
    /* Botones en verde */
    .stButton > button {{
        background-color: #16a34a !important; /* emerald-600 */
        border: 0;
        color: white !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.15);
    }}
    .stButton > button:hover {{
        filter: brightness(0.95);
    }}
    /* Cajas con scroll para respuestas largas */
    .scrollbox {{
        max-height: 360px;
        overflow-y: auto;
        padding: 0.8rem 1rem;
        background: rgba(255,255,255,0.80);
        border-left: 4px solid #16a34a;
        border-radius: 6px;
    }}
    /* Variante oscura para cuando las respuestas incluyan fondo */
    .dark .scrollbox {{
        background: rgba(0,0,0,0.55);
        color: #e5e7eb;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def scrollable_md(markdown_text: str, dark: bool=False, height: int=360):
    """
    Renderiza markdown con un contenedor scrollable.
    """
    h = max(240, int(height))
    class_extra = "dark" if dark else ""
    st.markdown(
        f"""<div class="{class_extra}"><div class="scrollbox" style="max-height:{h}px">{markdown_text}</div></div>""",
        unsafe_allow_html=True
    )
