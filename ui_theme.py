# ui_theme.py
from __future__ import annotations
import streamlit as st

# Imágenes de fondo (libres en Unsplash)
MAIN_BG = "https://images.unsplash.com/photo-1559757175-08ea0eac0e3f?q=80&w=1600&auto=format&fit=crop"
SIDE_BG = "https://images.unsplash.com/photo-1583912268180-6ec9a24b8301?q=80&w=1200&auto=format&fit=crop"

# Ilustración representativa para la sidebar (ADN)
SIDEBAR_ILLUSTRATION = "https://images.unsplash.com/photo-1535930749574-1399327ce78f?q=80&w=800&auto=format&fit=crop"


def apply_bio_theme(show_sidebar_image: bool = True, sidebar_caption: str = "Análisis de datos biológicos / genéticos"):
    """
    Inyecta estilos del tema (fondos, tabs con color, scrollbox, KPI cards) y
    opcionalmente coloca una imagen representativa en la barra lateral.
    """
    css = f"""
    <style>
    /* ===== Fondo general del App ===== */
    .stApp {{
        background: linear-gradient(rgba(22,163,74,0.10), rgba(6,95,70,0.15)), url('{MAIN_BG}');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}

    /* ===== Fondo de la Sidebar ===== */
    section[data-testid="stSidebar"] > div:first-child {{
        background: linear-gradient(rgba(16,185,129,0.20), rgba(6,78,59,0.35)), url('{SIDE_BG}');
        background-size: cover;
        background-position: center;
    }}

    /* ===== Botones ===== */
    .stButton > button {{
        background-color:#16a34a!important;
        border:0!important;
        color:white!important;
        box-shadow:0 1px 2px rgba(0,0,0,.15)!important;
    }}
    .stButton > button:hover {{ filter:brightness(0.95)!important; }}

    /* ===== Inputs destacados ===== */
    .stSlider > div, .stTextInput > div > div > input {{
        border-color: rgba(22,163,74,.45)!important;
    }}

    /* ===== Tabs más visibles (fondo y tipografía grande) ===== */
    div[data-baseweb="tab-list"]{{
        background: linear-gradient(180deg, rgba(22,163,74,.12), rgba(6,95,70,.06));
        padding: 10px; border-radius: 14px; gap: 8px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.6);
        margin-bottom: 10px;
    }}
    button[role="tab"]{{
        font-size: 1.05rem !important;
        color: #064e3b !important;
        border-radius: 10px !important;
        padding: 8px 12px !important;
    }}
    button[role="tab"][aria-selected="true"]{{
        background: #16a34a !important;
        color: white !important;
        box-shadow: 0 3px 10px rgba(22,163,74,.25);
    }}

    /* ===== Contenedor con scroll para textos largos ===== */
    .scrollbox {{
        max-height: 420px;
        overflow-y: auto;
        padding: .85rem 1rem;
        background: rgba(255,255,255,.92);
        border-left: 4px solid #16a34a; border-radius: 8px;
        white-space: pre-wrap; line-height: 1.45;
    }}

    /* ===== Tarjetas KPI (grid flexible) ===== */
    .kpi-grid {{ display:grid; gap:12px; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); }}
    .kpi-card {{
        background: linear-gradient(180deg, rgba(22,163,74,.10), rgba(22,163,74,.03));
        border: 1px solid rgba(22,163,74,.25);
        border-radius:14px; padding:14px 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,.05);
    }}
    .kpi-title {{ font-size: .85rem; color:#065f46; margin:0 0 6px 0; font-weight:600; letter-spacing:.2px; }}
    .kpi-value {{ font-size: 2.2rem; font-weight:800; color:#065f46; margin:0; line-height:1; }}
    .kpi-sub {{ font-size:.85rem; color:#0f766e; margin-top:6px; }}

    /* ===== DataFrames y tablas más "respiradas" ===== */
    .stDataFrame [data-testid="stTable"] tbody tr td {{
        padding-top: 8px; padding-bottom: 8px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Imagen representativa en la barra lateral
    if show_sidebar_image:
        with st.sidebar:
            st.image(SIDEBAR_ILLUSTRATION, caption=sidebar_caption, use_container_width=True)


def scrollable_md(markdown_text: str, dark: bool = False, height: int = 420):
    """
    Renderiza markdown dentro de un contenedor desplazable (usa clase .scrollbox).
    """
    h = max(240, int(height))
    safe = (markdown_text.replace("&", "&amp;")
                           .replace("<", "&lt;")
                           .replace(">", "&gt;")
                           .replace("\n", "<br/>"))
    extra_bg = "background: rgba(30,41,59,.85);" if dark else "background: rgba(255,255,255,.92);"
    st.markdown(
        f"""<div class="scrollbox" style="max-height:{h}px; {extra_bg}">
                {safe}
            </div>""",
        unsafe_allow_html=True
    )


def render_kpi_cards(items: list[tuple[str, str, str | None]], caption: str | None = None):
    """
    Muestra tarjetas KPI.
    items = [(title, value, subtext), ...]
    """
    if caption:
        st.caption(caption)
    html = ['<div class="kpi-grid">']
    for title, value, sub in items:
        sub = sub or ""
        html.append(
            f'''<div class="kpi-card">
                   <div class="kpi-title">{title}</div>
                   <div class="kpi-value">{value}</div>
                   <div class="kpi-sub">{sub}</div>
                </div>'''
        )
    html.append("</div>")
    st.markdown("\n".join(html), unsafe_allow_html=True)
