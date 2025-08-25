# ui_theme.py (añade estas partes si no las tienes)
import streamlit as st

MAIN_BG = "https://images.unsplash.com/photo-1559757175-08ea0eac0e3f?q=80&w=1600&auto=format&fit=crop"
SIDE_BG = "https://images.unsplash.com/photo-1583912268180-6ec9a24b8301?q=80&w=1200&auto=format&fit=crop"

def apply_bio_theme():
    css = f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(22,163,74,0.10), rgba(6,95,70,0.15)), url('{MAIN_BG}');
        background-size: cover; background-attachment: fixed; background-position: center;
    }}
    section[data-testid="stSidebar"] > div:first-child {{
        background: linear-gradient(rgba(16,185,129,0.20), rgba(6,78,59,0.35)), url('{SIDE_BG}');
        background-size: cover; background-position: center;
    }}
    .stButton > button {{ background-color:#16a34a!important; border:0; color:white!important; box-shadow:0 1px 2px rgba(0,0,0,.15);}}
    .stButton > button:hover {{ filter:brightness(0.95); }}

    /* Scroll para respuestas largas */
    .scrollbox {{
        max-height: 420px;
        overflow-y: auto;
        padding: .8rem 1rem;
        background: rgba(255,255,255,.85);
        border-left: 4px solid #16a34a; border-radius: 6px;
        white-space: pre-wrap; line-height: 1.4;
    }}

    /* Tarjetas KPI grandes */
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
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def scrollable_md(markdown_text: str, dark: bool=False, height: int=420):
    h = max(240, int(height))
    class_extra = ""  # (puedes usar .dark si lo necesitas)
    st.markdown(
        f"""<div class="{class_extra}"><div class="scrollbox" style="max-height:{h}px">{markdown_text}</div></div>""",
        unsafe_allow_html=True
    )

def render_kpi_cards(items, caption: str | None = None):
    """
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
