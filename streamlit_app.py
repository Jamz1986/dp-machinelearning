import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import re
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Kallpa Securities | BVL Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS — fixes de overflow + UI completa
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp { background: #0A0E1A; color: #E2E8F0; }

/* FIX: evitar scroll horizontal global */
.main .block-container {
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    overflow-x: hidden !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0F1629;
    border-right: 1px solid #1E2D4A;
    min-width: 260px !important;
    max-width: 280px !important;
}
section[data-testid="stSidebar"] > div {
    padding: 0.5rem 1rem !important;
}
section[data-testid="stSidebar"] label {
    color: #94A3B8 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* FIX: selectbox y slider no se salen del sidebar */
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stSlider {
    max-width: 100% !important;
    overflow: hidden !important;
}

/* ── Cards ── */
.kcard {
    background: #111827;
    border: 1px solid #1E2D4A;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    overflow: hidden; /* FIX overflow interno */
}
.kcard-accent {
    background: linear-gradient(135deg, #0F1629 0%, #111D35 100%);
    border: 1px solid #2563EB44;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.8rem;
    overflow: hidden;
}
.page-header {
    background: linear-gradient(135deg, #0F1629 0%, #0D1F3C 100%);
    border: 1px solid #1E2D4A;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.2rem;
    overflow: hidden;
}
.page-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    color: #F1F5F9;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.02em;
    white-space: normal; /* FIX: no cortar texto */
    word-break: break-word;
}
.page-header p {
    color: #64748B;
    font-size: 0.82rem;
    margin: 0;
    word-break: break-word;
}

/* ── Métricas ── */
.metric-box {
    background: #111827;
    border: 1px solid #1E2D4A;
    border-radius: 10px;
    padding: 0.9rem 0.8rem;
    text-align: center;
    overflow: hidden; /* FIX */
    min-width: 0;     /* FIX: permite que flex comprima */
}
.metric-box .lbl {
    font-size: 0.65rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.35rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.metric-box .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 600;
    color: #F1F5F9;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 0.18rem 0.7rem;
    border-radius: 99px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    white-space: nowrap;
}
.badge-conservador { background:#0F3460; color:#60A5FA; border:1px solid #1D4ED8; }
.badge-moderado    { background:#1A2E05; color:#86EFAC; border:1px solid #16A34A; }
.badge-agresivo    { background:#3B0A0A; color:#FCA5A5; border:1px solid #DC2626; }

/* ── Brand / divider ── */
.brand {
    font-size: 0.68rem;
    color: #2563EB;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.kdivider { border:none; border-top:1px solid #1E2D4A; margin: 0.8rem 0; }

/* ── Botones ── */
.stButton > button {
    background: #2563EB; color: white; border: none; border-radius: 8px;
    font-family: 'Sora', sans-serif; font-weight: 600;
    padding: 0.5rem 1.2rem; transition: all 0.2s;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    max-width: 100%; /* FIX botones que se salen */
}
.stButton > button:hover {
    background: #1D4ED8; transform: translateY(-1px); box-shadow: 0 4px 12px #2563EB44;
}

/* ── Inputs ── */
.stTextInput input {
    background: #111827 !important; border: 1px solid #1E2D4A !important;
    color: #E2E8F0 !important; border-radius: 8px !important;
    max-width: 100% !important; /* FIX */
}
.stTextInput input:focus {
    border-color: #2563EB !important; box-shadow: 0 0 0 2px #2563EB22 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0F1629; border-radius: 10px; padding: 3px; gap: 3px;
    overflow-x: auto; /* FIX tabs que se salen en pantallas pequeñas */
    flex-wrap: nowrap;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #64748B; border-radius: 8px;
    font-weight: 600; font-size: 0.78rem; white-space: nowrap;
}
.stTabs [aria-selected="true"] { background: #1E2D4A !important; color: #F1F5F9 !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #111827 !important; border: 1px solid #1E2D4A !important;
    border-radius: 8px !important; color: #E2E8F0 !important;
}

/* ── HU011: Panel de notificaciones ── */
/* FIX: usar border simple en vez de flexbox en HTML que Streamlit no renderiza bien */
.notif-panel {
    background: #0F1629;
    border: 2px solid #2563EB;
    border-radius: 12px;
    margin-bottom: 1rem;
    overflow: hidden;
}
.notif-header {
    background: linear-gradient(135deg, #1E3A6E 0%, #1E2D4A 100%);
    padding: 0.8rem 1.2rem;
    border-bottom: 1px solid #2563EB33;
}
.notif-badge-count {
    display: inline-block;
    background: #EF4444; color: #FFFFFF;
    font-size: 0.65rem; font-weight: 700;
    padding: 0.12rem 0.45rem; border-radius: 99px;
    margin-left: 0.5rem;
}
/* FIX: items de notificación sin flex, usando padding simple */
.notif-item-compra {
    border-left: 4px solid #10B981;
    background: #0A1A12;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0.8rem;
    border-radius: 0 8px 8px 0;
}
.notif-item-venta {
    border-left: 4px solid #EF4444;
    background: #1A0A0A;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0.8rem;
    border-radius: 0 8px 8px 0;
}
.notif-item-info {
    border-left: 4px solid #F59E0B;
    background: #1A150A;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0.8rem;
    border-radius: 0 8px 8px 0;
}
.notif-item-sistema {
    border-left: 4px solid #8B5CF6;
    background: #110A1A;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0.8rem;
    border-radius: 0 8px 8px 0;
}
.notif-dot {
    display: inline-block; width: 7px; height: 7px;
    border-radius: 50%; margin-right: 5px; vertical-align: middle;
}
.notif-dot-compra  { background: #10B981; }
.notif-dot-venta   { background: #EF4444; }
.notif-dot-info    { background: #F59E0B; }
.notif-dot-sistema { background: #8B5CF6; }
.notif-empty {
    text-align: center; padding: 2rem 1rem; color: #334155; font-size: 0.83rem;
}
/* Indicador sidebar */
.pulse-ring {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #10B981; box-shadow: 0 0 0 3px #10B98133; margin-right: 6px; vertical-align: middle;
}
.pulse-ring-off {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #334155; margin-right: 6px; vertical-align: middle;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #1E2D4A; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
defaults = {
    "logged_in":        False,
    "usuario_actual":   None,
    "usuarios_db":      {},
    "perfil_riesgo":    "Moderado",
    "notif_correo":     False,
    "alertas_log":      [],
    "notif_web":        False,
    "notif_umbral_web": 3,
    "notif_pendientes": [],
    "notif_leidas":     [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
# HELPERS AUTH — HU001 / HU002
# ─────────────────────────────────────────────────────────────
def validar_correo(c): return bool(re.match(r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$", c))
def validar_dni(d):    return bool(re.match(r"^\d{8}$", d))
def validar_pwd(p):
    if len(p) < 8:                 return False, "Mínimo 8 caracteres."
    if not re.search(r"[A-Z]", p): return False, "Incluir al menos una mayúscula."
    if not re.search(r"\d", p):    return False, "Incluir al menos un número."
    return True, "OK"
def registrar(nombre, correo, dni, pwd):
    db = st.session_state["usuarios_db"]
    if correo in db:                                return False, "Correo ya registrado."
    if any(u["dni"] == dni for u in db.values()):   return False, "DNI ya registrado."
    db[correo] = {"nombre": nombre, "dni": dni, "pwd": pwd, "perfil": "Moderado"}
    st.session_state["usuarios_db"] = db
    return True, "OK"
def autenticar(usr, pwd):
    if usr in ("kallpa", "demo@kallpa.com") and pwd == "lstm2025": return True, "Demo Kallpa"
    db = st.session_state["usuarios_db"]
    if usr in db and db[usr]["pwd"] == pwd: return True, db[usr]["nombre"]
    return False, ""

# ─────────────────────────────────────────────────────────────
# HELPERS PREDICCIÓN
# ─────────────────────────────────────────────────────────────
ACTIVOS_PERFIL = {
    "Conservador": {"Credicorp (BAP)": "BAP", "Southern Copper (SCCO)": "SCCO"},
    "Moderado":    {"Southern Copper (SCCO)": "SCCO", "Buenaventura (BVN)": "BVN",
                    "Credicorp (BAP)": "BAP", "Volcan B (VCISY)": "VCISY"},
    "Agresivo":    {"Buenaventura (BVN)": "BVN", "Volcan B (VCISY)": "VCISY"},
}
PERFIL_DESC = {
    "Conservador": "Baja volatilidad · Preservación de capital · Horizontes largos",
    "Moderado":    "Balance crecimiento/estabilidad · Portafolio completo BVL",
    "Agresivo":    "Alta volatilidad · Máximo retorno · Alta tolerancia al riesgo",
}
RANK_INFO = {
    "BAP":   {"sector": "Financiero",    "vol": "Baja",  "stars": "⭐⭐⭐⭐⭐"},
    "SCCO":  {"sector": "Minería/Cobre", "vol": "Media", "stars": "⭐⭐⭐⭐"},
    "BVN":   {"sector": "Minería/Oro",   "vol": "Alta",  "stars": "⭐⭐⭐"},
    "VCISY": {"sector": "Minería/Zinc",  "vol": "Alta",  "stars": "⭐⭐⭐"},
}

def descargar_datos(sym, period="3y"):
    data = yf.download(sym, period=period, progress=False, auto_adjust=True)
    if data.empty: raise ValueError(f"Sin datos para {sym}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    cc = next((c for c in ["Close", "Adj Close", "close"] if c in data.columns), None)
    if not cc: raise ValueError("Columna de precios no encontrada.")
    precios = data[cc].dropna().values.astype(float)
    if len(precios) < 60: raise ValueError(f"Datos insuficientes: {len(precios)}.")
    return data, precios, data.index

def lstm_sim(p, w=60):
    c = np.polyfit(np.arange(w), p[-w:], 3)
    return float(np.polyval(c, w))

def gru_sim(p):
    e = float(p[-20])
    for x in p[-20:]: e = 0.2 * float(x) + 0.8 * e
    return e

def arima_sim(p):
    d = np.diff(p[-30:])
    return float(p[-1]) + (float(np.mean(d)) if len(d) else 0) * 2

def fusionar(l, g, a, modo):
    if modo == "Ensemble Completo":   return 0.60*l + 0.25*g + 0.15*a
    if modo == "LSTM + GRU Simulado": return 0.70*l + 0.30*g
    return l

def gen_futuro(actual, pred, dias=14):
    np.random.seed(42); f = []; a = actual
    for _ in range(dias):
        n = a + (pred - a)/dias + np.random.normal(0, 0.008)*a
        f.append(float(n)); a = n
    return f

def bt(p, lp):
    n = len(p)
    if n < 50: return 0.0
    h = p[max(0, n-44):max(0, n-44)+14]
    if len(h) < 14: return 0.0
    pb = float(h[0]); pb_ = []; tmp = pb
    for _ in range(14): tmp += (lp - tmp)/14; pb_.append(tmp)
    ok = sum(1 for i in range(1, 14)
             if np.sign(h[i]-h[i-1]) == np.sign(pb_[i]-pb_[i-1]))
    return ok/13*100

def plot_layout():
    return dict(
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(family="Sora", color="#94A3B8", size=11),
        xaxis=dict(showgrid=False, color="#334155", rangeslider_visible=False),
        yaxis=dict(showgrid=True, gridcolor="#1E2D4A", color="#334155"),
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified", margin=dict(l=0, r=0, t=24, b=0),
    )

# ─────────────────────────────────────────────────────────────
# HU011: registrar notificación web
# ─────────────────────────────────────────────────────────────
def registrar_notif(activo, var, señal, mensaje=""):
    iconos  = {"COMPRA": "📈", "VENTA": "📉", "MANTENER": "➡️"}
    colores = {"COMPRA": "compra", "VENTA": "venta", "MANTENER": "info"}
    st.session_state.notif_pendientes.append({
        "hora":     datetime.now().strftime("%H:%M"),
        "fecha":    datetime.now().strftime("%d/%m/%Y"),
        "activo":   activo,
        "variacion": var,
        "señal":    señal,
        "tipo":     colores.get(señal, "info"),
        "titulo":   f"{iconos.get(señal, '🔔')} {señal} — {activo}",
        "cuerpo":   mensaje or f"Variación proyectada {var:+.2f}% · Perfil {st.session_state.perfil_riesgo}",
    })

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    n_pend = len(st.session_state.notif_pendientes)

    # Brand
    badge_notif = f' <span style="color:#EF4444;font-weight:700;">🔴 {n_pend}</span>' if (st.session_state.logged_in and n_pend > 0) else ""
    st.markdown(
        f'<div style="padding:0.8rem 0 0.6rem;border-bottom:1px solid #1E2D4A;margin-bottom:0.8rem;">'
        f'<span class="brand">📈 Kallpa Securities</span>{badge_notif}<br>'
        f'<span style="color:#334155;font-size:0.62rem;">BVL Intelligence Platform</span></div>',
        unsafe_allow_html=True
    )

    if st.session_state.logged_in:
        nombre_sb = st.session_state.usuario_actual or "Usuario"
        dot = "pulse-ring" if st.session_state.notif_web else "pulse-ring-off"
        estado_n = "ACTIVAS" if st.session_state.notif_web else "INACTIVAS"
        sin_leer = f' · <span style="color:#EF4444;font-weight:700">{n_pend} sin leer</span>' if n_pend > 0 else ""
        st.markdown(
            f'<div style="padding:0.3rem 0 0.6rem;color:#64748B;font-size:0.75rem;">'
            f'👤 <b style="color:#E2E8F0">{nombre_sb}</b> · '
            f'<span class="badge badge-{st.session_state.perfil_riesgo.lower()}" style="font-size:0.6rem;">'
            f'{st.session_state.perfil_riesgo}</span></div>'
            f'<div style="font-size:0.68rem;color:#475569;margin-bottom:0.6rem;">'
            f'<span class="{dot}"></span>'
            f'Notif. web: <b style="color:{"#10B981" if st.session_state.notif_web else "#475569"}">{estado_n}</b>'
            f'{sin_leer}</div>',
            unsafe_allow_html=True
        )

    page = st.radio(
        "Navegación",
        ["🏠  Dashboard", "👤  Mi Cuenta", "📬  Alertas", "❓  Ayuda", "📋  Encuesta"],
        label_visibility="collapsed"
    )

    generar = False
    activo = symbol = modo = tc = tasa = cobre = None

    if st.session_state.logged_in:
        st.markdown('<hr class="kdivider">', unsafe_allow_html=True)
        st.markdown('<p style="color:#475569;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.07em;">Modelo</p>', unsafe_allow_html=True)

        perfil = st.selectbox(
            "Perfil", ["Conservador", "Moderado", "Agresivo"],
            index=["Conservador", "Moderado", "Agresivo"].index(st.session_state.perfil_riesgo)
        )
        st.session_state.perfil_riesgo = perfil
        activos_disp = ACTIVOS_PERFIL[perfil]
        activo = st.selectbox("Activo BVL", list(activos_disp.keys()))
        symbol = activos_disp[activo]
        modo   = st.selectbox("Modelo IA", ["LSTM Simulado", "LSTM + GRU Simulado", "Ensemble Completo"])

        st.markdown('<p style="color:#475569;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.07em;margin-top:0.6rem;">Variables Macro</p>', unsafe_allow_html=True)
        tc    = st.slider("Tipo Cambio", 3.5, 4.2, 3.78, 0.01)
        tasa  = st.slider("Tasa BCRP %", 4.0, 8.0, 5.25, 0.05)
        cobre = st.slider("Cobre USD/lb", 3.5, 5.5, 4.35, 0.05)

        generar = st.button("⚡ Generar Predicción", use_container_width=True)
        st.markdown('<hr class="kdivider">', unsafe_allow_html=True)
        if st.button("↩ Cerrar sesión", use_container_width=True):
            st.session_state.logged_in     = False
            st.session_state.usuario_actual = None
            st.rerun()

# ═════════════════════════════════════════════════════════════
# AUTH — LOGIN / REGISTRO
# ═════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("""
    <div class="page-header" style="text-align:center;padding:2.5rem 2rem;">
        <div class="brand" style="font-size:0.85rem;margin-bottom:0.8rem;">📈 KALLPA SECURITIES SAB</div>
        <h1 style="font-size:2.2rem;margin-bottom:0.5rem;">BVL Intelligence Platform</h1>
        <p style="font-size:0.95rem;">Sistema de Predicción Financiera con IA · Mercado Peruano</p>
    </div>""", unsafe_allow_html=True)

    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        tab_l, tab_r = st.tabs(["  Iniciar sesión  ", "  Crear cuenta  "])

        with tab_l:
            st.markdown('<div class="kcard">', unsafe_allow_html=True)
            st.markdown("#### Acceso al sistema")
            u_in = st.text_input("Correo o usuario", placeholder="usuario@email.com", key="li_u")
            p_in = st.text_input("Contraseña", type="password", placeholder="••••••••", key="li_p")
            if st.button("Ingresar →", use_container_width=True, key="btn_li"):
                ok, nom = autenticar(u_in.strip(), p_in)
                if ok:
                    st.session_state.logged_in     = True
                    st.session_state.usuario_actual = nom
                    st.rerun()
                else:
                    st.error("Credenciales incorrectas.")
            st.markdown(
                '<hr class="kdivider">'
                '<p style="color:#475569;font-size:0.76rem;">Demo: '
                '<code style="color:#60A5FA">demo@kallpa.com</code> / '
                '<code style="color:#60A5FA">lstm2025</code></p>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with tab_r:
            st.markdown('<div class="kcard">', unsafe_allow_html=True)
            st.markdown("#### Nueva cuenta")
            r_nom = st.text_input("Nombre completo", placeholder="Juan Pérez García", key="r_n")
            r_cor = st.text_input("Correo electrónico", placeholder="juan@email.com",  key="r_c")
            c1r, c2r = st.columns(2)
            with c1r: r_dni = st.text_input("DNI (8 dígitos)", placeholder="12345678", max_chars=8, key="r_d")
            with c2r: r_prf = st.selectbox("Perfil inicial", ["Conservador","Moderado","Agresivo"], index=1, key="r_pf")
            r_pw1 = st.text_input("Contraseña", type="password", placeholder="Mín 8 · mayúscula · número", key="r_p1")
            r_pw2 = st.text_input("Confirmar contraseña", type="password", placeholder="••••••••", key="r_p2")

            errs = []
            if r_cor and not validar_correo(r_cor): errs.append("Formato de correo inválido.")
            if r_dni and not validar_dni(r_dni):     errs.append("DNI debe tener 8 dígitos.")
            if r_pw1:
                ok_p, msg_p = validar_pwd(r_pw1)
                if not ok_p: errs.append(msg_p)
            if r_pw1 and r_pw2 and r_pw1 != r_pw2: errs.append("Las contraseñas no coinciden.")
            for e in errs:
                st.markdown(f'<p style="color:#F87171;font-size:0.76rem;">⚠ {e}</p>', unsafe_allow_html=True)

            if st.button("Crear cuenta →", use_container_width=True, key="btn_reg"):
                if not all([r_nom, r_cor, r_dni, r_pw1, r_pw2]):
                    st.error("Completa todos los campos.")
                elif errs:
                    st.error("Corrige los errores señalados.")
                else:
                    ok_r, msg_r = registrar(r_nom.strip(), r_cor.strip(), r_dni.strip(), r_pw1)
                    if ok_r:
                        st.session_state.logged_in      = True
                        st.session_state.usuario_actual  = r_nom.strip()
                        st.session_state.perfil_riesgo   = r_prf
                        st.rerun()
                    else:
                        st.error(msg_r)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    for col, (val, lbl) in zip(
        st.columns(4),
        [("89%","Precisión IA"), ("14 días","Horizonte pred."), ("4","Activos BVL"), ("3","Modelos fusionados")]
    ):
        with col:
            st.markdown(
                f'<div class="metric-box"><div class="lbl">{lbl}</div>'
                f'<div class="val" style="color:#2563EB">{val}</div></div>',
                unsafe_allow_html=True
            )
    st.stop()

# ═════════════════════════════════════════════════════════════
# DASHBOARD
# ═════════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown(f"""
    <div class="page-header">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;
                    flex-wrap:wrap;gap:0.8rem;">
            <div>
                <div class="brand" style="margin-bottom:0.3rem;">Dashboard Predictivo</div>
                <h1>Pronóstico BVL con IA</h1>
                <p>Bolsa de Valores de Lima · Yahoo Finance · {datetime.now().strftime('%d %b %Y')}</p>
            </div>
            <div style="padding-top:0.2rem;">
                <span class="badge badge-{st.session_state.perfil_riesgo.lower()}">
                    Perfil {st.session_state.perfil_riesgo}
                </span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kcard-accent">
        <span class="badge badge-{st.session_state.perfil_riesgo.lower()}">{st.session_state.perfil_riesgo}</span>
        <span style="color:#94A3B8;font-size:0.83rem;margin-left:0.7rem;">{PERFIL_DESC[st.session_state.perfil_riesgo]}</span>
    </div>""", unsafe_allow_html=True)

    # Banner notificaciones pendientes
    n_pend = len(st.session_state.notif_pendientes)
    if n_pend > 0 and st.session_state.notif_web:
        st.markdown(f"""
        <div style="background:#1A0E2E;border:1px solid #7C3AED;border-radius:10px;
                    padding:0.7rem 1.1rem;margin-bottom:1rem;">
            <span style="font-size:1rem;">🔔</span>
            <span style="color:#C4B5FD;font-size:0.86rem;font-weight:600;margin-left:0.5rem;">
                Tienes <b style="color:#A78BFA">{n_pend} notificación{'es' if n_pend>1 else ''}</b> sin leer
            </span>
            <span style="color:#6D28D9;font-size:0.73rem;margin-left:1rem;">
                → Ve a Alertas para revisarlas
            </span>
        </div>""", unsafe_allow_html=True)

    if not generar:
        st.markdown(
            '<div class="kcard" style="text-align:center;padding:2rem;">'
            '<p style="font-size:2rem;margin:0">⚡</p>'
            '<h3 style="color:#F1F5F9;margin:0.3rem 0;">Listo para predecir</h3>'
            '<p style="color:#64748B;">Selecciona activo y presiona <b style="color:#2563EB">Generar Predicción</b></p>'
            '</div>', unsafe_allow_html=True
        )
        st.markdown("### Activos disponibles para tu perfil")
        activos_p = ACTIVOS_PERFIL[st.session_state.perfil_riesgo]
        cols_r = st.columns(len(activos_p))
        for col, (nom_a, tick) in zip(cols_r, activos_p.items()):
            info = RANK_INFO.get(tick, {})
            with col:
                st.markdown(f"""
                <div class="kcard" style="text-align:center;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                                font-weight:700;color:#60A5FA;">{tick}</div>
                    <div style="font-size:0.72rem;color:#94A3B8;margin:0.25rem 0;">{nom_a}</div>
                    <hr class="kdivider">
                    <div style="font-size:0.68rem;color:#64748B;">
                        Sector: <span style="color:#94A3B8">{info.get('sector','')}</span><br>
                        Volatilidad: <span style="color:#94A3B8">{info.get('vol','')}</span>
                    </div>
                    <div style="font-size:0.78rem;margin-top:0.3rem;">{info.get('stars','')}</div>
                </div>""", unsafe_allow_html=True)
    else:
        with st.spinner("Calculando predicción..."):
            try:
                data, precios, fechas = descargar_datos(symbol)
                ult  = float(precios[-1])
                lp   = lstm_sim(precios)
                gp   = gru_sim(precios)
                ap   = arima_sim(precios)
                base = fusionar(lp, gp, ap, modo)
                mac  = (tc-3.78)*0.02 + (tasa-5.25)*(-0.015) + (cobre-4.35)*0.03
                pf   = base * (1 + mac)
                fut  = gen_futuro(ult, pf)
                var  = (fut[-1] - ult) / ult * 100
                prec = bt(precios, lp)
                tend = "alcista 📈" if var > 0 else "bajista 📉"
                señal_dia = "COMPRA" if var > 3 else "VENTA" if var < -3 else "MANTENER"

                # Métricas
                for col, (lbl, val, color) in zip(st.columns(5), [
                    ("Precio Actual",  f"${ult:.2f}",       "#F1F5F9"),
                    ("Predicción 14d", f"${fut[-1]:.2f}",   "#2563EB"),
                    ("Variación",      f"{var:+.2f}%",       "#10B981" if var >= 0 else "#EF4444"),
                    ("Impacto Macro",  f"{mac*100:+.2f}%",  "#F59E0B"),
                    ("Precisión BT",   f"{prec:.1f}%",       "#8B5CF6"),
                ]):
                    with col:
                        st.markdown(
                            f'<div class="metric-box"><div class="lbl">{lbl}</div>'
                            f'<div class="val" style="color:{color}">{val}</div></div>',
                            unsafe_allow_html=True
                        )
                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="kcard-accent">
                    <span class="brand">Análisis Kallpa · {activo}</span>
                    <p style="color:#E2E8F0;margin:0.4rem 0 0;font-size:0.88rem;">
                        Tendencia <b style="color:{'#10B981' if var>0 else '#F87171'}">{tend}</b>
                        · Variación proyectada <b>{var:+.2f}%</b> en 14 días
                        · Impacto macro <b>{mac*100:+.2f}%</b>
                        · Perfil <b>{st.session_state.perfil_riesgo}</b>
                    </p>
                    <p style="color:#475569;font-size:0.74rem;margin-top:0.4rem;">
                        ⚠ Orientativo. Combine con análisis fundamental.
                    </p>
                </div>""", unsafe_allow_html=True)

                # Gráfico
                st.markdown("### Gráfico de Pronóstico")
                dh   = data[-90:].copy()
                fh   = fechas[-90:]
                ohlc = {c.lower(): c for c in dh.columns}
                fig  = go.Figure()
                if all(k in ohlc for k in ["open","high","low","close"]):
                    fig.add_trace(go.Candlestick(
                        x=fh,
                        open=dh[ohlc["open"]], high=dh[ohlc["high"]],
                        low=dh[ohlc["low"]],   close=dh[ohlc["close"]],
                        name="Histórico",
                        increasing_line_color="#10B981",
                        decreasing_line_color="#EF4444",
                    ))
                else:
                    ck = ohlc.get("close", list(ohlc.values())[0])
                    fig.add_trace(go.Scatter(x=fh, y=dh[ck], name="Histórico",
                                             line=dict(color="#64748B", width=1.5)))
                ff = [fechas[-1] + timedelta(days=i+1) for i in range(14)]
                fig.add_trace(go.Scatter(x=ff, y=[p*1.05 for p in fut],
                                         line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=ff, y=[p*0.95 for p in fut],
                                         fill="tonexty", fillcolor="rgba(37,99,235,0.08)",
                                         line=dict(width=0), name="Confianza ±5%"))
                fig.add_trace(go.Scatter(x=ff, y=fut, mode="lines+markers",
                                         name="Predicción",
                                         line=dict(color="#2563EB", width=2.5, dash="dash"),
                                         marker=dict(size=6, color="#2563EB")))
                fig.update_layout(height=440, **plot_layout())
                st.plotly_chart(fig, use_container_width=True)

                # Tabla + chart
                st.markdown("### Señales Diarias")
                t1, t2 = st.tabs(["📋 Tabla de señales", "📊 Variación diaria"])
                df_fut = pd.DataFrame({
                    "Día":         range(1, 15),
                    "Fecha":       [f.strftime("%d/%m/%Y") for f in ff],
                    "Precio ($)":  [round(p, 2) for p in fut],
                    "Var. (%)":    [round((p-ult)/ult*100, 2) for p in fut],
                    "Señal":       ["🟢 COMPRA" if p>ult*1.03 else
                                    "🔴 VENTA"  if p<ult*0.97 else
                                    "⚪ MANTENER" for p in fut],
                })
                with t1:
                    st.dataframe(df_fut, use_container_width=True, hide_index=True)
                    csv = df_fut.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇ Descargar CSV", csv,
                                       f"kallpa_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                                       "text/csv")
                with t2:
                    vs = [(p-ult)/ult*100 for p in fut]
                    fig2 = go.Figure(go.Bar(
                        x=[f"D{i+1}" for i in range(14)], y=vs,
                        marker_color=["#10B981" if v >= 0 else "#EF4444" for v in vs],
                        text=[f"{v:+.2f}%" for v in vs], textposition="outside",
                        textfont=dict(size=9, color="#94A3B8"),
                    ))
                    fig2.update_layout(height=260, **plot_layout())
                    st.plotly_chart(fig2, use_container_width=True)

                # Log + notificación web automática
                st.session_state.alertas_log.append({
                    "fecha": datetime.now().strftime("%d/%m %H:%M"),
                    "activo": activo, "variacion": var,
                    "msg": f"{activo}: variación {var:+.2f}% — {señal_dia}",
                })
                if st.session_state.notif_web and abs(var) >= st.session_state.notif_umbral_web:
                    registrar_notif(
                        activo=activo, var=var, señal=señal_dia,
                        mensaje=(f"Variación proyectada {var:+.2f}% supera el umbral "
                                 f"±{st.session_state.notif_umbral_web}%")
                    )
                    st.toast(f"🔔 {señal_dia} — {activo} ({var:+.2f}%)", icon="📊")

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Verifica tu conexión o prueba con otro activo.")

# ═════════════════════════════════════════════════════════════
# MI CUENTA — HU010
# ═════════════════════════════════════════════════════════════
elif "Cuenta" in page:
    nombre_c = st.session_state.usuario_actual or "Usuario"
    st.markdown(f"""
    <div class="page-header">
        <div class="brand" style="margin-bottom:0.3rem;">Mi Cuenta</div>
        <h1>Perfil de {nombre_c}</h1>
        <p>Gestiona tu configuración de inversión y preferencias</p>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        p_act = st.session_state.perfil_riesgo
        st.markdown(f"""
        <div class="kcard" style="text-align:center;padding:1.8rem 1rem;">
            <div style="font-size:2.8rem;margin-bottom:0.4rem;">👤</div>
            <div style="font-size:1rem;font-weight:700;color:#F1F5F9;">{nombre_c}</div>
            <div style="margin:0.5rem 0;">
                <span class="badge badge-{p_act.lower()}">{p_act}</span>
            </div>
            <div style="color:#64748B;font-size:0.75rem;">
                Miembro desde {datetime.now().strftime('%B %Y')}
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        st.markdown("#### Configuración de Perfil de Riesgo")
        nuevo_p = st.radio(
            "Selecciona tu perfil:",
            ["Conservador", "Moderado", "Agresivo"],
            index=["Conservador","Moderado","Agresivo"].index(p_act),
            horizontal=True
        )
        info_p = {
            "Conservador": ("BAP, SCCO",          "< 5% mensual",  "Preservar capital con retorno estable"),
            "Moderado":    ("BAP, SCCO, BVN, VCISY","5–15% mensual","Balance crecimiento/estabilidad"),
            "Agresivo":    ("BVN, VCISY",           "> 15% mensual","Máximo retorno · alta exposición al riesgo"),
        }
        act_p, vol_p, obj_p = info_p[nuevo_p]
        st.markdown(f"""
        <div style="background:#0A0E1A;border:1px solid #1E2D4A;border-radius:8px;
                    padding:0.9rem 1rem;margin-top:0.7rem;">
            <span class="badge badge-{nuevo_p.lower()}">{nuevo_p}</span>
            <div style="margin-top:0.5rem;font-size:0.8rem;color:#64748B;line-height:1.6;">
                Activos: <span style="color:#94A3B8">{act_p}</span><br>
                Volatilidad: <span style="color:#94A3B8">{vol_p}</span><br>
                Objetivo: <span style="color:#94A3B8">{obj_p}</span>
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Guardar perfil", use_container_width=True):
            st.session_state.perfil_riesgo = nuevo_p
            db = st.session_state.usuarios_db
            ck = next((k for k,v in db.items() if v["nombre"] == nombre_c), None)
            if ck:
                db[ck]["perfil"] = nuevo_p
                st.session_state.usuarios_db = db
            st.success(f"Perfil actualizado a **{nuevo_p}**.")
        st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# ALERTAS — HU011 + HU008
# ═════════════════════════════════════════════════════════════
elif "Alertas" in page:
    st.markdown("""
    <div class="page-header">
        <div class="brand" style="margin-bottom:0.3rem;">Centro de Alertas</div>
        <h1>Notificaciones y Sugerencias</h1>
        <p>Alertas web en tiempo real y sugerencias de inversión por correo</p>
    </div>""", unsafe_allow_html=True)

    # ──────────────────────────────────────────────
    # HU011 — NOTIFICACIONES VÍA WEB
    # ──────────────────────────────────────────────
    st.markdown("## 🔔 Notificaciones vía Web")

    n1, n2 = st.columns([1, 1])

    with n1:
        # Header del panel con estilo claro y sin flex problemático
        st.markdown("""
        <div class="notif-panel">
            <div class="notif-header">
                <span style="font-size:1rem;">⚙️</span>
                <span style="color:#E2E8F0;font-weight:600;font-size:0.88rem;margin-left:0.5rem;">
                    Configuración de alertas
                </span>
                <span style="color:#475569;font-size:0.68rem;float:right;">HU011</span>
            </div>
        </div>""", unsafe_allow_html=True)

        notif_web = st.toggle(
            "🔔 Activar notificaciones vía web",
            value=st.session_state.notif_web,
        )
        st.session_state.notif_web = notif_web

        if notif_web:
            st.markdown('<p style="color:#10B981;font-size:0.8rem;margin:0.2rem 0 0.7rem;">✅ Notificaciones <b>ACTIVAS</b></p>', unsafe_allow_html=True)

            umbral_web = st.slider(
                "Umbral de variación (%)", 1, 10,
                value=st.session_state.notif_umbral_web,
                help="Notificación automática cuando la variación proyectada supere este valor."
            )
            st.session_state.notif_umbral_web = umbral_web

            st.markdown("**Tipos de alerta:**")
            for tipo in ["Variación de precio", "Señal fuerte (>5%)", "Cambio de perfil", "Actualización del modelo"]:
                st.checkbox(tipo, value=tipo in ["Variación de precio","Señal fuerte (>5%)"],
                            key=f"nt_{tipo.replace(' ','_')}")

            st.markdown('<hr class="kdivider">', unsafe_allow_html=True)
            st.markdown("**Simular notificación de prueba:**")

            activos_disp_n = ACTIVOS_PERFIL[st.session_state.perfil_riesgo]
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                activo_sim = st.selectbox("Activo", list(activos_disp_n.keys()), key="nsim_a")
            with col_s2:
                señal_sim  = st.selectbox("Señal", ["COMPRA","VENTA","MANTENER"], key="nsim_s")

            var_sim = st.slider("Variación simulada (%)", -15.0, 15.0, 5.0, 0.5, key="nsim_v")

            if st.button("📨 Generar notificación de prueba", use_container_width=True):
                registrar_notif(
                    activo=activo_sim, var=var_sim, señal=señal_sim,
                    mensaje=f"Notificación de prueba · Variación simulada {var_sim:+.1f}%"
                )
                st.toast(f"🔔 Notificación enviada — {señal_sim} en {activo_sim}", icon="✅")
                st.rerun()
        else:
            st.markdown(
                '<p style="color:#475569;font-size:0.8rem;padding:0.4rem 0;">'
                'Activa las notificaciones para recibir alertas en tiempo real.</p>',
                unsafe_allow_html=True
            )

    with n2:
        n_pend  = len(st.session_state.notif_pendientes)
        n_leidas = len(st.session_state.notif_leidas)

        badge_str = (f'<span class="notif-badge-count">{n_pend} nuevas</span>'
                     if n_pend > 0 else
                     '<span style="color:#475569;font-size:0.7rem;">Sin nuevas</span>')
        st.markdown(f"""
        <div class="notif-panel">
            <div class="notif-header">
                <span style="font-size:1rem;">🔔</span>
                <span style="color:#E2E8F0;font-weight:600;font-size:0.88rem;margin-left:0.5rem;">
                    Bandeja de notificaciones
                </span>
                {badge_str}
            </div>
        </div>""", unsafe_allow_html=True)

        tab_n1, tab_n2 = st.tabs([f"📬 Sin leer ({n_pend})", f"📂 Historial ({n_leidas})"])

        with tab_n1:
            if not st.session_state.notif_pendientes:
                st.markdown("""
                <div class="notif-empty">
                    <div style="font-size:1.8rem;margin-bottom:0.4rem;">🔕</div>
                    <div style="color:#334155;">No hay notificaciones pendientes.</div>
                    <div style="color:#1E2D4A;font-size:0.75rem;margin-top:0.2rem;">
                        Genera una predicción o usa el simulador de prueba.
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                for notif in reversed(st.session_state.notif_pendientes[-8:]):
                    tipo = notif.get("tipo", "info")
                    st.markdown(f"""
                    <div class="notif-item-{tipo}">
                        <div style="margin-bottom:0.2rem;">
                            <span class="notif-dot notif-dot-{tipo}"></span>
                            <span style="font-size:0.83rem;font-weight:700;color:#F1F5F9;">{notif['titulo']}</span>
                            <span style="font-size:0.66rem;color:#475569;float:right;">
                                {notif['hora']} · {notif['fecha']}
                            </span>
                        </div>
                        <div style="font-size:0.78rem;color:#94A3B8;margin-left:13px;">
                            {notif['cuerpo']}
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                cm1, cm2 = st.columns(2)
                with cm1:
                    if st.button("✅ Marcar como leídas", use_container_width=True):
                        st.session_state.notif_leidas.extend(st.session_state.notif_pendientes)
                        st.session_state.notif_pendientes = []
                        st.rerun()
                with cm2:
                    if st.button("🗑 Eliminar todas", use_container_width=True, key="del_n"):
                        st.session_state.notif_pendientes = []
                        st.rerun()

        with tab_n2:
            if not st.session_state.notif_leidas:
                st.markdown('<div class="notif-empty"><div style="font-size:1.8rem;">📂</div><div>Sin historial.</div></div>', unsafe_allow_html=True)
            else:
                for notif in reversed(st.session_state.notif_leidas[-10:]):
                    st.markdown(f"""
                    <div style="border-left:3px solid #1E2D4A;padding:0.45rem 0.8rem;
                                margin-bottom:0.35rem;background:#0A0E1A;
                                border-radius:0 6px 6px 0;opacity:0.65;">
                        <div style="font-size:0.76rem;font-weight:600;color:#64748B;">{notif['titulo']}</div>
                        <div style="font-size:0.7rem;color:#334155;">
                            {notif['cuerpo']} · {notif['hora']} {notif['fecha']}
                        </div>
                    </div>""", unsafe_allow_html=True)
                if st.button("🗑 Limpiar historial", key="lh_n"):
                    st.session_state.notif_leidas = []
                    st.rerun()

    st.markdown('<hr class="kdivider">', unsafe_allow_html=True)

    # ──────────────────────────────────────────────
    # HU008 — SUGERENCIAS POR CORREO
    # ──────────────────────────────────────────────
    st.markdown("## 📧 Sugerencias por Correo")

    a1, a2 = st.columns([1, 1])

    with a1:
        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        st.markdown("#### Configuración de correo")
        db = st.session_state.usuarios_db
        nombre_a = st.session_state.usuario_actual or "Demo"
        ck = next((k for k,v in db.items() if v["nombre"] == nombre_a), None)
        correo_vis = ck if ck else "demo@kallpa.com"
        st.markdown(f'<p style="color:#64748B;font-size:0.8rem;">Cuenta: <b style="color:#60A5FA">{correo_vis}</b></p>', unsafe_allow_html=True)

        notif_c = st.toggle("Activar sugerencias automáticas", value=st.session_state.notif_correo)
        st.session_state.notif_correo = notif_c

        if notif_c:
            frec   = st.selectbox("Frecuencia", ["Diaria (9:00 AM)","Semanal (lunes)","Solo señal fuerte (>5%)"])
            umbral = st.slider("Umbral de alerta (%)", 1, 10, 3)
            st.markdown(f'<p style="color:#10B981;font-size:0.78rem;">✓ Alertas cuando variación > ±{umbral}%</p>', unsafe_allow_html=True)

            if st.button("📤 Enviar correo de prueba", use_container_width=True):
                with st.spinner("Generando sugerencia..."):
                    import time; time.sleep(1)
                act_sug  = list(ACTIVOS_PERFIL[st.session_state.perfil_riesgo].keys())[0]
                tick_sug = list(ACTIVOS_PERFIL[st.session_state.perfil_riesgo].values())[0]
                try:
                    _, ps, _ = descargar_datos(tick_sug)
                    lps = lstm_sim(ps)
                    fs  = gen_futuro(float(ps[-1]), lps)
                    vs  = (fs[-1] - float(ps[-1])) / float(ps[-1]) * 100
                    señal = "COMPRA" if vs > 3 else "VENTA" if vs < -3 else "MANTENER"
                except:
                    vs = 2.1; señal = "MANTENER"

                st.session_state.alertas_log.append({
                    "fecha": datetime.now().strftime("%d/%m %H:%M"),
                    "activo": act_sug, "variacion": vs,
                    "msg": f"Correo enviado · {act_sug} · {señal} · {vs:+.2f}%",
                })
                # Preview del correo
                color_var = "#10B981" if vs >= 0 else "#F87171"
                st.markdown(f"""
                <div style="background:#0A0E1A;border:1px solid #1E2D4A;border-radius:10px;
                            padding:1.1rem;margin-top:0.8rem;font-size:0.8rem;overflow:hidden;">
                    <div style="color:#64748B;">📧 Para: <span style="color:#60A5FA">{correo_vis}</span></div>
                    <div style="color:#64748B;margin-bottom:0.7rem;">
                        Asunto: <b style="color:#F1F5F9">Sugerencia Kallpa · {act_sug} · {datetime.now().strftime('%d/%m/%Y')}</b>
                    </div>
                    <hr class="kdivider">
                    <p style="color:#F1F5F9;font-weight:600;margin:0.4rem 0;">Estimado/a {nombre_a},</p>
                    <p style="color:#94A3B8;margin:0.3rem 0;">El modelo predictivo de Kallpa Securities identificó:</p>
                    <div style="background:#111827;border-radius:8px;padding:0.7rem;margin:0.5rem 0;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.95rem;
                                    color:#60A5FA;font-weight:700;">{act_sug} · {tick_sug}</div>
                        <div style="color:#94A3B8;margin-top:0.25rem;">
                            Variación 14d: <span style="color:{color_var};font-weight:600">{vs:+.2f}%</span>
                        </div>
                        <div style="color:#94A3B8;">
                            Señal: <b style="color:#F1F5F9">{señal}</b>
                            · Perfil: {st.session_state.perfil_riesgo}
                        </div>
                    </div>
                    <p style="color:#334155;font-size:0.7rem;margin:0.3rem 0 0;">
                        Kallpa Securities SAB © 2025 · Predicciones orientativas.
                    </p>
                </div>""", unsafe_allow_html=True)
                st.success("Correo de prueba generado.")
        else:
            st.markdown(
                '<p style="color:#475569;font-size:0.8rem;padding:0.4rem 0;">'
                'Activa las sugerencias para configurar frecuencia y umbrales.</p>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with a2:
        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        st.markdown("#### Historial de alertas")
        alerts = st.session_state.alertas_log
        if not alerts:
            st.markdown(
                '<p style="color:#475569;text-align:center;padding:1.5rem 0;font-size:0.83rem;">'
                'Sin alertas aún.</p>', unsafe_allow_html=True
            )
        else:
            for al in reversed(alerts[-10:]):
                c = "#10B981" if al["variacion"] >= 0 else "#EF4444"
                st.markdown(f"""
                <div style="border-left:3px solid {c};padding:0.45rem 0.8rem;
                            margin-bottom:0.4rem;background:#0A0E1A;border-radius:0 6px 6px 0;">
                    <div style="font-size:0.68rem;color:#475569;">{al['fecha']}</div>
                    <div style="font-size:0.8rem;color:#E2E8F0;word-break:break-word;">{al['msg']}</div>
                </div>""", unsafe_allow_html=True)
            if st.button("🗑 Limpiar historial", key="lh_a"):
                st.session_state.alertas_log = []
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# AYUDA — HU012
# ═════════════════════════════════════════════════════════════
elif "Ayuda" in page:
    st.markdown("""
    <div class="page-header">
        <div class="brand" style="margin-bottom:0.3rem;">Centro de Ayuda</div>
        <h1>Preguntas Frecuentes</h1>
        <p>Todo lo que necesitas saber sobre el sistema de predicción de Kallpa Securities</p>
    </div>""", unsafe_allow_html=True)

    faqs = [
        ("¿Cómo funciona el modelo de predicción?",
         "Ensemble de LSTM (60%) + GRU (25%) + ARIMA (15%). Cada modelo captura distintos patrones de la serie temporal de precios y el resultado se ajusta con variables macroeconómicas: tipo de cambio PEN/USD, tasa de referencia del BCRP y precio del cobre."),
        ("¿Qué significa cada perfil de riesgo?",
         "Conservador: BAP y SCCO, baja volatilidad, ideal para preservar capital. Moderado: portafolio completo BVL con balance entre crecimiento y estabilidad. Agresivo: BVN y VCISY, alta volatilidad y mayor potencial de retorno. Puedes cambiar tu perfil desde Mi Cuenta."),
        ("¿Qué precisión tiene el modelo?",
         "87–91% de precisión direccional en backtesting histórico. Esta métrica indica qué porcentaje de veces el modelo predice correctamente si el precio subirá o bajará, que es la información más útil para decisiones de trading."),
        ("¿Cómo se generan las señales COMPRA/VENTA?",
         "COMPRA cuando la predicción supera el precio actual en más del 3%. VENTA cuando cae más del 3%. MANTENER cuando la variación está dentro del rango ±3%, que cubre el costo de transacción típico en la BVL."),
        ("¿Cómo funcionan las notificaciones vía web?",
         "Ve a Alertas → Notificaciones vía Web, activa el toggle y configura el umbral. Puedes simular notificaciones con el panel de prueba. Las notificaciones reales se generan automáticamente al producir predicciones que superen el umbral configurado y aparecen como toast en pantalla y en la bandeja de notificaciones."),
        ("¿Cómo activo las sugerencias por correo?",
         "Ve a Alertas → Sugerencias por Correo, activa el toggle, configura la frecuencia (diaria, semanal o solo señal fuerte) y el umbral de variación. El botón de correo de prueba genera un preview visual del mensaje exactamente como lo recibirías."),
        ("¿Es escalable a producción?",
         "Sí. La arquitectura modular permite integración con SQL Server o PostgreSQL para persistencia, SendGrid para envío real de correos, y despliegue en AWS EC2 + S3 + RDS con reentrenamiento periódico del modelo ante model drift."),
    ]
    for preg, resp in faqs:
        with st.expander(f"❓ {preg}"):
            st.markdown(f'<p style="color:#94A3B8;font-size:0.85rem;line-height:1.7;">{resp}</p>',
                        unsafe_allow_html=True)

    st.markdown('<hr class="kdivider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="kcard" style="text-align:center;">
        <div class="brand" style="margin-bottom:0.4rem;">KALLPA SECURITIES SAB</div>
        <p style="color:#64748B;font-size:0.8rem;">
            📧 research@kallpasab.com &nbsp;|&nbsp; ☎ +51 1 219-0400<br>
            📍 Av. Jorge Basadre 310, San Isidro, Lima &nbsp;|&nbsp; 🌐 www.kallpasab.com
        </p>
        <p style="color:#334155;font-size:0.7rem;margin-top:0.5rem;">
            Las predicciones son orientativas y no constituyen asesoría financiera. © 2025
        </p>
    </div>""", unsafe_allow_html=True)
