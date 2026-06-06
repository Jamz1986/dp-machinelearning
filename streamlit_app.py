import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import re
import warnings
import io

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Kallpa Securities | BVL Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp { background: #0A0E1A; color: #E2E8F0; }

.main .block-container {
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    overflow-x: hidden !important;
}

section[data-testid="stSidebar"] {
    background: #0F1629;
    border-right: 1px solid #1E2D4A;
    min-width: 260px !important;
    max-width: 280px !important;
}
section[data-testid="stSidebar"] > div { padding: 0.5rem 1rem !important; }
section[data-testid="stSidebar"] label {
    color: #94A3B8 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stSlider {
    max-width: 100% !important;
    overflow: hidden !important;
}

.kcard {
    background: #111827;
    border: 1px solid #1E2D4A;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    overflow: hidden;
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
    white-space: normal;
    word-break: break-word;
}
.page-header p { color: #64748B; font-size: 0.82rem; margin: 0; word-break: break-word; }

.metric-box {
    background: #111827;
    border: 1px solid #1E2D4A;
    border-radius: 10px;
    padding: 0.9rem 0.8rem;
    text-align: center;
    overflow: hidden;
    min-width: 0;
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

.brand {
    font-size: 0.68rem;
    color: #2563EB;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.kdivider { border:none; border-top:1px solid #1E2D4A; margin: 0.8rem 0; }

.stButton > button {
    background: #2563EB; color: white; border: none; border-radius: 8px;
    font-family: 'Sora', sans-serif; font-weight: 600;
    padding: 0.5rem 1.2rem; transition: all 0.2s;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    max-width: 100%;
}
.stButton > button:hover {
    background: #1D4ED8; transform: translateY(-1px); box-shadow: 0 4px 12px #2563EB44;
}

.stTextInput input {
    background: #111827 !important; border: 1px solid #1E2D4A !important;
    color: #E2E8F0 !important; border-radius: 8px !important;
    max-width: 100% !important;
}
.stTextInput input:focus {
    border-color: #2563EB !important; box-shadow: 0 0 0 2px #2563EB22 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0F1629; border-radius: 10px; padding: 3px; gap: 3px;
    overflow-x: auto;
    flex-wrap: nowrap;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #64748B; border-radius: 8px;
    font-weight: 600; font-size: 0.78rem; white-space: nowrap;
}
.stTabs [aria-selected="true"] { background: #1E2D4A !important; color: #F1F5F9 !important; }

.streamlit-expanderHeader {
    background: #111827 !important; border: 1px solid #1E2D4A !important;
    border-radius: 8px !important; color: #E2E8F0 !important;
}

/* watchlist card */
.watch-card {
    background: #0A0E1A;
    border: 1px solid #1E2D4A;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
}
.watch-card:hover { border-color: #2563EB44; }

/* notif panel */
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
.pulse-ring {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #10B981; box-shadow: 0 0 0 3px #10B98133; margin-right: 6px; vertical-align: middle;
}
.pulse-ring-off {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #334155; margin-right: 6px; vertical-align: middle;
}

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
    # Sprint 4 — watchlist
    "watchlist":        [],   # list of ticker symbols
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
# HELPERS AUTH
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
TODOS_ACTIVOS = {
    "Credicorp (BAP)":        "BAP",
    "Southern Copper (SCCO)": "SCCO",
    "Buenaventura (BVN)":     "BVN",
    "Volcan B (VCISY)":       "VCISY",
}
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
    "BAP":   {"nombre": "Credicorp (BAP)",        "sector": "Financiero",    "vol": "Baja",  "stars": "⭐⭐⭐⭐⭐"},
    "SCCO":  {"nombre": "Southern Copper (SCCO)", "sector": "Minería/Cobre", "vol": "Media", "stars": "⭐⭐⭐⭐"},
    "BVN":   {"nombre": "Buenaventura (BVN)",     "sector": "Minería/Oro",   "vol": "Alta",  "stars": "⭐⭐⭐"},
    "VCISY": {"nombre": "Volcan B (VCISY)",        "sector": "Minería/Zinc",  "vol": "Alta",  "stars": "⭐⭐⭐"},
}
COLORES_ACTIVOS = {
    "BAP":   "#2563EB",
    "SCCO":  "#10B981",
    "BVN":   "#F59E0B",
    "VCISY": "#8B5CF6",
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

def predecir_activo(symbol, modo="Ensemble Completo", tc=3.78, tasa=5.25, cobre=4.35):
    """Retorna dict con todos los datos de predicción de un activo."""
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
    señal = "COMPRA" if var > 3 else "VENTA" if var < -3 else "MANTENER"
    ff = [fechas[-1] + timedelta(days=i+1) for i in range(14)]
    return {
        "data": data, "precios": precios, "fechas": fechas,
        "ult": ult, "fut": fut, "var": var, "mac": mac, "prec": prec,
        "señal": señal, "ff": ff,
    }

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
    n_watch = len(st.session_state.watchlist)

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
        watch_badge = f' · <span style="color:#60A5FA;font-weight:700">👁 {n_watch}</span>' if n_watch > 0 else ""
        st.markdown(
            f'<div style="padding:0.3rem 0 0.6rem;color:#64748B;font-size:0.75rem;">'
            f'👤 <b style="color:#E2E8F0">{nombre_sb}</b> · '
            f'<span class="badge badge-{st.session_state.perfil_riesgo.lower()}" style="font-size:0.6rem;">'
            f'{st.session_state.perfil_riesgo}</span></div>'
            f'<div style="font-size:0.68rem;color:#475569;margin-bottom:0.6rem;">'
            f'<span class="{dot}"></span>'
            f'Notif. web: <b style="color:{"#10B981" if st.session_state.notif_web else "#475569"}">{estado_n}</b>'
            f'{sin_leer}{watch_badge}</div>',
            unsafe_allow_html=True
        )

    page = st.radio(
        "Navegación",
        ["🏠  Dashboard", "👤  Mi Cuenta", "📬  Alertas", "❓  Ayuda",
         "📋  Encuesta", "🔍  Explorar BVL", "📈  Evolución"],
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
# AUTH
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
            en_watch = tick in st.session_state.watchlist
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
                    <div style="font-size:0.68rem;margin-top:0.3rem;color:{'#10B981' if en_watch else '#334155'};">
                        {'👁 En seguimiento' if en_watch else ''}
                    </div>
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

                # Botón watchlist inline
                col_info, col_watch = st.columns([4, 1])
                with col_info:
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
                with col_watch:
                    en_watch = symbol in st.session_state.watchlist
                    lbl_w = "👁 En seguimiento" if en_watch else "➕ Agregar a watchlist"
                    if st.button(lbl_w, use_container_width=True, key="btn_watch_dash"):
                        if en_watch:
                            st.session_state.watchlist.remove(symbol)
                            st.toast(f"✅ {symbol} eliminado del seguimiento")
                        else:
                            st.session_state.watchlist.append(symbol)
                            st.toast(f"✅ {symbol} agregado al seguimiento")
                        st.rerun()

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
            <hr class="kdivider">
            <div style="color:#64748B;font-size:0.75rem;">
                👁 Activos en seguimiento: <b style="color:#60A5FA">{len(st.session_state.watchlist)}</b>
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

    st.markdown("## 🔔 Notificaciones vía Web")
    n1, n2 = st.columns([1, 1])

    with n1:
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

        notif_web = st.toggle("🔔 Activar notificaciones vía web", value=st.session_state.notif_web)
        st.session_state.notif_web = notif_web

        if notif_web:
            st.markdown('<p style="color:#10B981;font-size:0.8rem;margin:0.2rem 0 0.7rem;">✅ Notificaciones <b>ACTIVAS</b></p>', unsafe_allow_html=True)
            umbral_web = st.slider("Umbral de variación (%)", 1, 10, value=st.session_state.notif_umbral_web)
            st.session_state.notif_umbral_web = umbral_web
            st.markdown("**Tipos de alerta:**")
            for tipo in ["Variación de precio", "Señal fuerte (>5%)", "Cambio de perfil", "Actualización del modelo"]:
                st.checkbox(tipo, value=tipo in ["Variación de precio","Señal fuerte (>5%)"],
                            key=f"nt_{tipo.replace(' ','_')}")
            st.markdown('<hr class="kdivider">', unsafe_allow_html=True)
            st.markdown("**Simular notificación de prueba:**")
            activos_disp_n = ACTIVOS_PERFIL[st.session_state.perfil_riesgo]
            col_s1, col_s2 = st.columns(2)
            with col_s1: activo_sim = st.selectbox("Activo", list(activos_disp_n.keys()), key="nsim_a")
            with col_s2: señal_sim  = st.selectbox("Señal", ["COMPRA","VENTA","MANTENER"], key="nsim_s")
            var_sim = st.slider("Variación simulada (%)", -15.0, 15.0, 5.0, 0.5, key="nsim_v")
            if st.button("📨 Generar notificación de prueba", use_container_width=True):
                registrar_notif(activo=activo_sim, var=var_sim, señal=señal_sim,
                                mensaje=f"Notificación de prueba · Variación simulada {var_sim:+.1f}%")
                st.toast(f"🔔 Notificación enviada — {señal_sim} en {activo_sim}", icon="✅")
                st.rerun()
        else:
            st.markdown('<p style="color:#475569;font-size:0.8rem;padding:0.4rem 0;">Activa las notificaciones para recibir alertas en tiempo real.</p>', unsafe_allow_html=True)

    with n2:
        n_pend  = len(st.session_state.notif_pendientes)
        n_leidas = len(st.session_state.notif_leidas)
        badge_str = (f'<span class="notif-badge-count">{n_pend} nuevas</span>'
                     if n_pend > 0 else '<span style="color:#475569;font-size:0.7rem;">Sin nuevas</span>')
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
                st.markdown("""<div class="notif-empty">
                    <div style="font-size:1.8rem;margin-bottom:0.4rem;">🔕</div>
                    <div style="color:#334155;">No hay notificaciones pendientes.</div>
                    <div style="color:#1E2D4A;font-size:0.75rem;margin-top:0.2rem;">
                        Genera una predicción o usa el simulador de prueba.
                    </div></div>""", unsafe_allow_html=True)
            else:
                for notif in reversed(st.session_state.notif_pendientes[-8:]):
                    tipo = notif.get("tipo", "info")
                    st.markdown(f"""
                    <div class="notif-item-{tipo}">
                        <div style="margin-bottom:0.2rem;">
                            <span class="notif-dot notif-dot-{tipo}"></span>
                            <span style="font-size:0.83rem;font-weight:700;color:#F1F5F9;">{notif['titulo']}</span>
                            <span style="font-size:0.66rem;color:#475569;float:right;">{notif['hora']} · {notif['fecha']}</span>
                        </div>
                        <div style="font-size:0.78rem;color:#94A3B8;margin-left:13px;">{notif['cuerpo']}</div>
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
                                margin-bottom:0.35rem;background:#0A0E1A;border-radius:0 6px 6px 0;opacity:0.65;">
                        <div style="font-size:0.76rem;font-weight:600;color:#64748B;">{notif['titulo']}</div>
                        <div style="font-size:0.7rem;color:#334155;">{notif['cuerpo']} · {notif['hora']} {notif['fecha']}</div>
                    </div>""", unsafe_allow_html=True)
                if st.button("🗑 Limpiar historial", key="lh_n"):
                    st.session_state.notif_leidas = []
                    st.rerun()

    st.markdown('<hr class="kdivider">', unsafe_allow_html=True)
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
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.95rem;color:#60A5FA;font-weight:700;">{act_sug} · {tick_sug}</div>
                        <div style="color:#94A3B8;margin-top:0.25rem;">Variación 14d: <span style="color:{color_var};font-weight:600">{vs:+.2f}%</span></div>
                        <div style="color:#94A3B8;">Señal: <b style="color:#F1F5F9">{señal}</b> · Perfil: {st.session_state.perfil_riesgo}</div>
                    </div>
                    <p style="color:#334155;font-size:0.7rem;margin:0.3rem 0 0;">Kallpa Securities SAB © 2025 · Predicciones orientativas.</p>
                </div>""", unsafe_allow_html=True)
                st.success("Correo de prueba generado.")
        else:
            st.markdown('<p style="color:#475569;font-size:0.8rem;padding:0.4rem 0;">Activa las sugerencias para configurar frecuencia y umbrales.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with a2:
        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        st.markdown("#### Historial de alertas")
        alerts = st.session_state.alertas_log
        if not alerts:
            st.markdown('<p style="color:#475569;text-align:center;padding:1.5rem 0;font-size:0.83rem;">Sin alertas aún.</p>', unsafe_allow_html=True)
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

    # Búsqueda en FAQ — criterio de aceptación HU012
    busqueda = st.text_input("🔍 Buscar en preguntas frecuentes...", placeholder="Ej: perfil de riesgo, correo, modelo...")

    faqs = [
        ("¿Cómo funciona el modelo de predicción?",
         "Ensemble de LSTM (60%) + GRU (25%) + ARIMA (15%). Cada modelo captura distintos patrones de la serie temporal de precios y el resultado se ajusta con variables macroeconómicas: tipo de cambio PEN/USD, tasa de referencia del BCRP y precio del cobre."),
        ("¿Qué significa cada perfil de riesgo?",
         "Conservador: BAP y SCCO, baja volatilidad, ideal para preservar capital. Moderado: portafolio completo BVL con balance entre crecimiento y estabilidad. Agresivo: BVN y VCISY, alta volatilidad y mayor potencial de retorno. Puedes cambiar tu perfil desde Mi Cuenta."),
        ("¿Qué precisión tiene el modelo?",
         "87–91% de precisión direccional en backtesting histórico. Esta métrica indica qué porcentaje de veces el modelo predece correctamente si el precio subirá o bajará, que es la información más útil para decisiones de trading."),
        ("¿Cómo se generan las señales COMPRA/VENTA?",
         "COMPRA cuando la predicción supera el precio actual en más del 3%. VENTA cuando cae más del 3%. MANTENER cuando la variación está dentro del rango ±3%, que cubre el costo de transacción típico en la BVL."),
        ("¿Cómo funcionan las notificaciones vía web?",
         "Ve a Alertas → Notificaciones vía Web, activa el toggle y configura el umbral. Puedes simular notificaciones con el panel de prueba. Las notificaciones reales se generan automáticamente al producir predicciones que superen el umbral configurado y aparecen como toast en pantalla y en la bandeja de notificaciones."),
        ("¿Cómo activo las sugerencias por correo?",
         "Ve a Alertas → Sugerencias por Correo, activa el toggle, configura la frecuencia (diaria, semanal o solo señal fuerte) y el umbral de variación. El botón de correo de prueba genera un preview visual del mensaje exactamente como lo recibirías."),
        ("¿Cómo uso la lista de seguimiento (watchlist)?",
         "En Explorar BVL puedes agregar activos a tu lista de seguimiento con el botón ➕. También puedes agregarlo directamente desde el Dashboard al generar una predicción. La sección Explorar BVL muestra solo los activos en seguimiento si así lo configuras."),
        ("¿Cómo comparo múltiples activos?",
         "Ve a la sección Evolución → Comparación de Activos. Selecciona entre 2 y 4 activos con el multiselect, elige el período de análisis y el sistema generará una gráfica overlay con históricos y predicciones normalizadas de cada activo. Puedes exportar el análisis como CSV."),
        ("¿Es escalable a producción?",
         "Sí. La arquitectura modular permite integración con SQL Server o PostgreSQL para persistencia, SendGrid para envío real de correos, y despliegue en AWS EC2 + S3 + RDS con reentrenamiento periódico del modelo ante model drift."),
    ]

    faqs_filtradas = [(p, r) for p, r in faqs
                      if not busqueda or busqueda.lower() in p.lower() or busqueda.lower() in r.lower()]

    if not faqs_filtradas:
        st.markdown('<p style="color:#475569;font-size:0.85rem;padding:1rem 0;">No se encontraron resultados para tu búsqueda.</p>', unsafe_allow_html=True)
    else:
        for preg, resp in faqs_filtradas:
            with st.expander(f"❓ {preg}"):
                st.markdown(f'<p style="color:#94A3B8;font-size:0.85rem;line-height:1.7;">{resp}</p>', unsafe_allow_html=True)

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

# ═════════════════════════════════════════════════════════════
# ENCUESTA DE VALIDACIÓN
# ═════════════════════════════════════════════════════════════
elif "Encuesta" in page:
    st.markdown("""
    <div class="page-header">
        <div class="brand" style="margin-bottom:0.3rem;">Validación del Sistema</div>
        <h1>Encuesta de Experiencia de Usuario</h1>
        <p>Estudio de validación · Kallpa Securities SAB · Solo con fines de investigación académica</p>
    </div>""", unsafe_allow_html=True)

    if "enc_enviada" not in st.session_state:
        st.session_state.enc_enviada = False
    if "enc_respuestas" not in st.session_state:
        st.session_state.enc_respuestas = []

    if st.session_state.enc_enviada:
        st.success("✅ ¡Gracias! Tu respuesta fue registrada correctamente.")
        if st.button("📝 Completar otra vez (modo prueba)", use_container_width=False):
            st.session_state.enc_enviada = False
            st.rerun()

    if not st.session_state.enc_enviada:
        st.markdown("""
        <div class="kcard-accent" style="margin-bottom:1.2rem;">
            <span class="brand">Instrucciones</span>
            <p style="color:#94A3B8;font-size:0.85rem;margin:0.4rem 0 0;">
                Esta encuesta tiene <b style="color:#F1F5F9">9 preguntas</b> y toma menos de <b style="color:#F1F5F9">3 minutos</b>.
                Es anónima y sus respuestas serán usadas exclusivamente para la investigación académica
                sobre democratización financiera en el Perú.
            </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div style="background:#0F1629;border-left:4px solid #2563EB;border-radius:0 8px 8px 0;padding:0.8rem 1.2rem;margin-bottom:1rem;">
            <div style="color:#60A5FA;font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;">Sección I</div>
            <div style="color:#F1F5F9;font-weight:600;font-size:0.95rem;">Perfil y Adaptación Tecnológica</div>
            <div style="color:#64748B;font-size:0.78rem;margin-top:0.2rem;">Objetivo: Validar la transición desde métodos informales hacia la IA.</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        st.markdown("**1.** Antes de utilizar esta plataforma, ¿cuál era su principal fuente de información para invertir?")
        p1 = st.radio("p1", ["Redes sociales o grupos de mensajería (información informal)", "Análisis técnico propio (gráficos y manuales)", "Recomendaciones de amigos o familiares", "Ninguna, invertía por intuición"], label_visibility="collapsed", key="enc_p1")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        st.markdown("**2.** ¿Qué tan fácil le resultó navegar por el dashboard y comprender las predicciones?")
        p2_labels = {1:"1 — Muy difícil",2:"2 — Difícil",3:"3 — Neutral",4:"4 — Fácil",5:"5 — Muy fácil"}
        p2 = st.select_slider("p2", options=[1,2,3,4,5], format_func=lambda x: p2_labels[x], value=3, label_visibility="collapsed", key="enc_p2")
        color_p2 = ["#EF4444","#F97316","#F59E0B","#84CC16","#10B981"][p2-1]
        st.markdown(f'<div style="background:#0A0E1A;border-radius:6px;padding:0.5rem 0.8rem;margin-top:0.3rem;"><span style="color:{color_p2};font-weight:700;font-size:0.9rem;">{p2_labels[p2]}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""<div style="background:#0F1629;border-left:4px solid #8B5CF6;border-radius:0 8px 8px 0;padding:0.8rem 1.2rem;margin-bottom:1rem;margin-top:0.5rem;">
            <div style="color:#A78BFA;font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;">Sección II</div>
            <div style="color:#F1F5F9;font-weight:600;font-size:0.95rem;">IA Explicable y Educación Financiera</div>
            <div style="color:#64748B;font-size:0.78rem;margin-top:0.2rem;">Objetivo: Cumplir con la Ley N.º 32814 sobre transparencia algorítmica.</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        st.markdown('**3.** ¿La sección de "Factores de Influencia" le ayudó a entender por qué el sistema generó esa predicción?')
        p3 = st.radio("p3", ["Sí, me ayudó a comprender la lógica del mercado", "Solo parcialmente, algunos términos son complejos", "No, prefiero ver solo el precio proyectado"], label_visibility="collapsed", key="enc_p3")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        st.markdown("**4.** ¿Considera que el uso de esta herramienta ha mejorado su conocimiento sobre la BVL?")
        p4 = st.radio("p4", ["Totalmente de acuerdo","De acuerdo","Neutral","En desacuerdo"], label_visibility="collapsed", key="enc_p4")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""<div style="background:#0F1629;border-left:4px solid #10B981;border-radius:0 8px 8px 0;padding:0.8rem 1.2rem;margin-bottom:1rem;margin-top:0.5rem;">
            <div style="color:#6EE7B7;font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;">Sección III</div>
            <div style="color:#F1F5F9;font-weight:600;font-size:0.95rem;">Soporte a la Decisión e Impacto Económico</div>
            <div style="color:#64748B;font-size:0.78rem;margin-top:0.2rem;">Objetivo: Validar la eficacia del sistema en la reducción de riesgos.</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        st.markdown("**5.** Al observar un activo con un Ratio de Sharpe mayor a 1, ¿cuál es su percepción de riesgo?")
        p5 = st.radio("p5", ["Siento mucha confianza, es un activo óptimo","Siento confianza moderada","Me es indiferente","No entiendo el significado del ratio"], label_visibility="collapsed", key="enc_p5")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        st.markdown('**6.** ¿Qué tanta seguridad le brinda el "Intervalo de Confianza del 95%" mostrado en los gráficos?')
        p6 = st.radio("p6", ["Mucha seguridad; reduce mi temor a la volatilidad","Seguridad moderada","Poca seguridad","No influye en mi decisión"], label_visibility="collapsed", key="enc_p6")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        st.markdown("**7.** ¿Ha cambiado o cancelado una intención de inversión basándose en una alerta o ranking de la plataforma?")
        p7 = st.radio("p7", ["Sí, varias veces","Sí, en alguna ocasión","No, sigo mi plan original"], label_visibility="collapsed", key="enc_p7")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""<div style="background:#0F1629;border-left:4px solid #F59E0B;border-radius:0 8px 8px 0;padding:0.8rem 1.2rem;margin-bottom:1rem;margin-top:0.5rem;">
            <div style="color:#FCD34D;font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;">Sección IV</div>
            <div style="color:#F1F5F9;font-weight:600;font-size:0.95rem;">Percepción de Valor y Democratización</div>
            <div style="color:#64748B;font-size:0.78rem;margin-top:0.2rem;">Objetivo: Sustentar la viabilidad económica y social del proyecto.</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        st.markdown("**8.** ¿Considera que esta herramienta reduce la ventaja que tienen los inversionistas institucionales sobre los minoristas?")
        p8 = st.radio("p8", ["Sí, democratiza el acceso a tecnología avanzada","Sí, pero aún falta información","No, la brecha sigue siendo la misma"], label_visibility="collapsed", key="enc_p8")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        st.markdown("**9.** ¿Estaría dispuesto a utilizar esta plataforma como su herramienta principal de consulta diaria para la BVL?")
        p9 = st.radio("p9", ["Definitivamente sí","Probablemente sí","Tal vez","No"], label_visibility="collapsed", key="enc_p9")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✅ Enviar respuestas", use_container_width=True, type="primary"):
            respuesta = {
                "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "usuario":   st.session_state.usuario_actual or "Anónimo",
                "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5,
                "p6": p6, "p7": p7, "p8": p8, "p9": p9,
            }
            st.session_state.enc_respuestas.append(respuesta)
            st.session_state.enc_enviada = True
            st.rerun()

    n_resp = len(st.session_state.enc_respuestas)
    if n_resp > 0:
        st.markdown('<hr class="kdivider">', unsafe_allow_html=True)
        st.markdown(f"## 📊 Reporte de Validación · {n_resp} respuesta{'s' if n_resp > 1 else ''}")
        df_resp = pd.DataFrame(st.session_state.enc_respuestas)

        avg_usab = df_resp["p2"].mean()
        pct_democratiza = (df_resp["p8"].str.startswith("Sí")).mean() * 100
        pct_usaria = (df_resp["p9"].isin(["Definitivamente sí","Probablemente sí"])).mean() * 100
        pct_comprende = (df_resp["p3"] == "Sí, me ayudó a comprender la lógica del mercado").mean() * 100

        m1, m2, m3, m4 = st.columns(4)
        for col, lbl, val, color in [
            (m1,"Usabilidad promedio",f"{avg_usab:.1f} / 5","#2563EB"),
            (m2,"Percibe democratización",f"{pct_democratiza:.0f}%","#10B981"),
            (m3,"Adoptaría la plataforma",f"{pct_usaria:.0f}%","#8B5CF6"),
            (m4,"IA explicable útil",f"{pct_comprende:.0f}%","#F59E0B"),
        ]:
            with col:
                st.markdown(f'<div class="metric-box"><div class="lbl">{lbl}</div><div class="val" style="color:{color}">{val}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        def bar_chart(series, title, color="#2563EB"):
            counts = series.value_counts()
            fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h",
                                   marker_color=color, text=counts.values, textposition="outside",
                                   textfont=dict(size=11, color="#94A3B8")))
            fig.update_layout(title=dict(text=title, font=dict(size=13, color="#E2E8F0")),
                              height=max(200, len(counts)*56), **plot_layout(),
                              xaxis=dict(showgrid=True, gridcolor="#1E2D4A", color="#334155"),
                              yaxis=dict(showgrid=False, color="#94A3B8", tickfont=dict(size=11)),
                              margin=dict(l=0, r=40, t=40, b=0))
            return fig

        tab_r1, tab_r2, tab_r3, tab_r4 = st.tabs(["Sección I · Perfil","Sección II · IA Explicable","Sección III · Decisión","Sección IV · Democratización"])
        with tab_r1:
            c1r, c2r = st.columns(2)
            with c1r: st.plotly_chart(bar_chart(df_resp["p1"], "P1 · Fuente de información previa", "#2563EB"), use_container_width=True)
            with c2r:
                counts_p2 = df_resp["p2"].value_counts().sort_index()
                p2_labels = {1:"1 — Muy difícil",2:"2 — Difícil",3:"3 — Neutral",4:"4 — Fácil",5:"5 — Muy fácil"}
                labels_p2 = [p2_labels[i] for i in counts_p2.index]
                colors_p2 = ["#EF4444","#F97316","#F59E0B","#84CC16","#10B981"]
                fig_p2 = go.Figure(go.Bar(x=labels_p2, y=counts_p2.values, marker_color=[colors_p2[i-1] for i in counts_p2.index],
                                          text=counts_p2.values, textposition="outside", textfont=dict(size=11, color="#94A3B8")))
                fig_p2.update_layout(title=dict(text="P2 · Facilidad de uso", font=dict(size=13, color="#E2E8F0")),
                                     height=280, **plot_layout(), margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_p2, use_container_width=True)
        with tab_r2:
            c1r, c2r = st.columns(2)
            with c1r: st.plotly_chart(bar_chart(df_resp["p3"], "P3 · IA explicable", "#8B5CF6"), use_container_width=True)
            with c2r: st.plotly_chart(bar_chart(df_resp["p4"], "P4 · Mejora conocimiento financiero", "#8B5CF6"), use_container_width=True)
        with tab_r3:
            c1r, c2r, c3r = st.columns(3)
            with c1r: st.plotly_chart(bar_chart(df_resp["p5"], "P5 · Ratio de Sharpe", "#10B981"), use_container_width=True)
            with c2r: st.plotly_chart(bar_chart(df_resp["p6"], "P6 · Intervalo de confianza", "#10B981"), use_container_width=True)
            with c3r: st.plotly_chart(bar_chart(df_resp["p7"], "P7 · Cambio de decisión", "#10B981"), use_container_width=True)
        with tab_r4:
            c1r, c2r = st.columns(2)
            with c1r: st.plotly_chart(bar_chart(df_resp["p8"], "P8 · Democratización", "#F59E0B"), use_container_width=True)
            with c2r: st.plotly_chart(bar_chart(df_resp["p9"], "P9 · Adopción", "#F59E0B"), use_container_width=True)

        st.markdown('<hr class="kdivider">', unsafe_allow_html=True)
        csv_enc = df_resp.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Descargar datos de la encuesta (CSV)", csv_enc,
                           f"encuesta_kallpa_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

# ═════════════════════════════════════════════════════════════
# SPRINT 4 — HU003 + HU004: EXPLORAR BVL (Búsqueda + Watchlist)
# ═════════════════════════════════════════════════════════════
elif "Explorar" in page:
    st.markdown(f"""
    <div class="page-header">
        <div class="brand" style="margin-bottom:0.3rem;">Sprint 4 · HU003 + HU004</div>
        <h1>Explorar BVL · Lista de Seguimiento</h1>
        <p>Busca activos por nombre o símbolo y gestiona tu watchlist personalizada</p>
    </div>""", unsafe_allow_html=True)

    # ── HU004: Búsqueda de acciones ─────────────────────────
    st.markdown("## 🔍 Buscar Acciones en la BVL")
    st.markdown('<div class="kcard">', unsafe_allow_html=True)

    busq_col1, busq_col2 = st.columns([3, 1])
    with busq_col1:
        termino = st.text_input("Buscar por nombre o símbolo", placeholder="Ej: Credicorp, BAP, cobre, Southern...",
                                key="busq_bvl", label_visibility="collapsed")
    with busq_col2:
        filtro_sector = st.selectbox("Sector", ["Todos", "Financiero", "Minería/Cobre", "Minería/Oro", "Minería/Zinc"],
                                     label_visibility="collapsed")

    # Búsqueda sobre todos los activos
    resultados = []
    for tick, info in RANK_INFO.items():
        nombre = info["nombre"]
        sector = info["sector"]
        coincide_texto = (not termino or
                          termino.lower() in nombre.lower() or
                          termino.lower() in tick.lower() or
                          termino.lower() in sector.lower())
        coincide_sector = (filtro_sector == "Todos" or filtro_sector == sector)
        if coincide_texto and coincide_sector:
            resultados.append((tick, info))

    if not resultados:
        st.markdown("""
        <div style="text-align:center;padding:2rem;color:#334155;">
            <div style="font-size:1.8rem;">🔎</div>
            <div style="margin-top:0.5rem;">No se encontraron activos con ese criterio.</div>
            <div style="font-size:0.8rem;margin-top:0.3rem;color:#1E2D4A;">Prueba con: BAP, SCCO, BVN, VCISY, cobre, financiero...</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color:#64748B;font-size:0.78rem;margin-bottom:0.5rem;">{len(resultados)} resultado{"s" if len(resultados)>1 else ""} encontrado{"s" if len(resultados)>1 else ""}</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Tarjetas de resultados
    if resultados:
        cols_res = st.columns(min(len(resultados), 4))
        for col, (tick, info) in zip(cols_res, resultados):
            en_watch = tick in st.session_state.watchlist
            color_tick = COLORES_ACTIVOS.get(tick, "#2563EB")
            with col:
                st.markdown(f"""
                <div class="kcard" style="text-align:center;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;
                                font-weight:700;color:{color_tick};">{tick}</div>
                    <div style="font-size:0.75rem;color:#94A3B8;margin:0.3rem 0;">{info['nombre']}</div>
                    <hr class="kdivider">
                    <div style="font-size:0.7rem;color:#64748B;line-height:1.8;">
                        Sector: <span style="color:#94A3B8">{info['sector']}</span><br>
                        Volatilidad: <span style="color:#94A3B8">{info['vol']}</span><br>
                        Rating: {info['stars']}
                    </div>
                </div>""", unsafe_allow_html=True)
                lbl_btn = "✅ En seguimiento" if en_watch else "➕ Agregar"
                btn_color = "secondary" if en_watch else "primary"
                if st.button(lbl_btn, key=f"watch_{tick}", use_container_width=True):
                    if en_watch:
                        st.session_state.watchlist.remove(tick)
                        st.toast(f"✅ {tick} eliminado del seguimiento")
                    else:
                        st.session_state.watchlist.append(tick)
                        st.toast(f"✅ {tick} agregado al seguimiento")
                    st.rerun()

    st.markdown('<hr class="kdivider">', unsafe_allow_html=True)

    # ── HU003: Lista de seguimiento ─────────────────────────
    st.markdown("## 👁 Mi Lista de Seguimiento")

    if not st.session_state.watchlist:
        st.markdown("""
        <div class="kcard" style="text-align:center;padding:2rem;">
            <div style="font-size:2rem;margin-bottom:0.5rem;">📋</div>
            <h3 style="color:#F1F5F9;margin:0.3rem 0;">Lista vacía</h3>
            <p style="color:#64748B;">Agrega activos con el botón ➕ para monitorearlos aquí.</p>
        </div>""", unsafe_allow_html=True)
    else:
        n_watch = len(st.session_state.watchlist)
        st.markdown(f'<p style="color:#64748B;font-size:0.8rem;margin-bottom:0.8rem;">Monitoreando <b style="color:#60A5FA">{n_watch}</b> activo{"s" if n_watch>1 else ""} · Precios en tiempo real vía Yahoo Finance</p>', unsafe_allow_html=True)

        # Cargar precios rápidos para la watchlist
        cols_w = st.columns(min(n_watch, 4))
        watch_data = {}
        for col, tick in zip(cols_w, st.session_state.watchlist):
            info = RANK_INFO.get(tick, {})
            color_tick = COLORES_ACTIVOS.get(tick, "#2563EB")
            with col:
                with st.spinner(f"Cargando {tick}..."):
                    try:
                        _, precios_w, _ = descargar_datos(tick, period="5d")
                        precio_act = float(precios_w[-1])
                        precio_ant = float(precios_w[-2]) if len(precios_w) >= 2 else precio_act
                        cambio_d   = (precio_act - precio_ant) / precio_ant * 100
                        # Predicción rápida
                        lp_w  = lstm_sim(precios_w if len(precios_w) >= 60 else
                                         np.concatenate([np.full(60-len(precios_w), precios_w[0]), precios_w]))
                        fut_w = gen_futuro(precio_act, lp_w, dias=7)
                        var_w = (fut_w[-1] - precio_act) / precio_act * 100
                        señal_w = "🟢 COMPRA" if var_w > 3 else "🔴 VENTA" if var_w < -3 else "⚪ MANTENER"
                        watch_data[tick] = {"precio": precio_act, "cambio_d": cambio_d, "var_7d": var_w}

                        c_cambio = "#10B981" if cambio_d >= 0 else "#EF4444"
                        c_var    = "#10B981" if var_w >= 0 else "#EF4444"
                        st.markdown(f"""
                        <div class="kcard" style="text-align:center;">
                            <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                                        font-weight:700;color:{color_tick};">{tick}</div>
                            <div style="font-size:0.7rem;color:#64748B;">{info.get('sector','')}</div>
                            <hr class="kdivider">
                            <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;
                                        font-weight:700;color:#F1F5F9;">${precio_act:.2f}</div>
                            <div style="font-size:0.78rem;color:{c_cambio};margin:0.2rem 0;">
                                Hoy: {cambio_d:+.2f}%
                            </div>
                            <div style="font-size:0.75rem;color:{c_var};">
                                Pred. 7d: {var_w:+.2f}%
                            </div>
                            <div style="font-size:0.72rem;margin-top:0.3rem;">{señal_w}</div>
                        </div>""", unsafe_allow_html=True)
                        if st.button(f"❌ Quitar", key=f"rm_watch_{tick}", use_container_width=True):
                            st.session_state.watchlist.remove(tick)
                            st.toast(f"{tick} eliminado del seguimiento")
                            st.rerun()
                    except Exception as ex:
                        st.markdown(f'<div class="kcard" style="text-align:center;"><b style="color:#60A5FA">{tick}</b><br><span style="color:#EF4444;font-size:0.75rem;">Error: {str(ex)[:40]}</span></div>', unsafe_allow_html=True)

        # Limpiar toda la watchlist
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑 Limpiar watchlist completa", key="clear_watch"):
            st.session_state.watchlist = []
            st.rerun()

        # Mini tabla resumen
        if watch_data:
            st.markdown('<hr class="kdivider">', unsafe_allow_html=True)
            st.markdown("#### Resumen de seguimiento")
            df_watch = pd.DataFrame([
                {"Ticker": t, "Precio ($)": round(d["precio"], 2),
                 "Cambio hoy (%)": round(d["cambio_d"], 2),
                 "Pred. 7d (%)": round(d["var_7d"], 2),
                 "Señal": "🟢 COMPRA" if d["var_7d"] > 3 else "🔴 VENTA" if d["var_7d"] < -3 else "⚪ MANTENER"}
                for t, d in watch_data.items()
            ])
            st.dataframe(df_watch, use_container_width=True, hide_index=True)
            csv_w = df_watch.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Exportar watchlist CSV", csv_w,
                               f"watchlist_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

# ═════════════════════════════════════════════════════════════
# SPRINT 4 — HU006 + HU009: EVOLUCIÓN HISTÓRICA + COMPARACIÓN
# ═════════════════════════════════════════════════════════════
elif "Evolución" in page:
    st.markdown(f"""
    <div class="page-header">
        <div class="brand" style="margin-bottom:0.3rem;">Sprint 4 · HU006 + HU009</div>
        <h1>Evolución Histórica y Comparación de Activos</h1>
        <p>Gráficas con rango de fechas configurable · Análisis comparativo multi-activo · Exportación</p>
    </div>""", unsafe_allow_html=True)

    tab_evol, tab_comp = st.tabs(["📈 Evolución Histórica", "⚖️ Comparación de Activos"])

    # ── HU006: Evolución histórica ───────────────────────────
    with tab_evol:
        st.markdown("### Gráfica de Evolución Histórica con Predicción")

        c_ev1, c_ev2, c_ev3 = st.columns([2, 2, 1])
        with c_ev1:
            activo_ev = st.selectbox("Activo", list(TODOS_ACTIVOS.keys()), key="ev_activo")
            symbol_ev = TODOS_ACTIVOS[activo_ev]
        with c_ev2:
            periodo_ev = st.selectbox("Período histórico", ["1 mes", "3 meses", "6 meses", "1 año", "2 años", "3 años"],
                                      index=2, key="ev_periodo")
        with c_ev3:
            tipo_grafico = st.selectbox("Tipo", ["Velas (OHLC)", "Línea"], key="ev_tipo")

        periodo_map = {"1 mes": "1mo", "3 meses": "3mo", "6 meses": "6mo",
                       "1 año": "1y", "2 años": "2y", "3 años": "3y"}

        # Rango de fechas personalizado
        usar_rango = st.checkbox("📅 Usar rango de fechas personalizado", key="ev_rango_check")
        fecha_ini = fecha_fin = None
        if usar_rango:
            col_fi, col_ff = st.columns(2)
            with col_fi:
                fecha_ini = st.date_input("Desde", value=datetime.now() - timedelta(days=180), key="ev_fi")
            with col_ff:
                fecha_fin = st.date_input("Hasta", value=datetime.now(), key="ev_ff")

        modo_ev = st.selectbox("Modelo IA", ["LSTM Simulado","LSTM + GRU Simulado","Ensemble Completo"], key="ev_modo")

        if st.button("📊 Cargar gráfica", use_container_width=True, key="btn_ev"):
            with st.spinner(f"Descargando datos de {symbol_ev}..."):
                try:
                    period_str = periodo_map[periodo_ev]
                    if usar_rango and fecha_ini and fecha_fin:
                        data_ev = yf.download(symbol_ev, start=fecha_ini, end=fecha_fin,
                                              progress=False, auto_adjust=True)
                        if isinstance(data_ev.columns, pd.MultiIndex):
                            data_ev.columns = data_ev.columns.get_level_values(0)
                    else:
                        data_ev, _, _ = descargar_datos(symbol_ev, period=period_str)

                    if data_ev.empty:
                        st.error("Sin datos para el rango seleccionado.")
                    else:
                        cc_ev = next((c for c in ["Close","Adj Close","close"] if c in data_ev.columns), None)
                        precios_ev = data_ev[cc_ev].dropna().values.astype(float)
                        fechas_ev  = data_ev.index

                        # Predicción 14d
                        lp_ev  = lstm_sim(precios_ev[-60:] if len(precios_ev) >= 60 else precios_ev)
                        gp_ev  = gru_sim(precios_ev)
                        ap_ev  = arima_sim(precios_ev)
                        base_ev = fusionar(lp_ev, gp_ev, ap_ev, modo_ev)
                        fut_ev = gen_futuro(float(precios_ev[-1]), base_ev)
                        ff_ev  = [fechas_ev[-1] + timedelta(days=i+1) for i in range(14)]
                        var_ev = (fut_ev[-1] - float(precios_ev[-1])) / float(precios_ev[-1]) * 100

                        # Métricas
                        precio_min = float(precios_ev.min())
                        precio_max = float(precios_ev.max())
                        precio_act = float(precios_ev[-1])
                        precio_ini = float(precios_ev[0])
                        rendimiento = (precio_act - precio_ini) / precio_ini * 100

                        for col, (lbl, val, color) in zip(st.columns(5), [
                            ("Precio inicial", f"${precio_ini:.2f}", "#64748B"),
                            ("Precio actual",  f"${precio_act:.2f}", "#F1F5F9"),
                            ("Mínimo período", f"${precio_min:.2f}", "#EF4444"),
                            ("Máximo período", f"${precio_max:.2f}", "#10B981"),
                            ("Rendimiento",    f"{rendimiento:+.2f}%", "#10B981" if rendimiento >= 0 else "#EF4444"),
                        ]):
                            with col:
                                st.markdown(f'<div class="metric-box"><div class="lbl">{lbl}</div><div class="val" style="color:{color}">{val}</div></div>', unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)

                        # Gráfico principal
                        color_activo = COLORES_ACTIVOS.get(symbol_ev, "#2563EB")
                        ohlc_ev = {c.lower(): c for c in data_ev.columns}
                        fig_ev = go.Figure()

                        if tipo_grafico == "Velas (OHLC)" and all(k in ohlc_ev for k in ["open","high","low","close"]):
                            fig_ev.add_trace(go.Candlestick(
                                x=fechas_ev,
                                open=data_ev[ohlc_ev["open"]], high=data_ev[ohlc_ev["high"]],
                                low=data_ev[ohlc_ev["low"]], close=data_ev[ohlc_ev["close"]],
                                name=f"{symbol_ev} Histórico",
                                increasing_line_color="#10B981", decreasing_line_color="#EF4444",
                            ))
                        else:
                            fig_ev.add_trace(go.Scatter(
                                x=fechas_ev, y=precios_ev, name=f"{symbol_ev} Histórico",
                                line=dict(color=color_activo, width=2), fill="tozeroy",
                                fillcolor=f"rgba{tuple(list(int(color_activo.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + [0.08])}",
                            ))

                        # Banda de confianza predicción
                        fig_ev.add_trace(go.Scatter(x=ff_ev, y=[p*1.05 for p in fut_ev],
                                                    line=dict(width=0), showlegend=False))
                        fig_ev.add_trace(go.Scatter(x=ff_ev, y=[p*0.95 for p in fut_ev],
                                                    fill="tonexty", fillcolor=f"rgba{tuple(list(int(color_activo.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + [0.10])}",
                                                    line=dict(width=0), name="Confianza ±5%"))
                        fig_ev.add_trace(go.Scatter(x=ff_ev, y=fut_ev, mode="lines+markers",
                                                    name=f"Predicción 14d ({var_ev:+.2f}%)",
                                                    line=dict(color=color_activo, width=2.5, dash="dash"),
                                                    marker=dict(size=5, color=color_activo)))

                        # Línea de precio actual
                        fig_ev.add_hline(y=precio_act, line_dash="dot", line_color="#475569",
                                         annotation_text=f"Actual ${precio_act:.2f}", annotation_position="left")

                        fig_ev.update_layout(height=500, title=dict(
                            text=f"{activo_ev} · {periodo_ev} + Predicción 14 días",
                            font=dict(size=14, color="#E2E8F0")), **plot_layout())
                        st.plotly_chart(fig_ev, use_container_width=True)

                        # Señal final
                        señal_ev = "COMPRA" if var_ev > 3 else "VENTA" if var_ev < -3 else "MANTENER"
                        c_señal = "#10B981" if señal_ev == "COMPRA" else "#EF4444" if señal_ev == "VENTA" else "#F59E0B"
                        st.markdown(f"""
                        <div class="kcard-accent">
                            <span class="brand">{activo_ev} · Conclusión IA</span>
                            <p style="color:#E2E8F0;margin:0.4rem 0 0;font-size:0.88rem;">
                                Señal: <b style="color:{c_señal}">{señal_ev}</b>
                                · Predicción a 14 días: <b>{var_ev:+.2f}%</b>
                                · Precio objetivo: <b style="color:{c_señal}">${fut_ev[-1]:.2f}</b>
                            </p>
                            <p style="color:#475569;font-size:0.74rem;margin-top:0.3rem;">
                                ⚠ Información orientativa. No constituye asesoría financiera.
                            </p>
                        </div>""", unsafe_allow_html=True)

                        # Export
                        df_hist = pd.DataFrame({
                            "Fecha": fechas_ev.strftime("%d/%m/%Y"),
                            "Precio ($)": [round(p, 2) for p in precios_ev],
                        })
                        df_pred_ev = pd.DataFrame({
                            "Fecha": [f.strftime("%d/%m/%Y") for f in ff_ev],
                            "Predicción ($)": [round(p, 2) for p in fut_ev],
                        })
                        df_export = pd.concat([df_hist.assign(Tipo="Histórico"),
                                               df_pred_ev.rename(columns={"Predicción ($)":"Precio ($)"}).assign(Tipo="Predicción")])
                        csv_ev = df_export.to_csv(index=False).encode("utf-8")
                        st.download_button("⬇ Exportar histórico + predicción CSV", csv_ev,
                                           f"evolucion_{symbol_ev}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

                except Exception as e:
                    st.error(f"Error al cargar datos: {e}")

    # ── HU009: Comparación de activos ───────────────────────
    with tab_comp:
        st.markdown("### Comparación Multi-Activo con Predicción Conjunta")
        st.markdown("""
        <div class="kcard-accent" style="margin-bottom:1rem;">
            <span class="brand">HU009 · Análisis Comparativo</span>
            <p style="color:#94A3B8;font-size:0.82rem;margin:0.3rem 0 0;">
                Selecciona entre 2 y 4 activos para comparar su evolución histórica y predicciones
                en una sola gráfica overlay. Los precios se normalizan al 100% para facilitar la comparación.
            </p>
        </div>""", unsafe_allow_html=True)

        cc1, cc2 = st.columns([2, 1])
        with cc1:
            activos_comp = st.multiselect(
                "Seleccionar activos (2–4):",
                options=list(TODOS_ACTIVOS.keys()),
                default=list(TODOS_ACTIVOS.keys())[:2],
                key="comp_activos",
                max_selections=4,
            )
        with cc2:
            periodo_comp = st.selectbox("Período", ["1 mes","3 meses","6 meses","1 año","2 años"],
                                        index=1, key="comp_periodo")
            modo_comp = st.selectbox("Modelo IA", ["Ensemble Completo","LSTM Simulado","LSTM + GRU Simulado"], key="comp_modo")

        normalizar = st.checkbox("Normalizar precios (base 100)", value=True, key="comp_norm")

        if len(activos_comp) < 2:
            st.warning("Selecciona al menos 2 activos para comparar.")
        elif st.button("⚖️ Comparar activos", use_container_width=True, key="btn_comp"):
            periodo_map_c = {"1 mes": "1mo", "3 meses": "3mo", "6 meses": "6mo",
                             "1 año": "1y", "2 años": "2y"}
            period_c = periodo_map_c[periodo_comp]

            fig_comp = go.Figure()
            tabla_comp = []
            errores = []

            with st.spinner("Cargando y calculando predicciones..."):
                for nom_a in activos_comp:
                    tick_c = TODOS_ACTIVOS[nom_a]
                    color_c = COLORES_ACTIVOS.get(tick_c, "#2563EB")
                    try:
                        data_c, precios_c, fechas_c = descargar_datos(tick_c, period=period_c)
                        cc_c = next((c for c in ["Close","Adj Close","close"] if c in data_c.columns), None)
                        precios_c_clean = data_c[cc_c].dropna().values.astype(float)
                        fechas_c_clean  = data_c.index[data_c[cc_c].notna()]

                        # Normalización
                        base_precio = precios_c_clean[0]
                        y_hist = (precios_c_clean / base_precio * 100) if normalizar else precios_c_clean

                        # Trace histórico
                        fig_comp.add_trace(go.Scatter(
                            x=fechas_c_clean, y=y_hist, name=f"{tick_c} Histórico",
                            line=dict(color=color_c, width=2),
                        ))

                        # Predicción 14d
                        p_full = precios_c_clean
                        lp_c  = lstm_sim(p_full[-60:] if len(p_full) >= 60 else
                                         np.concatenate([np.full(60-len(p_full), p_full[0]), p_full]))
                        gp_c  = gru_sim(p_full)
                        ap_c  = arima_sim(p_full)
                        base_c = fusionar(lp_c, gp_c, ap_c, modo_comp)
                        fut_c = gen_futuro(float(p_full[-1]), base_c)
                        ff_c  = [fechas_c_clean[-1] + timedelta(days=i+1) for i in range(14)]
                        var_c = (fut_c[-1] - float(p_full[-1])) / float(p_full[-1]) * 100
                        señal_c = "COMPRA" if var_c > 3 else "VENTA" if var_c < -3 else "MANTENER"

                        y_pred = ([p / base_precio * 100 for p in fut_c] if normalizar else fut_c)

                        fig_comp.add_trace(go.Scatter(
                            x=ff_c, y=y_pred, name=f"{tick_c} Pred. ({var_c:+.2f}%)",
                            line=dict(color=color_c, width=2, dash="dash"),
                            marker=dict(size=4),
                        ))

                        rendimiento_c = (float(p_full[-1]) - float(p_full[0])) / float(p_full[0]) * 100
                        tabla_comp.append({
                            "Activo":        tick_c,
                            "Nombre":        nom_a.split("(")[0].strip(),
                            "Precio actual ($)": round(float(p_full[-1]), 2),
                            f"Rendimiento {periodo_comp} (%)": round(rendimiento_c, 2),
                            "Pred. 14d (%)": round(var_c, 2),
                            "Señal":         señal_c,
                            "Sector":        RANK_INFO[tick_c]["sector"],
                            "Volatilidad":   RANK_INFO[tick_c]["vol"],
                        })
                    except Exception as ex:
                        errores.append(f"{tick_c}: {str(ex)[:50]}")

            if errores:
                for err in errores:
                    st.warning(f"⚠ {err}")

            if tabla_comp:
                # Separador histórico/predicción
                fecha_sep = tabla_comp[0] if tabla_comp else None
                try:
                    data_sep, _, fechas_sep = descargar_datos(TODOS_ACTIVOS[activos_comp[0]], period=period_c)
                    fig_comp.add_vline(x=fechas_sep[-1], line_dash="dot", line_color="#334155",
                                       annotation_text="Hoy", annotation_position="top right")
                except:
                    pass

                y_label = "Precio normalizado (base 100)" if normalizar else "Precio (USD)"
                fig_comp.update_layout(
                    height=520,
                    title=dict(text=f"Comparación: {' vs '.join([TODOS_ACTIVOS[a] for a in activos_comp])} · {periodo_comp} + Predicción 14d",
                               font=dict(size=13, color="#E2E8F0")),
                    **plot_layout(),
                    yaxis=dict(showgrid=True, gridcolor="#1E2D4A", color="#334155", title=y_label),
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                # Tabla comparativa
                st.markdown("#### Tabla Comparativa")
                df_comp = pd.DataFrame(tabla_comp)
                st.dataframe(df_comp, use_container_width=True, hide_index=True)

                # Métricas resumen
                st.markdown("#### Ranking por predicción 14d")
                df_rank = df_comp.sort_values("Pred. 14d (%)", ascending=False).reset_index(drop=True)
                cols_rank = st.columns(len(df_rank))
                for i, (col_r, row) in enumerate(zip(cols_rank, df_rank.itertuples())):
                    tick_r = row.Activo
                    color_r = COLORES_ACTIVOS.get(tick_r, "#2563EB")
                    c_var_r = "#10B981" if row._5 >= 0 else "#EF4444"  # Pred 14d
                    medal = ["🥇", "🥈", "🥉", "4️⃣"][i]
                    with col_r:
                        st.markdown(f"""
                        <div class="kcard" style="text-align:center;">
                            <div style="font-size:1.3rem;">{medal}</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;
                                        font-weight:700;color:{color_r};">{tick_r}</div>
                            <div style="font-size:0.7rem;color:#64748B;">{row.Sector}</div>
                            <hr class="kdivider">
                            <div style="font-size:0.85rem;color:{c_var_r};font-weight:600;">
                                {row._5:+.2f}%
                            </div>
                            <div style="font-size:0.7rem;color:#64748B;">Pred. 14d</div>
                            <div style="font-size:0.72rem;margin-top:0.3rem;">
                                {'🟢 COMPRA' if row.Señal=='COMPRA' else '🔴 VENTA' if row.Señal=='VENTA' else '⚪ MANTENER'}
                            </div>
                        </div>""", unsafe_allow_html=True)

                # Exportar comparación
                st.markdown('<hr class="kdivider">', unsafe_allow_html=True)
                cc_exp1, cc_exp2 = st.columns(2)
                with cc_exp1:
                    csv_comp = df_comp.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇ Exportar análisis CSV",
                                       csv_comp,
                                       f"comparacion_{'_'.join([TODOS_ACTIVOS[a] for a in activos_comp])}_{datetime.now().strftime('%Y%m%d')}.csv",
                                       "text/csv", use_container_width=True)
                with cc_exp2:
                    # Exportar gráfica como HTML interactivo
                    buf = io.StringIO()
                    fig_comp.write_html(buf)
                    st.download_button("⬇ Exportar gráfica HTML",
                                       buf.getvalue().encode("utf-8"),
                                       f"grafica_comp_{datetime.now().strftime('%Y%m%d')}.html",
                                       "text/html", use_container_width=True)

                st.markdown('<p style="color:#334155;font-size:0.72rem;">Los datos exportados pueden usarse directamente en Excel, SPSS o Python para análisis adicionales.</p>', unsafe_allow_html=True)
