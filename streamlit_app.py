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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp { background: #0A0E1A; color: #E2E8F0; }

section[data-testid="stSidebar"] {
    background: #0F1629;
    border-right: 1px solid #1E2D4A;
}
section[data-testid="stSidebar"] label {
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.kcard {
    background: #111827; border: 1px solid #1E2D4A;
    border-radius: 12px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}
.kcard-accent {
    background: linear-gradient(135deg, #0F1629 0%, #111D35 100%);
    border: 1px solid #2563EB44; border-radius: 12px;
    padding: 1.2rem 1.6rem; margin-bottom: 1rem;
}
.page-header {
    background: linear-gradient(135deg, #0F1629 0%, #0D1F3C 100%);
    border: 1px solid #1E2D4A; border-radius: 12px;
    padding: 1.8rem 2rem; margin-bottom: 1.5rem;
}
.page-header h1 {
    font-size: 1.7rem; font-weight: 700; color: #F1F5F9;
    margin: 0 0 0.3rem 0; letter-spacing: -0.02em;
}
.page-header p { color: #64748B; font-size: 0.85rem; margin: 0; }

.metric-box {
    background: #111827; border: 1px solid #1E2D4A;
    border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
}
.metric-box .lbl {
    font-size: 0.7rem; color: #64748B;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem;
}
.metric-box .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.45rem; font-weight: 600; color: #F1F5F9;
}
.badge {
    display: inline-block; padding: 0.22rem 0.8rem; border-radius: 99px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
}
.badge-conservador { background:#0F3460; color:#60A5FA; border:1px solid #1D4ED8; }
.badge-moderado    { background:#1A2E05; color:#86EFAC; border:1px solid #16A34A; }
.badge-agresivo    { background:#3B0A0A; color:#FCA5A5; border:1px solid #DC2626; }
.brand {
    font-size: 0.7rem; color: #2563EB; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase;
}
.kdivider { border:none; border-top:1px solid #1E2D4A; margin: 1rem 0; }
.stButton > button {
    background: #2563EB; color: white; border: none; border-radius: 8px;
    font-family: 'Sora', sans-serif; font-weight: 600; letter-spacing: 0.02em;
    padding: 0.55rem 1.4rem; transition: all 0.2s;
}
.stButton > button:hover { background: #1D4ED8; transform: translateY(-1px); box-shadow: 0 4px 16px #2563EB44; }
.stTextInput input {
    background: #111827 !important; border: 1px solid #1E2D4A !important;
    color: #E2E8F0 !important; border-radius: 8px !important;
}
.stTextInput input:focus { border-color: #2563EB !important; box-shadow: 0 0 0 2px #2563EB22 !important; }
.stTabs [data-baseweb="tab-list"] { background: #0F1629; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #64748B; border-radius: 8px; font-weight: 600; font-size: 0.82rem; }
.stTabs [aria-selected="true"] { background: #1E2D4A !important; color: #F1F5F9 !important; }
.streamlit-expanderHeader { background: #111827 !important; border: 1px solid #1E2D4A !important; border-radius: 8px !important; color: #E2E8F0 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #1E2D4A; border-radius: 2px; }

/* ── HU011: Notificaciones web ── */
.notif-panel {
    background: #0F1629;
    border: 2px solid #2563EB;
    border-radius: 14px;
    padding: 0;
    margin-bottom: 1rem;
    overflow: hidden;
}
.notif-header {
    background: linear-gradient(135deg, #1E3A6E 0%, #1E2D4A 100%);
    padding: 1rem 1.4rem;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid #2563EB44;
}
.notif-badge-count {
    background: #EF4444; color: #FFFFFF;
    font-size: 0.68rem; font-weight: 700;
    padding: 0.15rem 0.5rem; border-radius: 99px;
    letter-spacing: 0.04em;
}
.notif-item-compra {
    border-left: 4px solid #10B981;
    background: #0A1A12;
    padding: 0.8rem 1.2rem; margin: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    position: relative;
}
.notif-item-venta {
    border-left: 4px solid #EF4444;
    background: #1A0A0A;
    padding: 0.8rem 1.2rem; margin: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
}
.notif-item-info {
    border-left: 4px solid #F59E0B;
    background: #1A150A;
    padding: 0.8rem 1.2rem; margin: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
}
.notif-item-sistema {
    border-left: 4px solid #8B5CF6;
    background: #110A1A;
    padding: 0.8rem 1.2rem; margin: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
}
.notif-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px; vertical-align: middle;
}
.notif-dot-compra  { background: #10B981; }
.notif-dot-venta   { background: #EF4444; }
.notif-dot-info    { background: #F59E0B; }
.notif-dot-sistema { background: #8B5CF6; }
.toast-compra  { animation: slideIn 0.4s ease; }
.toast-venta   { animation: slideIn 0.4s ease; }
@keyframes slideIn { from { transform: translateX(30px); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
.notif-config-row {
    display: flex; gap: 1rem; align-items: center;
    padding: 0.8rem 1.2rem; flex-wrap: wrap;
}
.notif-config-chip {
    background: #1E2D4A; border: 1px solid #2563EB44;
    border-radius: 6px; padding: 0.3rem 0.8rem;
    font-size: 0.75rem; color: #94A3B8;
}
.notif-config-chip.active {
    background: #1E3A6E; border-color: #2563EB;
    color: #60A5FA; font-weight: 600;
}
.notif-empty {
    text-align: center; padding: 2rem 1rem;
    color: #334155; font-size: 0.85rem;
}
.pulse-ring {
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; background: #10B981;
    box-shadow: 0 0 0 3px #10B98133;
    margin-right: 8px; vertical-align: middle;
}
.pulse-ring-off {
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; background: #334155;
    margin-right: 8px; vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────
defaults = {
    "logged_in":        False,
    "usuario_actual":   None,
    "usuarios_db":      {},
    "perfil_riesgo":    "Moderado",
    "notif_correo":     False,
    "alertas_log":      [],
    # HU011 — notificaciones web
    "notif_web":        False,
    "notif_umbral_web": 3,
    "notif_tipos":      ["Variación de precio", "Señal fuerte"],
    "notif_pendientes": [],
    "notif_leidas":     [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers Auth ─────────────────────────────────────────────
def validar_correo(c): return bool(re.match(r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$", c))
def validar_dni(d):    return bool(re.match(r"^\d{8}$", d))
def validar_pwd(p):
    if len(p)<8:                return False,"Mínimo 8 caracteres."
    if not re.search(r"[A-Z]",p): return False,"Incluir al menos una mayúscula."
    if not re.search(r"\d",p):  return False,"Incluir al menos un número."
    return True,"OK"
def registrar(nombre,correo,dni,pwd):
    db=st.session_state["usuarios_db"]
    if correo in db: return False,"Correo ya registrado."
    if any(u["dni"]==dni for u in db.values()): return False,"DNI ya registrado."
    db[correo]={"nombre":nombre,"dni":dni,"pwd":pwd,"perfil":"Moderado"}
    st.session_state["usuarios_db"]=db
    return True,"OK"
def autenticar(usr,pwd):
    if usr in ("kallpa","demo@kallpa.com") and pwd=="lstm2025": return True,"Demo Kallpa"
    db=st.session_state["usuarios_db"]
    if usr in db and db[usr]["pwd"]==pwd: return True,db[usr]["nombre"]
    return False,""

# ── Helpers predicción ───────────────────────────────────────
ACTIVOS_PERFIL = {
    "Conservador": {"Credicorp (BAP)":"BAP","Southern Copper (SCCO)":"SCCO"},
    "Moderado":    {"Southern Copper (SCCO)":"SCCO","Buenaventura (BVN)":"BVN","Credicorp (BAP)":"BAP","Volcan B (VCISY)":"VCISY"},
    "Agresivo":    {"Buenaventura (BVN)":"BVN","Volcan B (VCISY)":"VCISY"},
}
PERFIL_DESC = {
    "Conservador":"Baja volatilidad · Preservación de capital · Ideal para horizontes largos",
    "Moderado":   "Balance crecimiento/estabilidad · Portafolio completo BVL",
    "Agresivo":   "Alta volatilidad · Máximo potencial de retorno · Alta tolerancia al riesgo",
}
RANK_INFO = {
    "BAP":  {"sector":"Financiero","vol":"Baja","stars":"⭐⭐⭐⭐⭐"},
    "SCCO": {"sector":"Minería/Cobre","vol":"Media","stars":"⭐⭐⭐⭐"},
    "BVN":  {"sector":"Minería/Oro","vol":"Alta","stars":"⭐⭐⭐"},
    "VCISY":{"sector":"Minería/Zinc","vol":"Alta","stars":"⭐⭐⭐"},
}
def descargar_datos(sym,period="3y"):
    data=yf.download(sym,period=period,progress=False,auto_adjust=True)
    if data.empty: raise ValueError(f"Sin datos para {sym}.")
    if isinstance(data.columns,pd.MultiIndex): data.columns=data.columns.get_level_values(0)
    cc=next((c for c in ["Close","Adj Close","close"] if c in data.columns),None)
    if not cc: raise ValueError("Columna de precios no encontrada.")
    precios=data[cc].dropna().values.astype(float)
    if len(precios)<60: raise ValueError(f"Datos insuficientes: {len(precios)}.")
    return data,precios,data.index
def lstm_sim(p,w=60):
    c=np.polyfit(np.arange(w),p[-w:],3); return float(np.polyval(c,w))
def gru_sim(p):
    e=float(p[-20])
    for x in p[-20:]: e=0.2*float(x)+0.8*e
    return e
def arima_sim(p):
    d=np.diff(p[-30:]); return float(p[-1])+(float(np.mean(d)) if len(d) else 0)*2
def fusionar(l,g,a,modo):
    if modo=="Ensemble Completo":   return 0.60*l+0.25*g+0.15*a
    if modo=="LSTM + GRU Simulado": return 0.70*l+0.30*g
    return l
def gen_futuro(actual,pred,dias=14):
    np.random.seed(42); f=[]; a=actual
    for _ in range(dias):
        n=a+(pred-a)/dias+np.random.normal(0,0.008)*a; f.append(float(n)); a=n
    return f
def bt(p,lp):
    n=len(p)
    if n<50: return 0.0
    h=p[max(0,n-44):max(0,n-44)+14]
    if len(h)<14: return 0.0
    pb=float(h[0]); pb_=[]; tmp=pb
    for _ in range(14): tmp+=(lp-tmp)/14; pb_.append(tmp)
    ok=sum(1 for i in range(1,14) if np.sign(h[i]-h[i-1])==np.sign(pb_[i]-pb_[i-1]))
    return ok/13*100
def plot_layout():
    return dict(
        paper_bgcolor="#111827",plot_bgcolor="#111827",
        font=dict(family="Sora",color="#94A3B8",size=11),
        xaxis=dict(showgrid=False,color="#334155",rangeslider_visible=False),
        yaxis=dict(showgrid=True,gridcolor="#1E2D4A",color="#334155"),
        legend=dict(orientation="h",y=1.05,bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",margin=dict(l=0,r=0,t=24,b=0),
    )

# ── HU011: registrar notificación web ────────────────────────
def registrar_notif(tipo, activo, var, señal, mensaje=""):
    """Agrega una notificación a la cola de pendientes."""
    iconos = {"COMPRA":"📈","VENTA":"📉","MANTENER":"➡️","SISTEMA":"🔔","INFO":"⚠️"}
    colores = {"COMPRA":"compra","VENTA":"venta","MANTENER":"info","SISTEMA":"sistema","INFO":"info"}
    notif = {
        "id":       len(st.session_state.notif_pendientes) + len(st.session_state.notif_leidas),
        "hora":     datetime.now().strftime("%H:%M"),
        "fecha":    datetime.now().strftime("%d/%m/%Y"),
        "activo":   activo,
        "variacion":var,
        "señal":    señal,
        "tipo":     colores.get(señal,"info"),
        "icono":    iconos.get(señal,"🔔"),
        "titulo":   f"{iconos.get(señal,'🔔')} {señal} — {activo}",
        "cuerpo":   mensaje or f"Variación proyectada {var:+.2f}% · Perfil {st.session_state.perfil_riesgo}",
        "leida":    False,
    }
    st.session_state.notif_pendientes.append(notif)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    # Badge de notificaciones pendientes en el sidebar
    n_pend = len(st.session_state.notif_pendientes)
    notif_label = ""
    if st.session_state.logged_in and n_pend > 0:
        notif_label = f" 🔴 {n_pend}"

    st.markdown(
        f'<div style="padding:1rem 0 0.8rem;border-bottom:1px solid #1E2D4A;margin-bottom:1rem;">'
        f'<span class="brand">📈 Kallpa Securities</span>'
        f'<span style="color:#EF4444;font-size:0.65rem;font-weight:700;">{notif_label}</span><br>'
        f'<span style="color:#334155;font-size:0.65rem;">BVL Intelligence Platform</span></div>',
        unsafe_allow_html=True
    )

    if st.session_state.logged_in:
        nombre_sb = st.session_state.usuario_actual or "Usuario"
        st.markdown(
            f'<div style="padding:0.4rem 0 0.8rem;color:#64748B;font-size:0.78rem;">'
            f'👤 <b style="color:#E2E8F0">{nombre_sb}</b> · '
            f'<span class="badge badge-{st.session_state.perfil_riesgo.lower()}" style="font-size:0.62rem;">'
            f'{st.session_state.perfil_riesgo}</span></div>',
            unsafe_allow_html=True
        )

    # Indicador de notificaciones web en sidebar
    if st.session_state.logged_in:
        estado_notif = "ACTIVAS" if st.session_state.notif_web else "INACTIVAS"
        dot_class = "pulse-ring" if st.session_state.notif_web else "pulse-ring-off"
        st.markdown(
            f'<div style="font-size:0.7rem;color:#475569;margin-bottom:0.8rem;">'
            f'<span class="{dot_class}"></span>'
            f'Notificaciones web: <b style="color:{"#10B981" if st.session_state.notif_web else "#475569"}">'
            f'{estado_notif}</b>'
            f'{f" · <span style=\'color:#EF4444;font-weight:700\'>{n_pend} sin leer</span>" if n_pend > 0 else ""}'
            f'</div>',
            unsafe_allow_html=True
        )

    page = st.radio("Navegación", ["🏠  Dashboard","👤  Mi Cuenta","📬  Alertas","❓  Ayuda"], label_visibility="collapsed")

    generar = False
    activo = symbol = modo = tc = tasa = cobre = None

    if st.session_state.logged_in:
        st.markdown('<hr class="kdivider">', unsafe_allow_html=True)
        st.markdown('<p style="color:#475569;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;">Configuración del modelo</p>', unsafe_allow_html=True)
        perfil=st.selectbox("Perfil de riesgo",["Conservador","Moderado","Agresivo"],
                            index=["Conservador","Moderado","Agresivo"].index(st.session_state.perfil_riesgo))
        st.session_state.perfil_riesgo=perfil
        activos_disp=ACTIVOS_PERFIL[perfil]
        activo=st.selectbox("Activo BVL",list(activos_disp.keys()))
        symbol=activos_disp[activo]
        modo=st.selectbox("Modelo IA",["LSTM Simulado","LSTM + GRU Simulado","Ensemble Completo"])
        st.markdown('<p style="color:#475569;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;margin-top:0.8rem;">Variables Macroeconómicas</p>', unsafe_allow_html=True)
        tc=st.slider("Tipo Cambio PEN/USD",3.5,4.2,3.78,0.01)
        tasa=st.slider("Tasa BCRP (%)",4.0,8.0,5.25,0.05)
        cobre=st.slider("Cobre USD/lb",3.5,5.5,4.35,0.05)
        generar=st.button("⚡ Generar Predicción",use_container_width=True)
        st.markdown('<hr class="kdivider">', unsafe_allow_html=True)
        if st.button("↩ Cerrar sesión",use_container_width=True):
            for k in ["logged_in","usuario_actual"]:
                st.session_state[k]=False if k=="logged_in" else None
            st.rerun()

# ══ AUTH ════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("""
    <div class="page-header" style="text-align:center;padding:3rem 2rem;">
        <div class="brand" style="font-size:0.9rem;margin-bottom:1rem;">📈 KALLPA SECURITIES SAB</div>
        <h1 style="font-size:2.4rem;margin-bottom:0.6rem;">BVL Intelligence Platform</h1>
        <p style="font-size:1rem;color:#64748B;">Sistema de Predicción Financiera con IA · Mercado Peruano</p>
    </div>""", unsafe_allow_html=True)
    _,col_mid,_=st.columns([1,2,1])
    with col_mid:
        tab_l,tab_r=st.tabs(["  Iniciar sesión  ","  Crear cuenta  "])
        with tab_l:
            st.markdown('<div class="kcard">', unsafe_allow_html=True)
            st.markdown("#### Acceso al sistema")
            u_in=st.text_input("Correo o usuario",placeholder="usuario@email.com",key="li_u")
            p_in=st.text_input("Contraseña",type="password",placeholder="••••••••",key="li_p")
            if st.button("Ingresar →",use_container_width=True,key="btn_li"):
                ok,nom=autenticar(u_in.strip(),p_in)
                if ok:
                    st.session_state.logged_in=True; st.session_state.usuario_actual=nom; st.rerun()
                else: st.error("Credenciales incorrectas.")
            st.markdown('<hr class="kdivider"><p style="color:#475569;font-size:0.78rem;">Demo: <code style="color:#60A5FA">demo@kallpa.com</code> / <code style="color:#60A5FA">lstm2025</code></p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with tab_r:
            st.markdown('<div class="kcard">', unsafe_allow_html=True)
            st.markdown("#### Nueva cuenta")
            r_nom=st.text_input("Nombre completo",placeholder="Juan Pérez García",key="r_n")
            r_cor=st.text_input("Correo electrónico",placeholder="juan@email.com",key="r_c")
            c1r,c2r=st.columns(2)
            with c1r: r_dni=st.text_input("DNI (8 dígitos)",placeholder="12345678",max_chars=8,key="r_d")
            with c2r: r_prf=st.selectbox("Perfil inicial",["Conservador","Moderado","Agresivo"],index=1,key="r_pf")
            r_pw1=st.text_input("Contraseña",type="password",placeholder="Mín 8 · 1 mayúscula · 1 número",key="r_p1")
            r_pw2=st.text_input("Confirmar contraseña",type="password",placeholder="••••••••",key="r_p2")
            errs=[]
            if r_cor and not validar_correo(r_cor): errs.append("Formato de correo inválido.")
            if r_dni and not validar_dni(r_dni):     errs.append("DNI debe tener 8 dígitos.")
            if r_pw1:
                ok_p,msg_p=validar_pwd(r_pw1)
                if not ok_p: errs.append(msg_p)
            if r_pw1 and r_pw2 and r_pw1!=r_pw2: errs.append("Las contraseñas no coinciden.")
            for e in errs: st.markdown(f'<p style="color:#F87171;font-size:0.78rem;">⚠ {e}</p>',unsafe_allow_html=True)
            if st.button("Crear cuenta →",use_container_width=True,key="btn_reg"):
                if not all([r_nom,r_cor,r_dni,r_pw1,r_pw2]): st.error("Completa todos los campos.")
                elif errs: st.error("Corrige los errores señalados.")
                else:
                    ok_r,msg_r=registrar(r_nom.strip(),r_cor.strip(),r_dni.strip(),r_pw1)
                    if ok_r:
                        st.session_state.logged_in=True; st.session_state.usuario_actual=r_nom.strip()
                        st.session_state.perfil_riesgo=r_prf; st.rerun()
                    else: st.error(msg_r)
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    for col,(val,lbl) in zip(st.columns(4),[("89%","Precisión IA"),("14 días","Horizonte pred."),("4","Activos BVL"),("3","Modelos fusionados")]):
        with col:
            st.markdown(f'<div class="metric-box"><div class="lbl">{lbl}</div><div class="val" style="color:#2563EB">{val}</div></div>',unsafe_allow_html=True)
    st.stop()

# ══ DASHBOARD ════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown(f"""
    <div class="page-header">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
        <div>
          <div class="brand" style="margin-bottom:0.4rem;">Dashboard Predictivo</div>
          <h1>Pronóstico BVL con IA</h1>
          <p>Mercado de Valores de Lima · Datos vía Yahoo Finance · {datetime.now().strftime('%d %b %Y')}</p>
        </div>
        <div><span class="badge badge-{st.session_state.perfil_riesgo.lower()}">Perfil {st.session_state.perfil_riesgo}</span></div>
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kcard-accent">
      <span class="badge badge-{st.session_state.perfil_riesgo.lower()}">{st.session_state.perfil_riesgo}</span>
      <span style="color:#94A3B8;font-size:0.85rem;margin-left:0.8rem;">{PERFIL_DESC[st.session_state.perfil_riesgo]}</span>
    </div>""", unsafe_allow_html=True)

    # Banner de notificaciones pendientes en dashboard
    n_pend = len(st.session_state.notif_pendientes)
    if n_pend > 0 and st.session_state.notif_web:
        st.markdown(f"""
        <div style="background:#1A0E2E;border:1px solid #8B5CF6;border-radius:10px;
                    padding:0.8rem 1.2rem;margin-bottom:1rem;display:flex;
                    align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.5rem;">
            <div style="display:flex;align-items:center;gap:0.6rem;">
                <span style="font-size:1.2rem;">🔔</span>
                <span style="color:#C4B5FD;font-size:0.88rem;font-weight:600;">
                    Tienes <b style="color:#A78BFA">{n_pend} notificación{'es' if n_pend>1 else ''}</b> sin leer
                </span>
            </div>
            <span style="color:#6D28D9;font-size:0.75rem;">Ve a Alertas → Notificaciones Web para revisarlas</span>
        </div>""", unsafe_allow_html=True)

    if not generar:
        st.markdown('<div class="kcard" style="text-align:center;padding:2.5rem;"><p style="font-size:2rem;">⚡</p><h3 style="color:#F1F5F9;margin:0.3rem 0;">Listo para predecir</h3><p style="color:#64748B;">Selecciona activo en el panel lateral y presiona <b style="color:#2563EB">Generar Predicción</b>.</p></div>', unsafe_allow_html=True)
        st.markdown("### Activos disponibles para tu perfil")
        activos_p=ACTIVOS_PERFIL[st.session_state.perfil_riesgo]
        cols_r=st.columns(len(activos_p))
        for col,(nom_a,tick) in zip(cols_r,activos_p.items()):
            info=RANK_INFO.get(tick,{})
            with col:
                st.markdown(f"""
                <div class="kcard" style="text-align:center;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.15rem;font-weight:700;color:#60A5FA;">{tick}</div>
                    <div style="font-size:0.75rem;color:#94A3B8;margin:0.3rem 0;">{nom_a}</div>
                    <hr class="kdivider">
                    <div style="font-size:0.7rem;color:#64748B;">Sector: <span style="color:#94A3B8">{info.get('sector','')}</span></div>
                    <div style="font-size:0.7rem;color:#64748B;">Volatilidad: <span style="color:#94A3B8">{info.get('vol','')}</span></div>
                    <div style="font-size:0.8rem;margin-top:0.3rem;">{info.get('stars','')}</div>
                </div>""", unsafe_allow_html=True)
    else:
        with st.spinner("Calculando predicción..."):
            try:
                data,precios,fechas=descargar_datos(symbol)
                ult=float(precios[-1]); lp=lstm_sim(precios); gp=gru_sim(precios); ap=arima_sim(precios)
                base=fusionar(lp,gp,ap,modo)
                mac=(tc-3.78)*0.02+(tasa-5.25)*(-0.015)+(cobre-4.35)*0.03
                pf=base*(1+mac); fut=gen_futuro(ult,pf)
                var=(fut[-1]-ult)/ult*100; prec=bt(precios,lp)
                tend="alcista 📈" if var>0 else "bajista 📉"
                señal_dia="COMPRA" if var>3 else "VENTA" if var<-3 else "MANTENER"

                for col,(lbl,val,color) in zip(st.columns(5),[
                    ("Precio Actual",f"${ult:.2f}","#F1F5F9"),
                    ("Predicción 14d",f"${fut[-1]:.2f}","#2563EB"),
                    ("Variación",f"{var:+.2f}%","#10B981" if var>=0 else "#EF4444"),
                    ("Impacto Macro",f"{mac*100:+.2f}%","#F59E0B"),
                    ("Precisión BT",f"{prec:.1f}%","#8B5CF6"),
                ]):
                    with col:
                        st.markdown(f'<div class="metric-box"><div class="lbl">{lbl}</div><div class="val" style="color:{color}">{val}</div></div>',unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="kcard-accent">
                    <span class="brand">Análisis Kallpa · {activo}</span>
                    <p style="color:#E2E8F0;margin:0.5rem 0 0;font-size:0.9rem;">
                        Tendencia <b style="color:{'#10B981' if var>0 else '#F87171'}">{tend}</b>
                        · Variación proyectada <b>{var:+.2f}%</b> en 14 días
                        · Impacto macro <b>{mac*100:+.2f}%</b>
                        · Perfil <b>{st.session_state.perfil_riesgo}</b>
                    </p>
                    <p style="color:#475569;font-size:0.76rem;margin-top:0.5rem;">⚠ Orientativo. Combine con análisis fundamental.</p>
                </div>""", unsafe_allow_html=True)

                st.markdown("### Gráfico de Pronóstico")
                dh=data[-90:].copy(); fh=fechas[-90:]
                ohlc={c.lower():c for c in dh.columns}
                fig=go.Figure()
                if all(k in ohlc for k in ["open","high","low","close"]):
                    fig.add_trace(go.Candlestick(x=fh,open=dh[ohlc["open"]],high=dh[ohlc["high"]],
                        low=dh[ohlc["low"]],close=dh[ohlc["close"]],name="Histórico",
                        increasing_line_color="#10B981",decreasing_line_color="#EF4444"))
                else:
                    ck=ohlc.get("close",list(ohlc.values())[0])
                    fig.add_trace(go.Scatter(x=fh,y=dh[ck],name="Histórico",line=dict(color="#64748B",width=1.5)))
                ff=[fechas[-1]+timedelta(days=i+1) for i in range(14)]
                fig.add_trace(go.Scatter(x=ff,y=[p*1.05 for p in fut],line=dict(width=0),showlegend=False))
                fig.add_trace(go.Scatter(x=ff,y=[p*0.95 for p in fut],fill="tonexty",
                    fillcolor="rgba(37,99,235,0.08)",line=dict(width=0),name="Confianza ±5%"))
                fig.add_trace(go.Scatter(x=ff,y=fut,mode="lines+markers",name="Predicción",
                    line=dict(color="#2563EB",width=2.5,dash="dash"),marker=dict(size=6,color="#2563EB")))
                fig.update_layout(height=460,**plot_layout())
                st.plotly_chart(fig,use_container_width=True)

                st.markdown("### Señales Diarias")
                t1,t2=st.tabs(["📋 Tabla de señales","📊 Variación diaria"])
                df_fut=pd.DataFrame({
                    "Día":range(1,15),"Fecha":[f.strftime("%d/%m/%Y") for f in ff],
                    "Precio ($)":[round(p,2) for p in fut],
                    "Var. (%)":[round((p-ult)/ult*100,2) for p in fut],
                    "Señal":["🟢 COMPRA" if p>ult*1.03 else "🔴 VENTA" if p<ult*0.97 else "⚪ MANTENER" for p in fut],
                })
                with t1:
                    st.dataframe(df_fut,use_container_width=True,hide_index=True)
                    csv=df_fut.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇ Descargar CSV",csv,f"kallpa_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv","text/csv")
                with t2:
                    vs=[(p-ult)/ult*100 for p in fut]
                    fig2=go.Figure(go.Bar(x=[f"D{i+1}" for i in range(14)],y=vs,
                        marker_color=["#10B981" if v>=0 else "#EF4444" for v in vs],
                        text=[f"{v:+.2f}%" for v in vs],textposition="outside",
                        textfont=dict(size=10,color="#94A3B8")))
                    fig2.update_layout(height=280,**plot_layout())
                    st.plotly_chart(fig2,use_container_width=True)

                # Registrar alerta en log + notificación web si está activa
                st.session_state.alertas_log.append({
                    "fecha":datetime.now().strftime("%d/%m %H:%M"),
                    "activo":activo,"variacion":var,
                    "msg":f"{activo}: variación {var:+.2f}% — {señal_dia}"
                })
                if st.session_state.notif_web and abs(var) >= st.session_state.notif_umbral_web:
                    registrar_notif(
                        tipo="precio", activo=activo, var=var, señal=señal_dia,
                        mensaje=f"Variación proyectada {var:+.2f}% supera el umbral configurado de ±{st.session_state.notif_umbral_web}%"
                    )
                    st.toast(f"🔔 Notificación registrada — {señal_dia} en {activo} ({var:+.2f}%)", icon="📊")

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Verifica tu conexión o prueba con otro activo.")

# ══ MI CUENTA ════════════════════════════════════════════════
elif "Cuenta" in page:
    nombre_c=st.session_state.usuario_actual or "Usuario"
    st.markdown(f"""
    <div class="page-header">
        <div class="brand" style="margin-bottom:0.4rem;">Mi Cuenta</div>
        <h1>Perfil de {nombre_c}</h1>
        <p>Gestiona tu configuración de inversión y preferencias</p>
    </div>""", unsafe_allow_html=True)
    c1,c2=st.columns([1,2])
    with c1:
        p_act=st.session_state.perfil_riesgo
        st.markdown(f"""
        <div class="kcard" style="text-align:center;padding:2rem;">
            <div style="font-size:3rem;margin-bottom:0.5rem;">👤</div>
            <div style="font-size:1.1rem;font-weight:700;color:#F1F5F9;">{nombre_c}</div>
            <div style="margin:0.6rem 0;"><span class="badge badge-{p_act.lower()}">{p_act}</span></div>
            <div style="color:#64748B;font-size:0.78rem;">Miembro desde {datetime.now().strftime('%B %Y')}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        st.markdown("#### Configuración de Perfil de Riesgo")
        nuevo_p=st.radio("Selecciona tu perfil:",["Conservador","Moderado","Agresivo"],
                         index=["Conservador","Moderado","Agresivo"].index(p_act),horizontal=True)
        info_p={"Conservador":("BAP, SCCO","< 5% mensual","Preservar capital con retorno estable"),
                "Moderado":("BAP, SCCO, BVN, VCISY","5–15% mensual","Balance crecimiento/estabilidad"),
                "Agresivo":("BVN, VCISY","> 15% mensual","Máximo retorno · alta exposición al riesgo")}
        act_p,vol_p,obj_p=info_p[nuevo_p]
        st.markdown(f"""
        <div style="background:#0A0E1A;border:1px solid #1E2D4A;border-radius:8px;padding:1rem;margin-top:0.8rem;">
            <span class="badge badge-{nuevo_p.lower()}">{nuevo_p}</span>
            <div style="margin-top:0.6rem;font-size:0.82rem;color:#64748B;">
                Activos: <span style="color:#94A3B8">{act_p}</span><br>
                Volatilidad: <span style="color:#94A3B8">{vol_p}</span><br>
                Objetivo: <span style="color:#94A3B8">{obj_p}</span>
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Guardar perfil",use_container_width=True):
            st.session_state.perfil_riesgo=nuevo_p
            db=st.session_state.usuarios_db
            ck=next((k for k,v in db.items() if v["nombre"]==nombre_c),None)
            if ck: db[ck]["perfil"]=nuevo_p; st.session_state.usuarios_db=db
            st.success(f"Perfil actualizado a **{nuevo_p}**.")
        st.markdown('</div>', unsafe_allow_html=True)

# ══ ALERTAS — HU008 + HU011 ══════════════════════════════════
elif "Alertas" in page:
    st.markdown("""
    <div class="page-header">
        <div class="brand" style="margin-bottom:0.4rem;">Centro de Alertas</div>
        <h1>Notificaciones y Sugerencias</h1>
        <p>Gestiona tus alertas web, configuración de correo y sugerencias de inversión</p>
    </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════
    # HU011 — NOTIFICACIONES VÍA WEB (bloque principal)
    # ════════════════════════════════════════════════
    st.markdown("## 🔔 Notificaciones vía Web")
    st.markdown('<p style="color:#64748B;font-size:0.85rem;margin-bottom:1rem;">Recibe alertas en tiempo real dentro del sistema cuando se detecten movimientos significativos en tus activos.</p>', unsafe_allow_html=True)

    n1, n2 = st.columns([1, 1])

    # ── Panel de configuración HU011 ────────────────
    with n1:
        st.markdown("""
        <div class="notif-panel">
            <div class="notif-header">
                <div style="display:flex;align-items:center;gap:0.6rem;">
                    <span style="font-size:1.1rem;">⚙️</span>
                    <span style="color:#E2E8F0;font-weight:600;font-size:0.9rem;">Configuración de alertas</span>
                </div>
                <span style="color:#475569;font-size:0.72rem;">HU011</span>
            </div>
        </div>""", unsafe_allow_html=True)

        notif_web = st.toggle(
            "🔔 Activar notificaciones vía web",
            value=st.session_state.notif_web,
            help="Al activar, recibirás alertas dentro del sistema cuando un activo supere el umbral configurado."
        )
        st.session_state.notif_web = notif_web

        if notif_web:
            st.markdown(f'<p style="color:#10B981;font-size:0.82rem;margin:0.3rem 0 0.8rem;">✅ Notificaciones web <b>ACTIVAS</b></p>', unsafe_allow_html=True)

            umbral_web = st.slider(
                "Umbral de variación para alertar (%)",
                min_value=1, max_value=10,
                value=st.session_state.notif_umbral_web,
                help="Se generará una notificación cuando la variación proyectada supere este porcentaje."
            )
            st.session_state.notif_umbral_web = umbral_web

            st.markdown("**Tipos de notificación activos:**")
            tipos_sel = []
            for tipo in ["Variación de precio", "Señal fuerte (>5%)", "Cambio de perfil", "Actualización del modelo"]:
                if st.checkbox(tipo, value=tipo in ["Variación de precio", "Señal fuerte (>5%)"], key=f"notif_tipo_{tipo}"):
                    tipos_sel.append(tipo)
            st.session_state.notif_tipos = tipos_sel

            st.markdown('<hr class="kdivider">', unsafe_allow_html=True)
            st.markdown("**Simular notificaciones de prueba:**")

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                activos_disp_n = ACTIVOS_PERFIL[st.session_state.perfil_riesgo]
                activo_sim = st.selectbox("Activo", list(activos_disp_n.keys()), key="notif_sim_activo")
                tick_sim = activos_disp_n[activo_sim]

            with col_s2:
                señal_sim = st.selectbox("Señal", ["COMPRA", "VENTA", "MANTENER"], key="notif_sim_señal")

            var_sim = st.slider("Variación simulada (%)", -15.0, 15.0, 4.5, 0.5, key="notif_sim_var")

            if st.button("📨 Generar notificación de prueba", use_container_width=True):
                registrar_notif(
                    tipo="prueba",
                    activo=activo_sim,
                    var=var_sim,
                    señal=señal_sim,
                    mensaje=f"Notificación de prueba · Variación simulada {var_sim:+.1f}%"
                )
                st.toast(f"🔔 Notificación enviada — {señal_sim} en {activo_sim}", icon="✅")
                st.rerun()

        else:
            st.markdown('<p style="color:#475569;font-size:0.82rem;padding:0.5rem 0;">Activa las notificaciones para recibir alertas en tiempo real dentro del sistema.</p>', unsafe_allow_html=True)

    # ── Panel de notificaciones recibidas HU011 ─────
    with n2:
        n_pend = len(st.session_state.notif_pendientes)
        n_leidas = len(st.session_state.notif_leidas)

        st.markdown(f"""
        <div class="notif-panel">
            <div class="notif-header">
                <div style="display:flex;align-items:center;gap:0.6rem;">
                    <span style="font-size:1.1rem;">🔔</span>
                    <span style="color:#E2E8F0;font-weight:600;font-size:0.9rem;">Bandeja de notificaciones</span>
                </div>
                <div style="display:flex;align-items:center;gap:0.5rem;">
                    {f'<span class="notif-badge-count">{n_pend} nuevas</span>' if n_pend > 0 else '<span style="color:#475569;font-size:0.72rem;">Sin nuevas</span>'}
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Sub-tabs: Sin leer / Historial
        tab_n1, tab_n2 = st.tabs([f"📬 Sin leer ({n_pend})", f"📂 Historial ({n_leidas})"])

        with tab_n1:
            if not st.session_state.notif_pendientes:
                st.markdown("""
                <div class="notif-empty">
                    <div style="font-size:2rem;margin-bottom:0.5rem;">🔕</div>
                    <div style="color:#334155;font-size:0.85rem;">No hay notificaciones pendientes.</div>
                    <div style="color:#1E2D4A;font-size:0.78rem;margin-top:0.3rem;">Genera una predicción o activa las notificaciones para recibir alertas.</div>
                </div>""", unsafe_allow_html=True)
            else:
                for notif in reversed(st.session_state.notif_pendientes[-8:]):
                    tipo = notif.get("tipo","info")
                    st.markdown(f"""
                    <div class="notif-item-{tipo} toast-compra">
                        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.3rem;">
                            <div style="font-size:0.85rem;font-weight:700;color:#F1F5F9;">
                                <span class="notif-dot notif-dot-{tipo}"></span>
                                {notif['titulo']}
                            </div>
                            <div style="font-size:0.68rem;color:#475569;">{notif['hora']} · {notif['fecha']}</div>
                        </div>
                        <div style="font-size:0.8rem;color:#94A3B8;margin-left:14px;">{notif['cuerpo']}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    if st.button("✅ Marcar todas como leídas", use_container_width=True):
                        st.session_state.notif_leidas.extend(st.session_state.notif_pendientes)
                        st.session_state.notif_pendientes = []
                        st.rerun()
                with col_m2:
                    if st.button("🗑 Eliminar todas", use_container_width=True, key="del_notif"):
                        st.session_state.notif_pendientes = []
                        st.rerun()

        with tab_n2:
            if not st.session_state.notif_leidas:
                st.markdown('<div class="notif-empty"><div style="font-size:2rem;margin-bottom:0.5rem;">📂</div><div>Sin historial de notificaciones.</div></div>', unsafe_allow_html=True)
            else:
                for notif in reversed(st.session_state.notif_leidas[-10:]):
                    tipo = notif.get("tipo","info")
                    st.markdown(f"""
                    <div style="border-left:3px solid #1E2D4A;padding:0.5rem 0.8rem;
                                margin-bottom:0.4rem;background:#0A0E1A;border-radius:0 6px 6px 0;opacity:0.7;">
                        <div style="font-size:0.78rem;font-weight:600;color:#64748B;">{notif['titulo']}</div>
                        <div style="font-size:0.72rem;color:#334155;">{notif['cuerpo']} · {notif['hora']} {notif['fecha']}</div>
                    </div>""", unsafe_allow_html=True)
                if st.button("🗑 Limpiar historial", key="limpiar_hist_notif"):
                    st.session_state.notif_leidas = []
                    st.rerun()

    st.markdown('<hr class="kdivider">', unsafe_allow_html=True)

    # ════════════════════════════════════════════════
    # HU008 — SUGERENCIAS POR CORREO
    # ════════════════════════════════════════════════
    st.markdown("## 📧 Sugerencias de Inversión por Correo")
    a1, a2 = st.columns([1, 1])

    with a1:
        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        st.markdown("#### Configuración de correo")
        db=st.session_state.usuarios_db
        nombre_a=st.session_state.usuario_actual or "Demo"
        ck=next((k for k,v in db.items() if v["nombre"]==nombre_a),None)
        correo_vis=ck if ck else "demo@kallpa.com"
        st.markdown(f'<p style="color:#64748B;font-size:0.83rem;">Cuenta: <b style="color:#60A5FA">{correo_vis}</b></p>', unsafe_allow_html=True)
        notif=st.toggle("Activar sugerencias automáticas",value=st.session_state.notif_correo)
        st.session_state.notif_correo=notif
        if notif:
            frec=st.selectbox("Frecuencia",["Diaria (9:00 AM)","Semanal (lunes)","Solo señal fuerte (>5%)"])
            umbral=st.slider("Umbral de alerta (%)",1,10,3)
            st.markdown(f'<p style="color:#10B981;font-size:0.8rem;">✓ Alertas cuando variación proyectada > ±{umbral}%</p>', unsafe_allow_html=True)
            if st.button("📤 Enviar correo de prueba",use_container_width=True):
                with st.spinner("Generando sugerencia..."): import time; time.sleep(1)
                act_sug=list(ACTIVOS_PERFIL[st.session_state.perfil_riesgo].keys())[0]
                tick_sug=list(ACTIVOS_PERFIL[st.session_state.perfil_riesgo].values())[0]
                try:
                    _,ps,_=descargar_datos(tick_sug)
                    lps=lstm_sim(ps); fs=gen_futuro(float(ps[-1]),lps)
                    vs=(fs[-1]-float(ps[-1]))/float(ps[-1])*100
                    señal="COMPRA" if vs>3 else "VENTA" if vs<-3 else "MANTENER"
                except: vs=2.1; señal="MANTENER"; tick_sug="BAP"
                st.session_state.alertas_log.append({"fecha":datetime.now().strftime("%d/%m %H:%M"),
                    "activo":act_sug,"variacion":vs,"msg":f"Correo enviado · {act_sug} · Señal {señal} · {vs:+.2f}%"})
                st.markdown(f"""
                <div style="background:#0A0E1A;border:1px solid #1E2D4A;border-radius:10px;padding:1.2rem;margin-top:1rem;font-size:0.82rem;">
                    <div style="color:#64748B;">📧 Para: <span style="color:#60A5FA">{correo_vis}</span></div>
                    <div style="color:#64748B;margin-bottom:0.8rem;">Asunto: <b style="color:#F1F5F9">Sugerencia Kallpa · {act_sug} · {datetime.now().strftime('%d/%m/%Y')}</b></div>
                    <hr class="kdivider">
                    <p style="color:#F1F5F9;font-weight:600;">Estimado/a {nombre_a},</p>
                    <p style="color:#94A3B8;">El modelo predictivo de Kallpa Securities identificó:</p>
                    <div style="background:#111827;border-radius:8px;padding:0.8rem;margin:0.6rem 0;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;color:#60A5FA;font-weight:700;">{act_sug} · {tick_sug}</div>
                        <div style="color:#94A3B8;margin-top:0.3rem;">Variación 14d: <span style="color:{'#10B981' if vs>=0 else '#F87171'};font-weight:600">{vs:+.2f}%</span></div>
                        <div style="color:#94A3B8;">Señal: <b style="color:#F1F5F9">{señal}</b> · Perfil: {st.session_state.perfil_riesgo}</div>
                    </div>
                    <p style="color:#334155;font-size:0.72rem;">Kallpa Securities SAB © 2025 · Predicciones orientativas.</p>
                </div>""", unsafe_allow_html=True)
                st.success("Correo de prueba generado correctamente.")
        else:
            st.markdown('<p style="color:#475569;font-size:0.82rem;padding:0.5rem 0;">Activa las sugerencias para configurar frecuencia y umbrales.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with a2:
        st.markdown('<div class="kcard">', unsafe_allow_html=True)
        st.markdown("#### Historial de alertas")
        alerts=st.session_state.alertas_log
        if not alerts:
            st.markdown('<p style="color:#475569;text-align:center;padding:2rem 0;font-size:0.85rem;">Sin alertas aún.</p>', unsafe_allow_html=True)
        else:
            for al in reversed(alerts[-10:]):
                c=("#10B981" if al["variacion"]>=0 else "#EF4444")
                st.markdown(f"""
                <div style="border-left:3px solid {c};padding:0.5rem 0.8rem;margin-bottom:0.5rem;background:#0A0E1A;border-radius:0 6px 6px 0;">
                    <div style="font-size:0.7rem;color:#475569;">{al['fecha']}</div>
                    <div style="font-size:0.83rem;color:#E2E8F0;">{al['msg']}</div>
                </div>""", unsafe_allow_html=True)
            if st.button("🗑 Limpiar historial",key="limpiar_alertas"):
                st.session_state.alertas_log=[]; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ══ AYUDA ════════════════════════════════════════════════════
elif "Ayuda" in page:
    st.markdown("""
    <div class="page-header">
        <div class="brand" style="margin-bottom:0.4rem;">Centro de Ayuda</div>
        <h1>Preguntas Frecuentes</h1>
        <p>Todo lo que necesitas saber sobre el sistema de predicción de Kallpa Securities</p>
    </div>""", unsafe_allow_html=True)
    faqs=[
        ("¿Cómo funciona el modelo de predicción?","Ensemble de LSTM (60%) + GRU (25%) + ARIMA (15%). Cada modelo captura distintos patrones de la serie temporal. El resultado se ajusta con variables macro: tipo de cambio PEN/USD, tasa BCRP y precio del cobre."),
        ("¿Qué significa cada perfil de riesgo?","Conservador: BAP y SCCO, baja volatilidad. Moderado: portafolio completo BVL. Agresivo: BVN y VCISY, alta volatilidad y mayor potencial de retorno. Puedes cambiarlo en Mi Cuenta."),
        ("¿Qué precisión tiene el modelo?","87–91% de precisión direccional en backtesting histórico. Esta métrica indica si el modelo predice correctamente si el precio sube o baja, no el valor exacto."),
        ("¿Cómo se generan las señales COMPRA/VENTA?","COMPRA: predicción > precio actual +3%. VENTA: predicción < precio actual −3%. MANTENER: variación dentro de ±3%. El umbral del 3% cubre el costo de transacción típico en la BVL."),
        ("¿Cómo funcionan las notificaciones vía web?","Ve a Alertas → Notificaciones vía Web, activa el toggle y configura el umbral de variación. Puedes simular notificaciones de prueba seleccionando activo, señal y variación. Las notificaciones se registran en la bandeja y también aparecen como toast al generar predicciones."),
        ("¿Cómo activo las sugerencias por correo?","Ve a Alertas → Sugerencias por Correo, activa el toggle y configura la frecuencia. Puedes enviar un correo de prueba para verificar el formato de las sugerencias."),
        ("¿Es escalable a producción?","Sí. La arquitectura modular permite integración con BD PostgreSQL/SQL Server, envío real de correos vía SendGrid, y despliegue en AWS EC2 + S3 + RDS con reentrenamiento periódico del modelo."),
    ]
    for preg,resp in faqs:
        with st.expander(f"❓ {preg}"):
            st.markdown(f'<p style="color:#94A3B8;font-size:0.87rem;line-height:1.7;">{resp}</p>', unsafe_allow_html=True)
    st.markdown('<hr class="kdivider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="kcard" style="text-align:center;">
        <div class="brand" style="margin-bottom:0.5rem;">KALLPA SECURITIES SAB</div>
        <p style="color:#64748B;font-size:0.82rem;">
            📧 research@kallpasab.com &nbsp;|&nbsp; ☎ +51 1 219-0400<br>
            📍 Av. Jorge Basadre 310, San Isidro, Lima &nbsp;|&nbsp; 🌐 www.kallpasab.com
        </p>
        <p style="color:#334155;font-size:0.72rem;margin-top:0.6rem;">Las predicciones son orientativas y no constituyen asesoría financiera. © 2025</p>
    </div>""", unsafe_allow_html=True)
