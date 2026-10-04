import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import os
import re
import base64
from pathlib import Path

try:
    from ybus import build_ybus
    from newton_raphson import newton_raphson
except ImportError:
    st.error("⚠️ Arquivos do motor não encontrados. Verifique se ybus.py, mismatch.py, jacobian.py e newton_raphson.py estão na mesma pasta.")
    st.stop()

# ======================================================
# FUNÇÕES AUXILIARES
# ======================================================
def formatar_vetor_latex(vec, precisao=4):
    elementos = [f"{v:.{precisao}f}" for v in vec]
    return r"\begin{bmatrix} " + r" \\ ".join(elementos) + r" \end{bmatrix}"

def formatar_ybus(ybus):
    df = pd.DataFrame(ybus)
    return df.map(lambda c: f"{c.real:.4f} {c.imag:+.4f}j")

# ======================================================
# CONFIGURAÇÃO DA PÁGINA
# ======================================================
st.set_page_config(page_title="PowerFlow", layout="wide", page_icon="⚡")

# Inicializa navegação
if "pagina" not in st.session_state:
    st.session_state.pagina = "home"

# ======================================================
# CSS GLOBAL — sem sidebar, cards da home, botão voltar
# ======================================================
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] { display: none !important; }

    .pf-main-title {
        font-size: clamp(40px, 8vw, 72px);
        font-weight: 900;
        text-align: center;
        color: #111;
        letter-spacing: -2px;
        margin-bottom: 6px;
        line-height: 1;
    }
    .pf-main-sub {
        text-align: center;
        color: #666;
        font-size: 16px;
        margin-bottom: 52px;
    }
    .pf-card {
        background: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 20px;
        padding: 44px 28px 36px 28px;
        text-align: center;
        min-height: 240px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 16px;
        transition: box-shadow 0.25s, border-color 0.25s, transform 0.2s;
    }
    .pf-card:hover {
        box-shadow: 0 10px 36px rgba(25,118,210,0.16);
        border-color: #1976d2;
        transform: translateY(-5px);
    }
    .pf-card-icon  { font-size: 56px; line-height: 1; }
    .pf-card-title { font-size: 22px; font-weight: 800; color: #111; margin: 0; }
    .pf-card-desc  { font-size: 14px; color: #666; margin: 0; line-height: 1.5; }
    .pf-divider    { border: none; border-top: 1px solid #e0e0e0; margin: 0 0 2rem 0; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# HOME
# ======================================================
if st.session_state.pagina == "home":
    st.markdown('<div class="pf-main-title">Power Flow</div>', unsafe_allow_html=True)
    st.markdown('<div class="pf-main-sub">Plataforma educacional para ensino de fluxo de potência em sistemas elétricos de potência</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""
        <div class="pf-card">
            <div class="pf-card-icon">🖥️</div>
            <div class="pf-card-title">Simulação FC</div>
            <div class="pf-card-desc">Monte a topologia da rede, configure os parâmetros e execute o fluxo de potência pelo Método de Newton-Raphson.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Acessar Simulação", key="btn_sim", use_container_width=True, type="primary"):
            st.session_state.pagina = "simulacao"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="pf-card">
            <div class="pf-card-icon">🎓</div>
            <div class="pf-card-title">Tutorial da Plataforma</div>
            <div class="pf-card-desc">Aprenda a usar a plataforma passo a passo: barras, linhas, transformadores e interpretação dos resultados.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Acessar Tutorial", key="btn_tut", use_container_width=True):
            st.session_state.pagina = "tutorial"
            st.rerun()

    with col3:
        st.markdown("""
        <div class="pf-card">
            <div class="pf-card-icon">📖</div>
            <div class="pf-card-title">Teoria SEP</div>
            <div class="pf-card-desc">Fundamentos teóricos: modelo π, matriz Ybus, classificação de barras, Jacobiana e o Método de Newton-Raphson.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Acessar Teoria", key="btn_teo", use_container_width=True):
            st.session_state.pagina = "teoria"
            st.rerun()

    st.stop()

# ======================================================
# BOTÃO VOLTAR — comum a todas as subpáginas
# ======================================================
if st.button("← Voltar ao início"):
    st.session_state.pagina = "home"
    st.rerun()
st.markdown('<hr class="pf-divider">', unsafe_allow_html=True)

# ======================================================
# PÁGINA: TUTORIAL
# ======================================================
if st.session_state.pagina == "tutorial":
    st.markdown("## 🎓 Tutorial da Plataforma PowerFlow")
    st.info("Esta seção está em construção. Em breve trará guias passo a passo com exemplos resolvidos.", icon="🚧")
    st.stop()

# ======================================================
# PÁGINA: TEORIA SEP
# Conteúdo conceitual, sem exemplos numéricos resolvidos: a resolução de
# exercícios fica sob mediação do professor. As seções "Para refletir"
# trazem perguntas abertas, sem gabarito.
# Figuras em SVG geradas por código; a caricatura do MNR é lida de
# assets/newton_raphson.jpg (se o arquivo faltar, o card mostra 📐).
# ======================================================

# ------------------------------------------------------
# TEORIA SEP — Figuras vetoriais (SVG)
# ------------------------------------------------------
AZUL   = "#1976d2"
ROXO   = "#6a1b9a"
VERM   = "#d32f2f"
VERDE  = "#2e7d32"
LARANJ = "#ef6c00"
AMAR   = "#f9a825"
TINTA  = "#212121"
CINZA  = "#616161"
FONTE  = "Arial, Helvetica, sans-serif"

DEFS_SETA = (
    '<defs>'
    '<marker id="seta" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{TINTA}"/></marker>'
    '<marker id="setaAzul" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{AZUL}"/></marker>'
    '</defs>'
)


# ======================================================================
# Primitivas
# ======================================================================
def _svg(w, h, corpo, fundo=True):
    bg = f'<rect x="0" y="0" width="{w}" height="{h}" rx="14" fill="#ffffff"/>' if fundo else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'font-family="{FONTE}">{DEFS_SETA}{bg}{corpo}</svg>')


def _markup(conteudo, size):
    """Converte  base_{sub}  e  base^{sup}  em <tspan> com deslocamento em px."""
    partes = re.split(r"([_^]\{[^}]*\})", conteudo)
    out, nivel = "", 0.0
    for p in partes:
        if not p:
            continue
        if p[0] in "_^" and p[1:2] == "{":
            alvo = 0.28 * size if p[0] == "_" else -0.42 * size
            out += f'<tspan font-size="{0.72*size:.1f}" dy="{alvo-nivel:.1f}">{p[2:-1]}</tspan>'
            nivel = alvo
        elif nivel:
            dx = f' dx="{0.28*size:.1f}"' if p.startswith(" ") else ""
            out += f'<tspan dy="{-nivel:.1f}"{dx}>{p.lstrip(" ")}</tspan>'
            nivel = 0.0
        else:
            out += p
    return out


def _t(x, y, conteudo, size=14, cor=TINTA, anchor="middle", peso="normal", italico=False):
    estilo = ' font-style="italic"' if italico else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{cor}" text-anchor="{anchor}" '
            f'font-weight="{peso}"{estilo}>{_markup(conteudo, size)}</text>')


def _barra(x, y1, y2, cor=TINTA):
    return f'<rect x="{x-4}" y="{y1}" width="8" height="{y2-y1}" rx="3" fill="{cor}"/>'


def _terra(x, y, cor=TINTA):
    return (f'<path d="M{x-12},{y} H{x+12} M{x-8},{y+5} H{x+8} M{x-4},{y+10} H{x+4}" '
            f'stroke="{cor}" stroke-width="2" fill="none"/>')


def _fio(d, cor=TINTA, w=2.2, extra=""):
    return f'<path d="{d}" stroke="{cor}" stroke-width="{w}" fill="none" {extra}/>'


def _capacitor_shunt(x, y_top, y_cap, y_terra, cor=AZUL):
    return (_fio(f"M{x},{y_top} V{y_cap-5} M{x},{y_cap+5} V{y_terra}")
            + f'<path d="M{x-14},{y_cap-5} H{x+14} M{x-14},{y_cap+5} H{x+14}" stroke="{cor}" stroke-width="3.5"/>'
            + _terra(x, y_terra))


def _resistor(x1, x2, y, cor=VERM, amp=7, n=5):
    dx = (x2 - x1) / (2 * n)
    d = f"M{x1},{y}"
    for i in range(2 * n):
        d += f" L{x1 + dx * (i + 0.5):.1f},{y - amp if i % 2 == 0 else y + amp}"
    d += f" L{x2},{y}"
    return _fio(d, cor, 2.5, 'stroke-linejoin="round"')


def _indutor(x1, x2, y, cor=AZUL, n=4):
    r = (x2 - x1) / (2 * n)
    d = f"M{x1},{y}" + "".join(f" a{r:.2f},{r:.2f} 0 0 1 {2*r:.2f},0" for _ in range(n))
    return _fio(d, cor, 2.5)


def _caixa_adm(x1, y1, w, h, texto, cor=ROXO, size=14):
    return (f'<rect x="{x1}" y="{y1}" width="{w}" height="{h}" rx="6" fill="#ffffff" stroke="{cor}" stroke-width="2.5"/>'
            + _t(x1 + w / 2, y1 + h / 2 + size * 0.35, texto, size, cor, peso="bold"))


def _gerador(cx, cy, r=17, cor=TINTA):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" stroke="{cor}" stroke-width="2.2"/>'
            + _fio(f"M{cx-9},{cy} q4.5,-9 9,0 t9,0", cor, 2))


def _trafo_ideal(cx, cy, r=18, cor=ROXO):
    return (f'<circle cx="{cx-r*0.62:.1f}" cy="{cy}" r="{r}" fill="none" stroke="{cor}" stroke-width="2.5"/>'
            f'<circle cx="{cx+r*0.62:.1f}" cy="{cy}" r="{r}" fill="none" stroke="{cor}" stroke-width="2.5"/>')


def _seta(x1, y1, x2, y2, cor=TINTA, w=2):
    marc = "setaAzul" if cor == AZUL else "seta"
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{cor}" stroke-width="{w}" marker-end="url(#{marc})"/>'


def _bloco(x, y, w, h, linhas, cor=AZUL, fundo="#e3f0fb", size=14, rx=8, peso_primeira=True):
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fundo}" stroke="{cor}" stroke-width="2"/>'
    n = len(linhas)
    lh = size * 1.35
    y0 = y + h / 2 - (n - 1) * lh / 2 + size * 0.35
    for i, txt in enumerate(linhas):
        peso = "bold" if (i == 0 and peso_primeira) else "normal"
        out += _t(x + w / 2, y0 + i * lh, txt, size, TINTA, peso=peso)
    return out


def _losango(cx, cy, w, h, linhas, cor=LARANJ, fundo="#fff3e0", size=14):
    out = (f'<polygon points="{cx},{cy-h/2} {cx+w/2},{cy} {cx},{cy+h/2} {cx-w/2},{cy}" '
           f'fill="{fundo}" stroke="{cor}" stroke-width="2"/>')
    n = len(linhas)
    lh = size * 1.3
    y0 = cy - (n - 1) * lh / 2 + size * 0.35
    for i, txt in enumerate(linhas):
        out += _t(cx, y0 + i * lh, txt, size, TINTA, peso="bold")
    return out


def _terminal(x, y, w, h, texto, cor=VERDE, fundo="#e8f5e9", size=14):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" fill="{fundo}" stroke="{cor}" stroke-width="2"/>'
            + _t(x + w / 2, y + h / 2 + size * 0.35, texto, size, TINTA, peso="bold"))


# ======================================================================
# Contorno simplificado do Brasil (lon, lat) — uso esquemático
# ======================================================================
_BRASIL_LONLAT = [
    (-60, 5), (-51, 4), (-50, 0), (-44, -2.5), (-38.5, -3.7), (-35, -5.5), (-35, -9),
    (-39, -13), (-39.5, -18), (-41, -22), (-45, -23.8), (-48.5, -26), (-49, -29),
    (-53, -33.7), (-57.5, -30), (-54, -26), (-55, -24), (-58, -20), (-60, -16),
    (-65, -11), (-70, -11), (-74, -7.5), (-70, -4), (-69.5, 1), (-66, 1.5), (-64, 4),
]


def _proj(lon, lat, esc, ox, oy):
    return ox + esc * (lon + 75) * 5, oy + esc * (6 - lat) * 5


def _contorno_brasil(esc, ox, oy, fill="#e8f0fa", stroke="#90a4ae", w=2):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (_proj(lo, la, esc, ox, oy) for lo, la in _BRASIL_LONLAT))
    return (f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" '
            f'stroke-linejoin="round"/>')


# Posições esquemáticas (lon, lat) dos subsistemas e de grandes usinas
_SUBSIS = {
    "N":     ((-58.6, -4.4), VERDE,  "N"),
    "NE":    ((-40.6, -8.4), LARANJ, "NE"),
    "SE/CO": ((-47.0, -18.4), AZUL,  "SE/CO"),
    "S":     ((-52.6, -27.6), ROXO,  "S"),
}
_MADEIRA    = (-64.0, -9.2)
_BELO_MONTE = (-51.6, -2.8)


# ======================================================================
# Ícones dos cards
# ======================================================================
def icone_sin():
    esc, ox, oy = 0.55, 2, 2
    corpo = _contorno_brasil(esc, ox, oy, w=1.5)
    p = {k: _proj(*v[0], esc, ox, oy) for k, v in _SUBSIS.items()}
    for a, b in [("N", "NE"), ("N", "SE/CO"), ("NE", "SE/CO"), ("SE/CO", "S")]:
        corpo += _fio(f"M{p[a][0]:.1f},{p[a][1]:.1f} L{p[b][0]:.1f},{p[b][1]:.1f}", AZUL, 2.6)
    for k, (_, cor, _) in _SUBSIS.items():
        corpo += f'<circle cx="{p[k][0]:.1f}" cy="{p[k][1]:.1f}" r="6.5" fill="{cor}" stroke="#fff" stroke-width="1.5"/>'
    return _svg(118, 116, corpo, fundo=False)


def icone_pi():
    c = (_barra(14, 16, 74) + _barra(186, 16, 74)
         + _fio("M18,34 H62 M138,34 H182")
         + _resistor(62, 92, 34, n=3, amp=6) + _indutor(92, 138, 34, n=3)
         + _capacitor_shunt(40, 34, 62, 80) + _capacitor_shunt(160, 34, 62, 80))
    return _svg(200, 100, c, fundo=False)


def icone_trafo():
    c = (_barra(14, 16, 74) + _barra(186, 16, 74)
         + _fio("M18,42 H45 M83,42 H118 M168,42 H182")
         + _trafo_ideal(64, 42, r=15)
         + f'<rect x="118" y="30" width="50" height="24" rx="4" fill="#fff" stroke="{ROXO}" stroke-width="2.4"/>'
         + _t(64, 18, "1 : a", 13, ROXO, peso="bold")
         + _t(143, 47, "y", 14, ROXO, peso="bold", italico=True))
    return _svg(200, 92, c, fundo=False)


def icone_ybus():
    padrao = [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 2], [0, 0, 2, 1]]
    c = (_fio("M22,8 H12 V108 H22", TINTA, 3) + _fio("M114,8 H124 V108 H114", TINTA, 3))
    for i in range(4):
        for j in range(4):
            cx, cy = 32 + j * 24, 22 + i * 24
            v = padrao[i][j]
            if i == j:
                c += f'<circle cx="{cx}" cy="{cy}" r="8" fill="{AZUL}"/>'
            elif v == 1:
                c += f'<circle cx="{cx}" cy="{cy}" r="8" fill="{LARANJ}"/>'
            elif v == 2:
                c += f'<circle cx="{cx}" cy="{cy}" r="8" fill="{ROXO}"/>'
            else:
                c += f'<circle cx="{cx}" cy="{cy}" r="3" fill="#bdbdbd"/>'
    return _svg(136, 116, c, fundo=False)


# ======================================================================
# INTRODUÇÃO
# ======================================================================
def fig_estrutura_sep():
    W, H = 816, 228
    etapas = [
        ("Geração", "usinas"),
        ("Subestação", "elevadora"),
        ("Transmissão", "Rede Básica ≥ 230 kV"),
        ("Subestação", "abaixadora"),
        ("Distribuição", "e consumo"),
    ]
    c = ""
    bw, gap, y, bh = 136, 26, 26, 128
    for i, (t1, t2) in enumerate(etapas):
        x = 16 + i * (bw + gap)
        destaque = i in (1, 2, 3)
        c += (f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="10" '
              f'fill="{"#e3f0fb" if destaque else "#f5f5f5"}" stroke="{AZUL if destaque else "#9e9e9e"}" stroke-width="2"/>')
        cx, cy = x + bw / 2, y + 44
        if i == 0:
            c += _gerador(cx, cy, 19)
        elif i in (1, 3):
            c += _trafo_ideal(cx, cy, 15, AZUL)
        elif i == 2:
            c += _fio(f"M{cx-15},{cy+22} L{cx},{cy-22} L{cx+15},{cy+22} M{cx-22},{cy-10} H{cx+22} "
                      f"M{cx-12},{cy+2} H{cx+12} M{cx-8},{cy-10} L{cx+11},{cy+16} M{cx+8},{cy-10} L{cx-11},{cy+16}",
                      TINTA, 2)
        else:
            c += _fio(f"M{cx-17},{cy+18} V{cy-2} L{cx},{cy-18} L{cx+17},{cy-2} V{cy+18} Z", TINTA, 2.2)
            c += f'<rect x="{cx-5}" y="{cy+5}" width="10" height="13" fill="{TINTA}"/>'
        c += _t(cx, y + 94, t1, 15, peso="bold")
        c += _t(cx, y + 114, t2, 12.5, CINZA)
        if i < 4:
            c += _seta(x + bw + 3, y + bh / 2, x + bw + gap - 3, y + bh / 2)
    x1, x2 = 16 + 1 * (bw + gap), 16 + 3 * (bw + gap) + bw
    c += _fio(f"M{x1},{y+bh+16} V{y+bh+24} H{x2} V{y+bh+16}", AZUL, 2)
    c += _t((x1 + x2) / 2, y + bh + 46, "Escopo usual dos estudos de fluxo de potência em redes de transmissão", 14, AZUL, peso="bold")
    return _svg(W, H, c)


def fig_sin():
    W, H = 760, 450
    esc, ox, oy = 2.0, 10, 14
    c = _contorno_brasil(esc, ox, oy)
    p = {k: _proj(*v[0], esc, ox, oy) for k, v in _SUBSIS.items()}
    mad = _proj(*_MADEIRA, esc, ox, oy)
    bmo = _proj(*_BELO_MONTE, esc, ox, oy)
    se = p["SE/CO"]
    # Elos CC
    c += _fio(f"M{mad[0]:.0f},{mad[1]:.0f} Q{mad[0]+30:.0f},{se[1]+30:.0f} {se[0]-14:.0f},{se[1]+6:.0f}",
              VERM, 3, 'stroke-dasharray="9 6"')
    c += _fio(f"M{bmo[0]:.0f},{bmo[1]:.0f} Q{bmo[0]+70:.0f},{bmo[1]+90:.0f} {se[0]+6:.0f},{se[1]-14:.0f}",
              VERM, 3, 'stroke-dasharray="9 6"')
    # Interligações CA
    for a, b in [("N", "NE"), ("N", "SE/CO"), ("NE", "SE/CO"), ("SE/CO", "S")]:
        c += _fio(f"M{p[a][0]:.0f},{p[a][1]:.0f} L{p[b][0]:.0f},{p[b][1]:.0f}", AZUL, 4.5)
    # Usinas
    for (x, y), nome, dx, anc in [(mad, "Rio Madeira", -12, "end"), (bmo, "Belo Monte", 0, "middle")]:
        c += f'<rect x="{x-8:.0f}" y="{y-8:.0f}" width="16" height="16" rx="3" fill="{VERM}" stroke="#fff" stroke-width="2"/>'
        c += _t(x + dx, y - 14, nome, 12.5, VERM, anc, "bold")
    # Subsistemas
    for k, (_, cor, rot) in _SUBSIS.items():
        x, y = p[k]
        c += f'<circle cx="{x:.0f}" cy="{y:.0f}" r="21" fill="{cor}" stroke="#fff" stroke-width="3"/>'
        c += _t(x, y + 5, rot, 12 if len(rot) > 2 else 14, "#fff", peso="bold")
    # Legenda
    lx = 470
    c += _t(lx, 52, "Sistema Interligado Nacional", 18, peso="bold", anchor="start")
    c += _t(lx, 74, "representação esquemática, fora de escala", 12.5, CINZA, "start", italico=True)
    itens = [("N", "Norte", VERDE), ("NE", "Nordeste", LARANJ), ("SE/CO", "Sudeste/Centro-Oeste", AZUL), ("S", "Sul", ROXO)]
    for i, (_, nome, cor) in enumerate(itens):
        yy = 112 + i * 30
        c += f'<circle cx="{lx+10}" cy="{yy-5}" r="9" fill="{cor}"/>'
        c += _t(lx + 28, yy, f"Subsistema {nome}", 14, anchor="start")
    yy = 250
    c += _fio(f"M{lx},{yy-5} H{lx+30}", AZUL, 4.5) + _t(lx + 40, yy, "Interligação CA entre subsistemas", 13.5, anchor="start")
    c += _fio(f"M{lx},{yy+25} H{lx+30}", VERM, 3, 'stroke-dasharray="9 6"') + _t(lx + 40, yy + 30, "Elo de corrente contínua (HVDC)", 13.5, anchor="start")
    c += f'<rect x="{lx+7}" y="{yy+47}" width="16" height="16" rx="3" fill="{VERM}"/>' + _t(lx + 40, yy + 60, "Grande usina hidrelétrica", 13.5, anchor="start")
    c += _t(lx, yy + 110, "Operação coordenada pelo ONS", 13.5, CINZA, "start", "bold")
    c += _t(lx, yy + 130, "(Operador Nacional do Sistema Elétrico)", 12.5, CINZA, "start")
    return _svg(W, H, c)


def fig_tipos_barra():
    W, H = 780, 262
    tipos = [
        ("Barra Slack (Vθ)", AZUL, "#e3f0fb", "|V|, θ", "P, Q", "gerador"),
        ("Barra PV", AMAR, "#fff8e1", "P, |V|", "Q, θ", "gerador"),
        ("Barra PQ", VERDE, "#e8f5e9", "P, Q", "|V|, θ", "carga"),
    ]
    c = ""
    for i, (nome, cor, fundo, esp, inc, simb) in enumerate(tipos):
        x0 = 15 + i * 255
        cx = x0 + 120
        c += f'<rect x="{x0}" y="12" width="240" height="238" rx="12" fill="#fff" stroke="{cor}" stroke-width="2.5"/>'
        c += f'<path d="M{x0},{24} a12,12 0 0 1 12,-12 h216 a12,12 0 0 1 12,12 v28 h-240 z" fill="{fundo}" stroke="{cor}" stroke-width="2.5"/>'
        c += _t(cx, 44, nome, 16, peso="bold")
        c += _barra(cx + 22, 76, 146)
        if simb == "gerador":
            c += _fio(f"M{cx-24},{111} H{cx+22}") + _gerador(cx - 42, 111, 18)
        else:
            c += _fio(f"M{cx+22},{111} H{cx+70} V{126}") + f'<polygon points="{cx+60},{124} {cx+80},{124} {cx+70},{146}" fill="{TINTA}"/>'
            c += _fio(f"M{cx-40},{111} H{cx+22}", "#9e9e9e", 2, 'stroke-dasharray="5 4"')
            c += _t(cx - 44, 104, "rede", 11.5, CINZA, "end", italico=True)
        c += f'<rect x="{x0+16}" y="168" width="208" height="30" rx="6" fill="#e8f5e9"/>'
        c += _t(x0 + 26, 188, "Especificadas:", 13.5, VERDE, "start", "bold") + _t(x0 + 214, 188, esp, 15, TINTA, "end", "bold")
        c += f'<rect x="{x0+16}" y="206" width="208" height="30" rx="6" fill="#fff3e0"/>'
        c += _t(x0 + 26, 226, "Calculadas:", 13.5, LARANJ, "start", "bold") + _t(x0 + 214, 226, inc, 15, TINTA, "end", "bold")
    return _svg(W, H, c)


def fig_subsistemas():
    W, H = 830, 200
    c = ""
    blocos = [
        (20, ["Dados de entrada", "PQ: P e Q  |  PV: P e |V|", "Slack: |V| e θ", "Rede: matriz Ybus"], CINZA, "#f5f5f5"),
        (300, ["Subsistema 1", "Incógnitas: θ (PV, PQ) e |V| (PQ)", "Equações não lineares", "Solução iterativa (MNR)"], AZUL, "#e3f0fb"),
        (580, ["Subsistema 2", "P e Q da barra Slack", "Q das barras PV", "Fluxos e perdas nos ramos"], VERDE, "#e8f5e9"),
    ]
    for x, linhas, cor, fundo in blocos:
        c += _bloco(x, 30, 230, 140, linhas, cor, fundo, 13.5)
    c += _seta(253, 100, 297, 100) + _seta(533, 100, 577, 100)
    return _svg(W, H, c)


# ======================================================================
# LINHA DE TRANSMISSÃO
# ======================================================================
def fig_linha_distribuida():
    W, H = 780, 210
    c = _fio("M40,60 H60") + _fio("M40,140 H740")
    c += _t(30, 64, "k", 16, anchor="end", peso="bold")
    c += _t(752, 64, "m", 16, anchor="start", peso="bold")
    x = 60
    for i in range(5):
        c += _resistor(x + 8, x + 42, 60, n=3, amp=6) + _indutor(x + 48, x + 96, 60, n=3)
        c += _fio(f"M{x},60 H{x+8} M{x+42},60 H{x+48} M{x+96},60 H{x+136}")
        c += _fio(f"M{x+116},60 V{96} M{x+116},{104} V140")
        c += f'<path d="M{x+104},96 H{x+128} M{x+104},104 H{x+128}" stroke="{AZUL}" stroke-width="3.2"/>'
        c += f'<circle cx="{x+116}" cy="60" r="3" fill="{TINTA}"/><circle cx="{x+116}" cy="140" r="3" fill="{TINTA}"/>'
        if i == 0:
            c += _t(x + 25, 40, "r·Δx", 13, VERM, peso="bold") + _t(x + 72, 40, "L·Δx", 13, AZUL, peso="bold")
            c += _t(x + 132, 104, "C·Δx", 13, AZUL, "start", "bold")
        x += 136
    c += _fio("M60,166 V176 M196,166 V176 M60,171 H196", CINZA, 1.6)
    c += _t(128, 192, "trecho elementar Δx", 12.5, CINZA)
    c += _t(470, 192, "Parâmetros distribuídos ao longo de todo o comprimento ℓ da linha", 13.5, CINZA)
    return _svg(W, H, c)


def fig_classificacao_linha():
    W, H = 780, 178
    faixas = [
        (40, 260, "Linha curta", "até ≈ 80 km", "Modelo série (shunt desprezado)", "#e8f5e9", VERDE),
        (260, 500, "Linha média", "≈ 80 a 240 km", "π nominal", "#e3f0fb", AZUL),
        (500, 740, "Linha longa", "acima de ≈ 240 km", "π equivalente (correção hiperbólica)", "#f3e5f5", ROXO),
    ]
    c = ""
    for x1, x2, nome, faixa, modelo, fundo, cor in faixas:
        c += f'<rect x="{x1}" y="30" width="{x2-x1}" height="66" fill="{fundo}" stroke="{cor}" stroke-width="2"/>'
        c += _t((x1 + x2) / 2, 58, nome, 16, cor, peso="bold") + _t((x1 + x2) / 2, 80, faixa, 13, CINZA)
        c += _t((x1 + x2) / 2, 124, modelo, 13.5, TINTA, peso="bold")
    c += _seta(40, 144, 750, 144, CINZA, 1.6) + _t(745, 164, "comprimento", 12.5, CINZA, "end", italico=True)
    c += _t(40, 164, "Limites usuais da literatura, para 60 Hz", 12.5, CINZA, "start", italico=True)
    return _svg(W, H, c)


def fig_pi():
    W, H = 690, 290
    yk, ym = 70, 620
    yw = 100
    c = _barra(yk, 50, 190) + _barra(ym, 50, 190)
    c += _fio(f"M{yk+4},{yw} H252 M322,{yw} H336 M434,{yw} H{ym-4}")
    c += _resistor(252, 322, yw, n=5) + _indutor(336, 434, yw, n=4)
    c += _capacitor_shunt(160, yw, 160, 214) + _capacitor_shunt(530, yw, 160, 214)
    c += f'<circle cx="160" cy="{yw}" r="3.5" fill="{TINTA}"/><circle cx="530" cy="{yw}" r="3.5" fill="{TINTA}"/>'
    c += _t(287, 80, "r_{km}", 15, VERM, peso="bold", italico=True)
    c += _t(385, 80, "j" + "x_{km}", 15, AZUL, peso="bold", italico=True)
    c += _fio("M252,52 V58 H434 V52", CINZA, 1.5)
    c += _t(343, 44, "z_{km}" + " = " + "r_{km}" + " + j" + "x_{km}", 14, CINZA, italico=True)
    c += _t(180, 166, "j" + "b_{km}^{sh}", 15, AZUL, "start", "bold", True)
    c += _t(510, 166, "j" + "b_{km}^{sh}", 15, AZUL, "end", "bold", True)
    c += _seta(88, 86, 140, 86, AZUL) + _t(112, 77, "I_{km}", 14, AZUL, peso="bold", italico=True)
    c += _seta(602, 86, 550, 86, AZUL) + _t(578, 77, "I_{mk}", 14, AZUL, peso="bold", italico=True)
    c += _t(yk, 40, "k", 17, peso="bold") + _t(ym, 40, "m", 17, peso="bold")
    c += _t(yk, 212, "E_{k}", 15, italico=True) + _t(ym, 212, "E_{m}", 15, italico=True)
    c += _t(345, 262, "b_{km}^{sh}" + " = B^{sh}/2  — metade da susceptância shunt total da linha em cada extremidade", 13.5, CINZA)
    return _svg(W, H, c)


# ======================================================================
# TRANSFORMADOR
# ======================================================================
def fig_trafo_classico():
    W, H = 700, 260
    xk, xm, yw = 70, 630, 110
    c = _barra(xk, 60, 175) + _barra(xm, 60, 175)
    c += _fio(f"M{xk+4},{yw} H187 M233,{yw} H400 M520,{yw} H{xm-4}")
    c += _trafo_ideal(210, yw, 19)
    c += f'<rect x="400" y="{yw-20}" width="120" height="40" rx="6" fill="#fff" stroke="{ROXO}" stroke-width="2.5"/>'
    c += _t(460, yw + 6, "y_{km}", 17, ROXO, peso="bold", italico=True)
    c += f'<circle cx="320" cy="{yw}" r="5" fill="{TINTA}"/>'
    c += _t(320, yw - 14, "p", 16, peso="bold", italico=True)
    c += _t(320, yw + 30, "E_{p}" + " = a·" + "E_{k}", 14, italico=True)
    c += _t(210, 76, "1 : a", 16, ROXO, peso="bold")
    c += _fio("M178,150 V158 H242 V150", CINZA, 1.5) + _t(210, 176, "transformador ideal", 12.5, CINZA)
    c += _fio("M400,150 V158 H520 V150", CINZA, 1.5)
    c += _t(460, 176, "y_{km}" + " = 1 / (" + "r_{km}" + " + j" + "x_{km}" + ")", 13, CINZA, italico=True)
    c += _t(460, 200, "dispersão e perdas no cobre", 12.5, CINZA)
    c += _seta(88, 96, 140, 96, AZUL) + _t(112, 87, "I_{km}", 14, AZUL, peso="bold", italico=True)
    c += _seta(612, 96, 560, 96, AZUL) + _t(588, 87, "I_{mk}", 14, AZUL, peso="bold", italico=True)
    c += _t(xk, 50, "k", 17, peso="bold") + _t(xm, 50, "m", 17, peso="bold")
    c += _t(xk, 197, "E_{k}", 15, italico=True) + _t(xm, 197, "E_{m}", 15, italico=True)
    c += _t(350, 238, "A barra k é o lado do tap: a relação 1 : a está do lado de k", 13.5, ROXO, peso="bold")
    return _svg(W, H, c)


def fig_trafo_pi():
    W, H = 690, 270
    xk, xm, yw = 70, 620, 90
    c = _barra(xk, 45, 175) + _barra(xm, 45, 175)
    c += _fio(f"M{xk+4},{yw} H265 M425,{yw} H{xm-4}")
    c += _caixa_adm(265, yw - 20, 160, 40, "A = a·y", ROXO, 15)
    for x, rot, anc, dx in [(160, "B = a(a − 1)·y", "start", 28), (530, "C = (1 − a)·y", "end", -28)]:
        c += _fio(f"M{x},{yw} V122 M{x},172 V198") + _terra(x, 198)
        c += f'<rect x="{x-17}" y="122" width="34" height="50" rx="5" fill="#fff" stroke="{ROXO}" stroke-width="2.5"/>'
        c += f'<circle cx="{x}" cy="{yw}" r="3.5" fill="{TINTA}"/>'
        c += _t(x + dx, 152, rot, 15, ROXO, anc, "bold")
    c += _t(xk, 34, "k (tap)", 15, peso="bold") + _t(xm, 34, "m", 17, peso="bold")
    c += _t(345, 246, "com y = " + "y_{km}" + " — as três admitâncias dependem do tap a", 13.5, CINZA)
    return _svg(W, H, c)


# ======================================================================
# MATRIZ DE ADMITÂNCIA
# ======================================================================
def fig_ybus_rede_matriz():
    W, H = 820, 350
    c = ""
    pos = {1: (80, 90), 2: (320, 90), 3: (200, 220), 4: (360, 290)}
    for a, b in [(1, 2), (1, 3), (2, 3)]:
        c += _fio(f"M{pos[a][0]},{pos[a][1]} L{pos[b][0]},{pos[b][1]}", TINTA, 2.6)
    x3, y3 = pos[3]; x4, y4 = pos[4]
    c += _fio(f"M{x3},{y3} L{x4},{y4}", ROXO, 2.6, 'stroke-dasharray="8 4"')
    mx, my = (x3 + x4) / 2, (y3 + y4) / 2
    c += f'<circle cx="{mx-8:.0f}" cy="{my-3:.0f}" r="11" fill="#fff" stroke="{ROXO}" stroke-width="2.4"/>'
    c += f'<circle cx="{mx+6:.0f}" cy="{my+3:.0f}" r="11" fill="none" stroke="{ROXO}" stroke-width="2.4"/>'
    c += _fio("M60,90 H30") + _gerador(18, 90, 13)
    c += _fio("M340,90 H380 V110") + f'<polygon points="372,108 388,108 380,126" fill="{TINTA}"/>'
    c += _fio("M380,290 H420 V308") + f'<polygon points="412,306 428,306 420,324" fill="{TINTA}"/>'
    for k, (x, y) in pos.items():
        c += f'<circle cx="{x}" cy="{y}" r="20" fill="{AZUL}" stroke="#fff" stroke-width="3"/>'
        c += _t(x, y + 6, str(k), 17, "#fff", peso="bold")
    c += _t(200, 30, "Rede exemplo: 3 linhas e 1 transformador", 14, CINZA, peso="bold")
    # Matriz
    padrao = [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 2], [0, 0, 2, 1]]
    gx, gy, cel = 560, 66, 52
    c += _t(gx - 40, gy + 2 * cel + 6, "Y_{bus} =", 20, anchor="end", peso="bold", italico=True)
    c += _fio(f"M{gx-6},{gy} H{gx-16} V{gy+4*cel} H{gx-6}", TINTA, 3)
    c += _fio(f"M{gx+4*cel+6},{gy} H{gx+4*cel+16} V{gy+4*cel} H{gx+4*cel+6}", TINTA, 3)
    for j in range(4):
        c += _t(gx + j * cel + cel / 2, gy - 12, str(j + 1), 13, CINZA, peso="bold")
        c += _t(gx + 4 * cel + 34, gy + j * cel + cel / 2 + 5, str(j + 1), 13, CINZA, peso="bold")
    for i in range(4):
        for j in range(4):
            cx, cy = gx + j * cel + cel / 2, gy + i * cel + cel / 2
            v = padrao[i][j]
            if i == j:
                c += f'<circle cx="{cx}" cy="{cy}" r="15" fill="{AZUL}"/>'
            elif v == 1:
                c += f'<circle cx="{cx}" cy="{cy}" r="15" fill="{LARANJ}"/>'
            elif v == 2:
                c += f'<circle cx="{cx}" cy="{cy}" r="15" fill="{ROXO}"/>'
            else:
                c += _t(cx, cy + 6, "0", 17, "#9e9e9e", peso="bold")
    ly = 300
    c += f'<circle cx="500" cy="{ly-5}" r="8" fill="{AZUL}"/>' + _t(514, ly, "Y_{kk}" + " (diagonal)", 13, anchor="start")
    c += f'<circle cx="640" cy="{ly-5}" r="8" fill="{LARANJ}"/>' + _t(654, ly, "linha k–m", 13, anchor="start")
    c += f'<circle cx="500" cy="{ly+21}" r="8" fill="{ROXO}"/>' + _t(514, ly + 26, "transformador k–m", 13, anchor="start")
    c += _t(640, ly + 26, "0", 15, "#9e9e9e", "middle", "bold") + _t(654, ly + 26, "sem ligação direta", 13, anchor="start")
    return _svg(W, H, c)


def fig_fluxograma_ybus():
    W, H = 600, 690
    cx, bw = 300, 260
    c = ""
    c += _terminal(cx - 130, 18, 260, 42, "Início: dados de barras e ramos")
    c += _seta(cx, 60, cx, 84)
    c += _bloco(cx - bw / 2, 86, bw, 46, ["Ybus ← matriz nula (NB × NB)"], AZUL, "#e3f0fb", 14, peso_primeira=False)
    c += _seta(cx, 132, cx, 156)
    c += _bloco(cx - bw / 2, 158, bw, 46, ["Tomar o próximo ramo k–m"], AZUL, "#e3f0fb", 14, peso_primeira=False)
    c += _seta(cx, 204, cx, 226)
    c += _losango(cx, 268, 200, 80, ["Tipo do ramo?"])
    c += _fio(f"M{cx-100},268 H130 V316") + _seta(130, 314, 130, 320)
    c += _fio(f"M{cx+100},268 H470 V316") + _seta(470, 314, 470, 320)
    c += _t(cx - 120, 258, "Linha", 13, CINZA, "end", "bold") + _t(cx + 120, 258, "Transformador", 13, CINZA, "start", "bold")
    c += _bloco(20, 322, 220, 66, ["Somar contribuições", "do modelo π", "(y e b^{sh})"], AZUL, "#e3f0fb", 13.5, peso_primeira=False)
    c += _bloco(360, 322, 220, 66, ["Somar contribuições", "do modelo 1 : a", "(a²y, y e −a·y)"], ROXO, "#f3e5f5", 13.5, peso_primeira=False)
    c += _fio(f"M130,388 V412 H{cx}") + _fio(f"M470,388 V412 H{cx}") + _seta(cx, 412, cx, 428)
    c += _losango(cx, 470, 200, 80, ["Restam ramos?"])
    c += _fio(f"M{cx-100},470 H14 V181") + _seta(14, 181, cx - bw / 2 - 2, 181)
    c += _t(cx - 112, 460, "Sim", 13, CINZA, "end", "bold")
    c += _seta(cx, 510, cx, 538) + _t(cx + 10, 528, "Não", 13, CINZA, "start", "bold")
    c += _bloco(cx - 150, 540, 300, 54, ["Somar os shunts de barra", "nos elementos diagonais"], AZUL, "#e3f0fb", 14, peso_primeira=False)
    c += _seta(cx, 594, cx, 620)
    c += _terminal(cx - 110, 622, 220, 42, "Ybus montada")
    return _svg(W, H, c)


# ======================================================================
# MÉTODO DE NEWTON-RAPHSON
# ======================================================================
def fig_newton_1d():
    W, H = 640, 360
    f = lambda x: x ** 2 - 2.0
    df = lambda x: 2.0 * x
    xmin, xmax, ymin, ymax = 0.55, 3.3, -2.3, 8.6
    X0, X1, Y0, Y1 = 60, 610, 320, 24
    px = lambda x: X0 + (x - xmin) / (xmax - xmin) * (X1 - X0)
    py = lambda y: Y0 - (y - ymin) / (ymax - ymin) * (Y0 - Y1)
    c = ""
    c += _seta(X0, py(0), X1 + 14, py(0), CINZA, 1.6) + _t(X1 + 14, py(0) + 22, "x", 15, CINZA, "end", italico=True)
    c += _seta(X0, Y0, X0, Y1 - 6, CINZA, 1.6) + _t(X0 + 10, Y1 + 4, "g(x)", 15, CINZA, "start", italico=True)
    xs = np.linspace(0.75, 3.12, 120)
    d = "M" + " L".join(f"{px(x):.1f},{py(f(x)):.1f}" for x in xs)
    c += _fio(d, AZUL, 3)
    x = 3.0
    rotulos = []
    for k in range(3):
        xn = x - f(x) / df(x)
        c += _fio(f"M{px(x):.1f},{py(f(x)):.1f} L{px(xn):.1f},{py(0):.1f}", LARANJ, 2.2)
        c += f'<circle cx="{px(x):.1f}" cy="{py(f(x)):.1f}" r="5" fill="{LARANJ}"/>'
        c += _fio(f"M{px(xn):.1f},{py(0):.1f} L{px(xn):.1f},{py(f(xn)):.1f}", CINZA, 1.4, 'stroke-dasharray="5 4"')
        rotulos.append((x, k))
        x = xn
    rotulos.append((x, 3))
    for xv, k in rotulos[:3]:
        c += f'<circle cx="{px(xv):.1f}" cy="{py(0):.1f}" r="4" fill="{TINTA}"/>'
        c += _t(px(xv), py(0) + 24, f"x^{{({k})}}", 15, italico=True)
    raiz = np.sqrt(2.0)
    c += f'<circle cx="{px(raiz):.1f}" cy="{py(0):.1f}" r="6" fill="{VERDE}"/>'
    c += _t(px(raiz) - 18, py(0) - 12, "x*", 15, VERDE, peso="bold", italico=True)
    c += _t(px(3.0) - 10, py(f(3.0)) + 4, "reta tangente em x^{(0)}", 13, LARANJ, "end", "bold")
    c += _t(330, 348, "Cada nova estimativa é a raiz da linearização (reta tangente) da anterior", 13, CINZA)
    return _svg(W, H, c)


def fig_jacobiana_blocos():
    W, H = 700, 510
    x0, y0, a, b = 240, 80, 210, 130
    c = ""
    blocos = [
        (x0, y0, a, a, "H", "∂P/∂θ", "#e3f0fb", AZUL),
        (x0 + a, y0, b, a, "N", "∂P/∂|V|", "#e8f5e9", VERDE),
        (x0, y0 + a, a, b, "M", "∂Q/∂θ", "#fff3e0", LARANJ),
        (x0 + a, y0 + a, b, b, "L", "∂Q/∂|V|", "#f3e5f5", ROXO),
    ]
    for x, y, w, h, nome, der, fundo, cor in blocos:
        c += f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fundo}" stroke="{cor}" stroke-width="2.5"/>'
        c += _t(x + w / 2, y + h / 2 - 2, nome, 30, cor, peso="bold")
        c += _t(x + w / 2, y + h / 2 + 24, der, 15, TINTA, italico=True)
    c += _t(x0 + a / 2, y0 - 34, "θ", 17, peso="bold", italico=True) + _t(x0 + a / 2, y0 - 14, "barras PV e PQ", 12.5, CINZA)
    c += _t(x0 + a + b / 2, y0 - 34, "|V|", 17, peso="bold", italico=True) + _t(x0 + a + b / 2, y0 - 14, "barras PQ", 12.5, CINZA)
    c += _t(x0 - 16, y0 + a / 2 - 4, "ΔP", 17, anchor="end", peso="bold") + _t(x0 - 16, y0 + a / 2 + 16, "barras PV e PQ", 12.5, CINZA, "end")
    c += _t(x0 - 16, y0 + a + b / 2 - 4, "ΔQ", 17, anchor="end", peso="bold") + _t(x0 - 16, y0 + a + b / 2 + 16, "barras PQ", 12.5, CINZA, "end")
    yb = y0 + a + b + 18
    c += _fio(f"M{x0},{yb} V{yb+6} H{x0+a} V{yb}", CINZA, 1.5) + _t(x0 + a / 2, yb + 26, "n_{PV}" + " + " + "n_{PQ}", 14, CINZA, italico=True)
    c += _fio(f"M{x0+a},{yb} V{yb+6} H{x0+a+b} V{yb}", CINZA, 1.5) + _t(x0 + a + b / 2, yb + 26, "n_{PQ}", 14, CINZA, italico=True)
    c += _t(W / 2 + 60, H - 12, "Dimensão de J: (" + "n_{PV}" + " + 2" + "n_{PQ}" + ") × (" + "n_{PV}" + " + 2" + "n_{PQ}" + ")", 14, TINTA, peso="bold")
    return _svg(W, H, c)


def fig_fluxograma_mnr():
    W, H = 660, 800
    cx, bw = 280, 270
    xl = cx - bw / 2
    c = ""
    c += _terminal(cx - 80, 16, 160, 40, "Início")
    c += _seta(cx, 56, cx, 80)
    c += _bloco(xl, 82, bw, 46, ["Dados da rede e montagem da Ybus"], AZUL, "#e3f0fb", 14, peso_primeira=False)
    c += _seta(cx, 128, cx, 150)
    c += _bloco(xl, 152, bw, 46, ["Estimativa inicial (ν = 0)"], AZUL, "#e3f0fb", 14, peso_primeira=False)
    c += _seta(cx, 198, cx, 220)
    c += _bloco(xl, 222, bw, 52, ["Calcular P e Q", "com o estado atual"], AZUL, "#e3f0fb", 14, peso_primeira=False)
    c += _seta(cx, 274, cx, 296)
    c += _bloco(xl, 298, bw, 52, ["Montar os resíduos", "ΔP (PV, PQ) e ΔQ (PQ)"], AZUL, "#e3f0fb", 14, peso_primeira=False)
    c += _seta(cx, 350, cx, 366)
    c += _losango(cx, 410, 220, 86, ["max |Δ| ≤ ε ?"])
    c += _seta(cx + 110, 410, 428, 410) + _t(cx + 114, 400, "Sim", 13, VERDE, "start", "bold")
    c += _bloco(430, 378, 214, 64, ["Subsistema 2", "P, Q da Slack; Q das PV;", "fluxos nos ramos"], VERDE, "#e8f5e9", 13.5)
    c += _seta(537, 442, 537, 456)
    c += _terminal(477, 458, 120, 36, "Fim")
    c += _seta(cx, 453, cx, 480) + _t(cx + 10, 472, "Não", 13, VERM, "start", "bold")
    c += _losango(cx, 522, 220, 80, ["ν &lt; ν_{máx} ?"])
    c += _seta(cx + 110, 522, 428, 522) + _t(cx + 114, 512, "Não", 13, VERM, "start", "bold")
    c += _bloco(430, 502, 214, 40, ["Parar: sem convergência"], VERM, "#ffebee", 13.5, peso_primeira=False)
    c += _seta(cx, 562, cx, 588) + _t(cx + 10, 580, "Sim", 13, VERDE, "start", "bold")
    c += _bloco(xl, 590, bw, 46, ["Montar a Jacobiana J(ν)"], ROXO, "#f3e5f5", 14, peso_primeira=False)
    c += _seta(cx, 636, cx, 658)
    c += _bloco(xl, 660, bw, 46, ["Resolver  J · Δx = [ΔP ; ΔQ]"], ROXO, "#f3e5f5", 14, peso_primeira=False)
    c += _seta(cx, 706, cx, 728)
    c += _bloco(xl, 730, bw, 56, ["Atualizar θ (PV, PQ) e |V| (PQ)", "ν ← ν + 1"], ROXO, "#f3e5f5", 14, peso_primeira=False)
    c += _fio(f"M{xl},758 H{xl-110} V248") + _seta(xl - 110, 248, xl - 2, 248)
    return _svg(W, H, c)


def fig_convergencia():
    W, H = 640, 340
    X0, X1, Y0, Y1 = 70, 600, 280, 30
    nmax = 8
    lmin, lmax = -12, 0.5
    px = lambda k: X0 + k / nmax * (X1 - X0)
    py = lambda l: Y0 - (l - lmin) / (lmax - lmin) * (Y0 - Y1)
    c = _seta(X0, Y0, X1 + 12, Y0, CINZA, 1.6) + _seta(X0, Y0, X0, Y1 - 8, CINZA, 1.6)
    c += _t(X1 + 10, Y0 + 26, "iteração ν", 13.5, CINZA, "end", italico=True)
    c += f'<text x="26" y="{(Y0+Y1)/2}" font-size="13.5" fill="{CINZA}" text-anchor="middle" font-style="italic" transform="rotate(-90 26 {(Y0+Y1)/2})">log |erro| (escala logarítmica)</text>'
    lin = [0 - 0.9 * k for k in range(nmax + 1)]
    quad = [0, -0.6, -1.6, -3.6, -7.4, -12]
    c += _fio(f"M{X0},{py(-6):.1f} H{X1}", CINZA, 1.5, 'stroke-dasharray="6 5"') + _t(X0 + 10, py(-6) - 8, "tolerância ε", 13, CINZA, "start", "bold")
    d = "M" + " L".join(f"{px(k):.1f},{py(v):.1f}" for k, v in enumerate(lin))
    c += _fio(d, LARANJ, 2.8)
    d = "M" + " L".join(f"{px(k):.1f},{py(v):.1f}" for k, v in enumerate(quad))
    c += _fio(d, AZUL, 3.2)
    for k, v in enumerate(lin):
        c += f'<circle cx="{px(k):.1f}" cy="{py(v):.1f}" r="4" fill="{LARANJ}"/>'
    for k, v in enumerate(quad):
        c += f'<circle cx="{px(k):.1f}" cy="{py(v):.1f}" r="4.5" fill="{AZUL}"/>'
    c += _t(px(0.3), py(-9.4), "convergência quadrática", 14, AZUL, "start", "bold")
    c += _t(px(0.3), py(-9.4) + 17, "(Newton-Raphson, próximo da solução)", 12.5, AZUL, "start")
    c += _t(px(5.6), py(-2.6), "convergência linear", 14, LARANJ, "middle", "bold")
    c += _t(px(5.6), py(-2.6) + 17, "(ex.: Gauss-Seidel)", 12.5, LARANJ, "middle")
    c += _t(W / 2, H - 14, "Comportamento ilustrativo — curvas qualitativas, sem dados de um sistema real", 12.5, CINZA, italico=True)
    return _svg(W, H, c)


# ------------------------------------------------------
# TEORIA SEP — Conteúdo e layout
# ------------------------------------------------------
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
IMG_NEWTON_RAPHSON = "newton_raphson.jpg"


# ======================================================================
# Utilitários de exibição
# ======================================================================
def _uri_svg(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


@st.cache_data(show_spinner=False)
def _uri_arquivo(nome: str):
    caminho = ASSETS_DIR / nome
    if not caminho.exists():
        return None
    mime = "image/png" if caminho.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(caminho.read_bytes()).decode("ascii")


@st.cache_data(show_spinner=False)
def _figuras_cache():
    """Gera todas as figuras uma única vez por sessão do servidor."""
    g = globals()
    nomes = [n for n in g if n.startswith(("fig_", "icone_")) and callable(g[n])]
    return {n: _uri_svg(g[n]()) for n in nomes}


def _figura(nome: str, legenda: str, largura_max: int = 760):
    uri = _figuras_cache()[nome]
    st.markdown(
        f'<figure class="teo-fig"><img src="{uri}" alt="{legenda}" style="max-width:{largura_max}px"/>'
        f'<figcaption>{legenda}</figcaption></figure>',
        unsafe_allow_html=True,
    )


def _refletir(perguntas):
    """Perguntas abertas, sem gabarito, para mediação do professor."""
    corpo = "**Para refletir**\n\n" + "\n".join(f"{i}. {p}" for i, p in enumerate(perguntas, 1))
    st.info(corpo, icon="💬")


def _nota(texto: str, icone: str = "📌"):
    if icone == "⚠️":
        st.warning(texto, icon=icone)
    else:
        st.info(texto, icon=icone)


CSS = """
<style>
.teo-card{background:#fff;border:2px solid #e0e0e0;border-radius:14px;padding:16px 10px 14px 10px;
  text-align:center;min-height:178px;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:10px;transition:box-shadow .2s,border-color .2s,transform .18s;}
.teo-card:hover{box-shadow:0 6px 24px rgba(25,118,210,.15);border-color:#1976d2;transform:translateY(-3px);}
.teo-card.active{border-color:#1976d2;background:#e3f0fb;box-shadow:0 4px 18px rgba(25,118,210,.18);}
.teo-card-fig{height:96px;display:flex;align-items:center;justify-content:center;}
.teo-card-fig img{max-height:96px;max-width:100%;}
.teo-card-fig img.foto{border-radius:10px;}
.teo-card-icon{font-size:52px;line-height:1;}
.teo-card-title{font-size:13px;font-weight:800;color:#111;margin:0;line-height:1.25;
  text-transform:uppercase;letter-spacing:.3px;}
figure.teo-fig{margin:1.4rem auto 1.6rem auto;text-align:center;}
figure.teo-fig img{width:100%;height:auto;border-radius:14px;border:1px solid #e0e0e0;background:#fff;}
figure.teo-fig figcaption{font-size:13px;color:#757575;margin-top:8px;line-height:1.4;}
.teo-historia img{width:100%;border-radius:14px;}
</style>
"""

TOPICOS = [
    ("intro", "Introdução", "icone_sin", None),
    ("linha", "Linha de Transmissão", "icone_pi", None),
    ("trafo", "Modelo de Transformador", "icone_trafo", None),
    ("ybus", "Matriz de Admitância", "icone_ybus", None),
    ("mnr", "MNR", None, IMG_NEWTON_RAPHSON),
]


def _html_card(titulo, icone_svg, arquivo, ativo):
    if arquivo:
        uri = _uri_arquivo(arquivo)
        midia = (f'<img class="foto" src="{uri}" alt="Caricatura de Isaac Newton e Joseph Raphson"/>'
                 if uri else '<div class="teo-card-icon">📐</div>')
    else:
        midia = f'<img src="{_figuras_cache()[icone_svg]}" alt="{titulo}"/>'
    classe = "teo-card active" if ativo else "teo-card"
    return (f'<div class="{classe}"><div class="teo-card-fig">{midia}</div>'
            f'<div class="teo-card-title">{titulo}</div></div>')


# ======================================================================
# Página
# ======================================================================
def render_teoria():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## 📖 Teoria dos Sistemas Elétricos de Potência")
    st.caption("Selecione um tópico abaixo para estudar.")

    if "teoria_topico" not in st.session_state:
        st.session_state.teoria_topico = None

    cols = st.columns(len(TOPICOS), gap="small")
    for col, (chave, titulo, icone, arquivo) in zip(cols, TOPICOS):
        ativo = st.session_state.teoria_topico == chave
        col.markdown(_html_card(titulo, icone, arquivo, ativo), unsafe_allow_html=True)
        if col.button(f"Ver {titulo}", key=f"teo_{chave}", use_container_width=True,
                      type="primary" if ativo else "secondary"):
            st.session_state.teoria_topico = chave
            st.rerun()

    topico = st.session_state.teoria_topico
    if topico is None:
        st.info("👆 Clique em um dos tópicos acima para exibir o conteúdo aqui.", icon="📌")
        return

    with st.container(border=True):
        {"intro": _intro, "linha": _linha, "trafo": _trafo, "ybus": _ybus, "mnr": _mnr}[topico]()


# ======================================================================
# INTRODUÇÃO
# ======================================================================
def _intro():
    st.markdown("### 💡 Introdução")
    st.markdown(r"""
O crescimento dos sistemas elétricos de potência (SEP) tornou indispensável o uso de
ferramentas matemáticas capazes de analisar o comportamento das redes em **regime
permanente**. Entre elas, o estudo do **fluxo de potência** (ou fluxo de carga) ocupa papel
central: ele determina as magnitudes e os ângulos das tensões nas barras e, a partir
deles, os fluxos de potência ativa e reativa nos ramos e as perdas da rede.

O cálculo do fluxo de carga consiste em determinar o **estado da rede** (tensões
complexas nas barras), a distribuição dos fluxos e outras grandezas de interesse. A
modelagem é **estática**: a rede é representada por um conjunto de equações e inequações
algébricas, e não por equações diferenciais.
""")

    st.markdown("#### Estrutura de um sistema elétrico de potência")
    st.markdown(r"""
A energia é produzida em usinas, tem a tensão elevada em subestações para reduzir
as perdas no transporte, percorre longas distâncias pela rede de transmissão e é rebaixada
para chegar aos consumidores pela distribuição. No Brasil, as instalações de transmissão
com tensão igual ou superior a 230 kV compõem a **Rede Básica**.
""")
    _figura("fig_estrutura_sep", "Figura 1 — Cadeia geração–transmissão–distribuição e o escopo usual dos estudos de fluxo de potência.")

    st.markdown("#### O Sistema Interligado Nacional (SIN)")
    st.markdown(r"""
O sistema de produção e transmissão de energia elétrica do Brasil é um sistema
**hidro-termo-eólico-solar de grande porte**, com forte predominância de usinas
hidrelétricas e participação crescente das fontes eólica e solar, concentradas
sobretudo no Nordeste. Ele é dividido em quatro **subsistemas** — Sul, Sudeste/Centro-Oeste,
Nordeste e Norte — conectados por **interligações** de grande capacidade.

As interligações permitem transferir energia entre regiões e aproveitar a
**diversidade hidrológica** entre bacias: quando uma região atravessa o período seco, outra
pode exportar energia. Grandes usinas distantes dos centros de carga, como as do rio Madeira
e Belo Monte, conectam-se ao Sudeste por **elos de corrente contínua em alta tensão (HVDC)**.
A operação coordenada do SIN é responsabilidade do **Operador Nacional do Sistema Elétrico
(ONS)**, que usa estudos de fluxo de potência rotineiramente. Parte da região amazônica
ainda é atendida por **sistemas isolados**.
""")
    _figura("fig_sin", "Figura 2 — Representação esquemática dos subsistemas do SIN, das interligações CA e dos elos HVDC (fora de escala).")

    st.markdown("#### Para que serve o fluxo de potência")
    st.markdown(r"""
O fluxo de potência é a base de praticamente todos os estudos de SEP:

- **Planejamento da expansão**: avaliar se a rede futura atende à carga prevista;
- **Planejamento e programação da operação**: definir despachos e perfis de tensão;
- **Análise de contingências**: verificar o efeito da saída de linhas, transformadores ou geradores;
- **Ponto de partida** para estudos de curto-circuito, estabilidade e confiabilidade.
""")

    st.markdown("#### Hipóteses de modelagem")
    st.markdown(r"""
- **Regime permanente senoidal**, com frequência constante;
- **Sistema trifásico equilibrado**, representado pelo seu equivalente monofásico (sequência positiva);
- Grandezas expressas no **sistema por unidade (pu)**;
- Modelagem **estática**: equações algébricas não lineares.
""")

    st.markdown("#### Sistema por unidade (pu)")
    st.markdown(r"""
Cada grandeza é dividida por um valor de base. Escolhem-se uma potência de base
$S_{base}$, única para todo o sistema, e uma tensão de base $V_{base}$ para cada nível de
tensão; as demais bases decorrem delas:
""")
    st.latex(r"x_{pu} = \frac{x_{real}}{x_{base}} \qquad Z_{base} = \frac{V_{base}^2}{S_{base}} \qquad Y_{base} = \frac{1}{Z_{base}}")
    st.markdown(r"""
Em pu, transformadores com tensões de base escolhidas segundo suas relações nominais
aparecem com relação 1 : 1, e os valores de impedância ficam em faixas semelhantes para
equipamentos de portes muito diferentes. Na página de simulação, a base padrão é
$S_{base} = 100$ MVA.
""")

    st.markdown("#### Variáveis e classificação das barras")
    st.markdown(r"""
A cada barra $k$ associam-se quatro variáveis: magnitude da tensão $V_k$, ângulo
$\theta_k$, injeção líquida de potência ativa $P_k$ e injeção líquida de potência reativa
$Q_k$ (geração menos carga). Duas delas são especificadas e duas são calculadas, o que
define três tipos de barra:
""")
    _figura("fig_tipos_barra", "Figura 3 — Tipos de barra: grandezas especificadas e calculadas.")
    st.markdown(r"""
- **Slack (Vθ, referência)**: fornece a **referência angular** do sistema e fecha o balanço de
  potência. Como as perdas só são conhecidas após a solução, alguma barra precisa ter $P$ livre.
- **PV**: representa geradores com controle de tensão — a injeção de reativos se ajusta para
  manter $V$ no valor especificado.
- **PQ**: representa barras de carga (ou de passagem, com $P = Q = 0$).
""")

    st.markdown("#### Estrutura do problema")
    st.markdown(r"""
O problema pode ser dividido em dois subsistemas. O **subsistema 1** reúne as equações cujas
incógnitas são o estado ainda desconhecido; é não linear e exige um método iterativo. Uma vez
conhecido o estado, o **subsistema 2** é resolvido por cálculo direto. Em formulações mais
completas, acrescentam-se **inequações** que representam limites operacionais, como os
limites de potência reativa dos geradores.
""")
    _figura("fig_subsistemas", "Figura 4 — Decomposição do problema de fluxo de potência em dois subsistemas.")

    _refletir([
        "Por que não é possível especificar $P$ em todas as barras de um sistema real?",
        "O que aconteceria com o problema se duas barras fossem definidas como Slack? E se nenhuma fosse?",
        "Uma barra PV pode deixar de conseguir manter sua tensão especificada? Em que situação?",
        "Que vantagens e riscos operacionais as interligações entre os subsistemas do SIN trazem?",
    ])


# ======================================================================
# LINHA DE TRANSMISSÃO
# ======================================================================
def _linha():
    st.markdown(r"### 🔌 Modelo de Linha de Transmissão Tipo $\pi$")
    st.markdown(r"""
As linhas de transmissão têm resistência, indutância, capacitância e condutância
**distribuídas** ao longo de toda a extensão. Cada trecho elementar $\Delta x$ contribui com
uma pequena parcela de cada parâmetro.
""")
    _figura("fig_linha_distribuida", "Figura 1 — Linha representada por parâmetros distribuídos (cadeia de trechos elementares).")

    st.markdown("#### Parâmetros da linha")
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown(r"**Resistência série** $r$  \nPerdas ôhmicas nos condutores")
    with c2.container(border=True):
        st.markdown(r"**Reatância série** $x$  \nEfeito do campo magnético")
    with c3.container(border=True):
        st.markdown(r"**Susceptância shunt** $B^{sh}$  \nEfeito do campo elétrico")
    st.markdown(r"""
- A **resistência** depende do material, da seção e da temperatura do condutor.
- A **reatância série** $x = \omega L$ resulta do fluxo magnético concatenado com os condutores.
- A **susceptância shunt** $B^{sh} = \omega C$ resulta da capacitância entre condutores e entre
  condutores e terra.
- A **condutância shunt** (correntes de fuga em isoladores e efeito corona) costuma ser desprezada
  em estudos de fluxo de potência.

Em linhas de transmissão, em geral $x \gg r$. Essa característica tem consequências importantes
para o acoplamento entre as grandezas da rede, discutido no tópico do MNR.
""")

    st.markdown("#### Escolha do modelo pelo comprimento")
    _figura("fig_classificacao_linha", "Figura 2 — Classificação usual das linhas pelo comprimento e modelo correspondente.")
    st.markdown(r"""
Para linhas longas, o modelo $\pi$ nominal perde precisão; os parâmetros do circuito $\pi$
passam a ser corrigidos por funções hiperbólicas do comprimento (**$\pi$ equivalente**). A
**topologia** do circuito, porém, é a mesma — por isso o modelo $\pi$ é o padrão em fluxo de
potência.
""")

    st.markdown(r"#### Modelo $\pi$")
    _figura("fig_pi", r"Figura 3 — Modelo π de uma linha entre as barras k e m.")
    st.markdown("A impedância série e a admitância série do ramo são:")
    st.latex(r"z_{km} = r_{km} + jx_{km}")
    st.latex(r"y_{km} = \frac{1}{z_{km}} = g_{km} + jb_{km} \quad \text{com} \quad g_{km} = \frac{r_{km}}{r_{km}^2 + x_{km}^2}, \quad b_{km} = \frac{-x_{km}}{r_{km}^2 + x_{km}^2}")
    _nota(r"**Notação.** Letras minúsculas ($g_{km}$, $b_{km}$) indicam parâmetros do **ramo**; "
          r"maiúsculas ($G_{km}$, $B_{km}$) indicam elementos da **matriz** $Y_{bus}$. "
          r"$B^{sh}$ é a susceptância shunt **total** da linha (campo *Bsh_linha* da página de simulação) "
          r"e $b^{sh}_{km} = B^{sh}/2$ é a parcela em cada extremidade. Para uma linha, $b_{km} < 0$ e $b^{sh}_{km} > 0$.")

    st.markdown(r"#### Correntes no modelo $\pi$")
    st.markdown(r"A corrente que sai de cada barra pelo ramo tem uma componente série e uma componente shunt:")
    st.latex(r"I_{km} = y_{km}(E_k - E_m) + jb^{sh}_{km}\,E_k")
    st.latex(r"I_{mk} = y_{km}(E_m - E_k) + jb^{sh}_{km}\,E_m")
    st.markdown(r"onde $E_k = V_k e^{j\theta_k}$ e $E_m = V_m e^{j\theta_m}$ são as tensões complexas terminais.")

    st.markdown("#### Fluxos de potência no ramo")
    st.markdown(r"A partir de $S_{km}^* = E_k^* I_{km}$, com $\theta_{km} = \theta_k - \theta_m$:")
    st.latex(r"P_{km} = V_k^2\,g_{km} - V_kV_m\,g_{km}\cos\theta_{km} - V_kV_m\,b_{km}\sin\theta_{km}")
    st.latex(r"Q_{km} = -V_k^2\,(b_{km} + b^{sh}_{km}) + V_kV_m\,b_{km}\cos\theta_{km} - V_kV_m\,g_{km}\sin\theta_{km}")
    st.markdown(r"Os fluxos $P_{mk}$ e $Q_{mk}$ têm a mesma forma, trocando-se os índices. Somando os dois sentidos obtêm-se as perdas:")
    st.latex(r"P_{km} + P_{mk} = g_{km}\left(V_k^2 + V_m^2 - 2V_kV_m\cos\theta_{km}\right) = g_{km}\,|E_k - E_m|^2")
    st.latex(r"Q_{km} + Q_{mk} = -b^{sh}_{km}\left(V_k^2 + V_m^2\right) - b_{km}\,|E_k - E_m|^2")
    st.markdown(r"""
A primeira expressão representa as perdas ôhmicas na resistência série. Na segunda, o termo
série corresponde à potência reativa **absorvida** pela reatância da linha, e o termo shunt à
potência reativa **gerada** pelo efeito capacitivo.
""")

    st.markdown("#### Contribuições na $Y_{bus}$")
    st.latex(r"Y_{kk} \mathrel{+}= y_{km} + jb^{sh}_{km} \qquad Y_{mm} \mathrel{+}= y_{km} + jb^{sh}_{km} \qquad Y_{km} = Y_{mk} = -y_{km}")

    _refletir([
        "Observe as expressões de $P_{km}$ e $Q_{km}$ com $r_{km} \\approx 0$: qual grandeza da rede governa principalmente o fluxo de potência ativa? E o de reativa?",
        "Em uma linha longa operando a vazio, o que se espera da tensão na extremidade aberta? Qual parâmetro do modelo explica esse comportamento?",
        "Por que o efeito capacitivo das linhas fica mais relevante em tensões mais altas?",
        "Que erro se comete ao representar uma linha longa pelo modelo $\\pi$ nominal?",
    ])


# ======================================================================
# TRANSFORMADOR
# ======================================================================
def _trafo():
    st.markdown(r"### 🔀 Modelo do Transformador em-Fase")
    st.markdown(r"""
Os transformadores conectam redes de níveis de tensão diferentes e, quando dotados de
**comutação de tap**, também controlam magnitudes de tensão. No transformador **em-fase**, a
relação de transformação é um número **real** positivo: as tensões dos dois lados ficam em fase
na parte ideal do modelo. Os transformadores **defasadores**, cuja relação é complexa e que
controlam o fluxo de potência ativa, não fazem parte deste estudo.
""")

    st.markdown("#### Modelo clássico")
    st.markdown(r"""
O transformador é representado por um **transformador ideal** com relação $1 : a$ em série com a
admitância $y_{km}$, que reúne a reatância de dispersão e a resistência dos enrolamentos. O ramo
magnetizante costuma ser desprezado em fluxo de potência.
""")
    _figura("fig_trafo_classico", "Figura 1 — Modelo clássico do transformador em-fase: transformador ideal 1 : a em série com y_km.")
    st.markdown(r"""
No nó intermediário $p$, a tensão é $E_p = a\,E_k$. Como o transformador ideal não consome
potência ($E_k I_{km}^* + E_p I_{pk}^* = 0$), a corrente do lado $k$ é $a$ vezes a corrente que
atravessa $y_{km}$. Resultam as correntes terminais:
""")
    st.latex(r"I_{km} = a^2\,y_{km}\,E_k - a\,y_{km}\,E_m")
    st.latex(r"I_{mk} = -a\,y_{km}\,E_k + y_{km}\,E_m")

    st.markdown(r"#### Circuito $\pi$ equivalente")
    st.markdown(r"""
Identificando os coeficientes das equações acima com os de um circuito $\pi$ genérico,
$I_{km} = A(E_k - E_m) + B\,E_k$ e $I_{mk} = A(E_m - E_k) + C\,E_m$, obtêm-se:
""")
    st.latex(r"A = a\,y_{km} \qquad B = a(a-1)\,y_{km} \qquad C = (1-a)\,y_{km}")
    _figura("fig_trafo_pi", "Figura 2 — Circuito π equivalente do transformador em-fase.")

    st.markdown("#### Contribuições na $Y_{bus}$")
    st.latex(r"Y_{kk} \mathrel{+}= a^2\,y_{km} \qquad Y_{mm} \mathrel{+}= y_{km} \qquad Y_{km} = Y_{mk} = -a\,y_{km}")

    st.markdown("#### Efeito do tap $a$")
    st.markdown(r"""
Como $y_{km}$ é predominantemente indutiva, um múltiplo **positivo** de $y_{km}$ comporta-se como
um elemento indutivo (absorve reativos) e um múltiplo **negativo**, como capacitivo (fornece
reativos).

| Valor de $a$ | Shunt $B$ (lado $k$) | Shunt $C$ (lado $m$) | Tendência |
|:---:|:---:|:---:|:---|
| $a = 1$ | $0$ | $0$ | Reduz-se à admitância série $y_{km}$ |
| $a > 1$ | Indutivo | Capacitivo | **Eleva** $V_m$ e **reduz** $V_k$ |
| $a < 1$ | Capacitivo | Indutivo | **Reduz** $V_m$ e **eleva** $V_k$ |

O resultado é coerente com o modelo clássico: com $a > 1$, a tensão $E_p = a\,E_k$ "vista" pelo
lado $m$ é maior que $E_k$. Comutadores de tap sob carga operam em faixas típicas da ordem de
$\pm 10\%$ em torno da relação nominal.
""")
    _nota(r"**A ordem de conexão importa.** O modelo é assimétrico: $Y_{kk} \neq Y_{mm}$ quando $a \neq 1$, "
          r"pois a barra $k$ é o lado do tap. A matriz $Y_{bus}$, porém, continua **simétrica** ($Y_{km} = Y_{mk}$). "
          r"Na página de simulação, a coluna *De (lado tap)* corresponde à barra $k$.", "⚠️")

    _refletir([
        "Por que a posição do tap ($k$ ou $m$) altera a $Y_{bus}$, mas não a sua simetria?",
        "Se uma barra de carga apresenta subtensão, em que sentido o tap de um transformador conectado a ela deveria ser ajustado? Justifique pelo circuito $\\pi$.",
        "O tap altera a potência reativa que circula pelo transformador. De onde vem essa potência?",
        "Em que situação seria inadequado desprezar o ramo magnetizante?",
    ])


# ======================================================================
# MATRIZ DE ADMITÂNCIA NODAL
# ======================================================================
def _ybus():
    st.markdown(r"### 📊 Matriz de Admitância Nodal ($Y_{bus}$)")
    st.markdown(r"""
A representação nodal relaciona as **injeções de corrente** nas barras com as **tensões
nodais**. Aplicando a Lei de Kirchhoff das correntes a cada barra, a injeção $I_k$ é a soma das
correntes que saem pelos ramos e pelos elementos shunt conectados a ela:
""")
    st.latex(r"I_k = jb^{sh}_k\,E_k + \sum_{m \in \Omega_k} I_{km} \quad\Longrightarrow\quad I_k = \sum_{m=1}^{NB} Y_{km}\,E_m \quad\Longrightarrow\quad \mathbf{I} = \mathbf{Y}_{bus}\,\mathbf{E}")
    st.markdown(r"onde $\Omega_k$ é o conjunto das barras vizinhas de $k$ e $b^{sh}_k$ é a susceptância shunt conectada diretamente à barra.")

    st.markdown("#### Estrutura da matriz")
    _figura("fig_ybus_rede_matriz", "Figura 1 — Rede de 4 barras e o padrão de elementos não nulos da sua Y_bus.")
    st.markdown(r"""
Um elemento fora da diagonal $Y_{km}$ só é não nulo se existir um ramo ligando diretamente as
barras $k$ e $m$. A diagonal $Y_{kk}$ acumula tudo o que está conectado à barra $k$.
""")

    st.markdown("#### Regra geral de montagem")
    st.markdown(r"Linhas e transformadores em-fase podem ser tratados por uma única expressão:")
    st.latex(r"Y_{km} = -a_{km}\,y_{km} \qquad Y_{kk} = jb^{sh}_k + \sum_{m \in \Omega_k}\left(jb^{sh}_{km} + a_{km}^2\,y_{km}\right)")
    st.markdown(r"""
com $a_{km} = 1$ para linhas e para o lado sem tap do transformador, e $b^{sh}_{km} = 0$ para
transformadores. Separando parte real e imaginária, escreve-se $Y_{km} = G_{km} + jB_{km}$ —
são esses $G$ e $B$ que aparecem nas equações de potência do MNR.
""")

    st.markdown("#### Contribuições por tipo de elemento")
    st.markdown(r"""
| Elemento | $Y_{kk}$ | $Y_{mm}$ | $Y_{km} = Y_{mk}$ |
|:---|:---:|:---:|:---:|
| Linha de transmissão | $y_{km} + jb^{sh}_{km}$ | $y_{km} + jb^{sh}_{km}$ | $-y_{km}$ |
| Transformador ($1 : a$, tap em $k$) | $a^2\,y_{km}$ | $y_{km}$ | $-a\,y_{km}$ |
| Shunt de barra | $+jb^{sh}_k$ | — | — |
""")

    st.markdown("#### Algoritmo de montagem")
    _figura("fig_fluxograma_ybus", "Figura 2 — Fluxograma da montagem da Y_bus ramo a ramo.", 600)

    st.markdown("#### Propriedades")
    st.markdown(r"""
- **Simétrica** em redes com linhas e transformadores em-fase: $Y_{km} = Y_{mk}$. Transformadores
  defasadores quebram essa simetria.
- **Esparsa**: cada barra se liga a poucas outras, então a fração de elementos nulos cresce com o
  tamanho da rede. Programas profissionais armazenam e operam apenas os elementos não nulos.
- **Singular** quando a rede não tem nenhuma ligação à terra (sem shunts): nesse caso, a soma dos
  elementos de cada linha da matriz é nula. Como o fluxo de potência usa a $Y_{bus}$ diretamente,
  sem invertê-la, isso não impede a solução.
- **Sinais típicos**: em ramos predominantemente indutivos, $B_{km} > 0$ fora da diagonal e
  $B_{kk} < 0$ na diagonal.
- **Atualização local**: a entrada ou saída de um ramo altera apenas quatro elementos da matriz.
""")
    _nota(r"Uma rede com uma **ilha** sem barra de referência não tem solução bem definida para os ângulos: "
          r"nesse caso, quem se torna singular é a **Jacobiana** do MNR, não necessariamente a $Y_{bus}$.")

    _refletir([
        "Por que a $Y_{bus}$ é esparsa? Que impacto isso tem no tempo de cálculo e na memória para sistemas com milhares de barras?",
        "Como fica a $Y_{bus}$ após a saída de operação de uma linha? Quais elementos mudam?",
        "Duas linhas em paralelo entre as mesmas barras: como elas aparecem na matriz?",
        "Que informação física carrega a soma dos elementos de uma linha da $Y_{bus}$?",
    ])


# ======================================================================
# MÉTODO DE NEWTON-RAPHSON
# ======================================================================
def _mnr():
    st.markdown("### 📐 Método de Newton-Raphson (MNR)")

    col_img, col_txt = st.columns([2, 3], gap="large")
    with col_img:
        uri = _uri_arquivo(IMG_NEWTON_RAPHSON)
        if uri:
            st.markdown(f'<div class="teo-historia"><img src="{uri}" alt="Caricatura de Isaac Newton e Joseph Raphson"/></div>',
                        unsafe_allow_html=True)
            st.caption("Isaac Newton e Joseph Raphson (caricatura).")
    with col_txt:
        st.markdown("#### Um pouco de história")
        st.markdown(r"""
**Isaac Newton** descreveu, por volta de 1669, um procedimento para aproximar raízes de
polinômios por correções sucessivas. Em 1690, **Joseph Raphson** publicou uma formulação mais
simples e direta do mesmo procedimento, próxima da forma usada hoje — por isso o método leva o
nome de ambos.

Em sistemas de potência, o método se consolidou na década de 1960, quando técnicas de
**matrizes esparsas** tornaram viável aplicá-lo a redes de grande porte (Tinney e Hart, 1967).
Desde então, é o método de referência dos programas de fluxo de potência.
""")

    st.markdown("#### 1. A ideia: linearizar e corrigir")
    st.markdown(r"""
Considere uma equação de uma variável $g(x) = 0$. A partir de uma estimativa $x^{(\nu)}$,
aproxima-se $g$ pela sua **reta tangente** (série de Taylor truncada no termo de primeira ordem)
e toma-se a raiz dessa reta como nova estimativa:
""")
    st.latex(r"g\left(x^{(\nu)} + \Delta x\right) \approx g\left(x^{(\nu)}\right) + g'\left(x^{(\nu)}\right)\Delta x = 0 \quad\Longrightarrow\quad x^{(\nu+1)} = x^{(\nu)} - \frac{g\left(x^{(\nu)}\right)}{g'\left(x^{(\nu)}\right)}")
    _figura("fig_newton_1d", "Figura 1 — Interpretação geométrica do método de Newton-Raphson para uma variável.", 680)
    st.markdown(r"""
Para um sistema de $n$ equações $\mathbf{g}(\mathbf{x}) = \mathbf{0}$, a derivada é substituída
pela **matriz Jacobiana** $\mathbf{J} = \partial\mathbf{g}/\partial\mathbf{x}$, e a divisão, pela
solução de um sistema linear:
""")
    st.latex(r"\mathbf{J}\left(\mathbf{x}^{(\nu)}\right)\Delta\mathbf{x}^{(\nu)} = -\mathbf{g}\left(\mathbf{x}^{(\nu)}\right) \qquad \mathbf{x}^{(\nu+1)} = \mathbf{x}^{(\nu)} + \Delta\mathbf{x}^{(\nu)}")

    st.markdown("#### 2. Equações de potência")
    st.markdown(r"""
A potência complexa injetada na barra $k$ é $S_k = P_k + jQ_k = E_k I_k^*$. Substituindo
$I_k = \sum_m Y_{km} E_m$, com $Y_{km} = G_{km} + jB_{km}$ e $\theta_{km} = \theta_k - \theta_m$:
""")
    st.latex(r"P_k = V_k \sum_{m=1}^{NB} V_m\left(G_{km}\cos\theta_{km} + B_{km}\sin\theta_{km}\right)")
    st.latex(r"Q_k = V_k \sum_{m=1}^{NB} V_m\left(G_{km}\sin\theta_{km} - B_{km}\cos\theta_{km}\right)")
    st.markdown(r"""
Como as variáveis da Slack são conhecidas e o $Q$ das barras PV é incógnita, o subsistema 1
tem $n_{PV} + 2\,n_{PQ}$ equações e o mesmo número de incógnitas.
""")

    st.markdown("#### 3. Vetor de resíduos (mismatch)")
    st.latex(r"\mathbf{f} = \begin{bmatrix} \Delta\mathbf{P} \\ \Delta\mathbf{Q} \end{bmatrix} = \begin{bmatrix} \mathbf{P}^{esp} - \mathbf{P}^{calc}(\mathbf{V},\boldsymbol{\theta}) \\ \mathbf{Q}^{esp} - \mathbf{Q}^{calc}(\mathbf{V},\boldsymbol{\theta}) \end{bmatrix}")
    st.markdown(r"""
- $\Delta P_k$: barras **PV e PQ** (a Slack é excluída, pois seu $P$ é incógnita);
- $\Delta Q_k$: apenas barras **PQ** (Slack e PV excluídas, pois seu $Q$ é incógnita).
""")

    st.markdown("#### 4. Matriz Jacobiana")
    st.markdown(r"""
A Jacobiana agrupa as derivadas das potências calculadas em relação às variáveis de estado.
Como $\mathbf{f} = \text{esp} - \text{calc}$, o sinal negativo da forma geral é absorvido e o
sistema de correção fica $\mathbf{J}\,\Delta\mathbf{x} = \mathbf{f}$.
""")
    _figura("fig_jacobiana_blocos", "Figura 2 — Estrutura em blocos da Jacobiana: linhas = resíduos, colunas = variáveis de estado.", 640)
    st.latex(r"\begin{bmatrix} \Delta\mathbf{P} \\ \Delta\mathbf{Q} \end{bmatrix} = \begin{bmatrix} \mathbf{H} & \mathbf{N} \\ \mathbf{M} & \mathbf{L} \end{bmatrix} \begin{bmatrix} \Delta\boldsymbol{\theta} \\ \Delta\mathbf{V} \end{bmatrix} \qquad \mathbf{H} = \frac{\partial\mathbf{P}}{\partial\boldsymbol{\theta}},\; \mathbf{N} = \frac{\partial\mathbf{P}}{\partial\mathbf{V}},\; \mathbf{M} = \frac{\partial\mathbf{Q}}{\partial\boldsymbol{\theta}},\; \mathbf{L} = \frac{\partial\mathbf{Q}}{\partial\mathbf{V}}")

    st.markdown(r"**Termos fora da diagonal** ($k \neq m$):")
    st.latex(r"H_{km} = V_kV_m\left(G_{km}\sin\theta_{km} - B_{km}\cos\theta_{km}\right) \qquad N_{km} = V_k\left(G_{km}\cos\theta_{km} + B_{km}\sin\theta_{km}\right)")
    st.latex(r"M_{km} = -V_kV_m\left(G_{km}\cos\theta_{km} + B_{km}\sin\theta_{km}\right) \qquad L_{km} = V_k\left(G_{km}\sin\theta_{km} - B_{km}\cos\theta_{km}\right)")
    st.markdown(r"**Termos diagonais**, em forma fechada usando $P_k^{calc}$ e $Q_k^{calc}$:")
    st.latex(r"H_{kk} = -Q_k^{calc} - V_k^2\,B_{kk} \qquad N_{kk} = \frac{P_k^{calc} + V_k^2\,G_{kk}}{V_k}")
    st.latex(r"M_{kk} = P_k^{calc} - V_k^2\,G_{kk} \qquad L_{kk} = \frac{Q_k^{calc} - V_k^2\,B_{kk}}{V_k}")
    st.markdown(r"""
Os termos fora da diagonal são nulos sempre que $Y_{km} = 0$: a Jacobiana herda a **esparsidade**
da $Y_{bus}$. Ela também muda a cada iteração, pois depende do estado atual.
""")

    st.markdown("#### 5. Correção e atualização do estado")
    st.latex(r"\boldsymbol{\theta}^{(\nu+1)} = \boldsymbol{\theta}^{(\nu)} + \Delta\boldsymbol{\theta}^{(\nu)} \qquad \mathbf{V}^{(\nu+1)} = \mathbf{V}^{(\nu)} + \Delta\mathbf{V}^{(\nu)}")
    st.markdown(r"""
A Slack mantém $V$ e $\theta$ fixos. As barras PV mantêm $V$ fixo e só têm $\theta$ atualizado.
""")

    st.markdown("#### 6. Critério de convergência")
    st.latex(r"\max\left\{\,|\Delta P_k|,\; |\Delta Q_k|\,\right\} \leq \varepsilon")
    st.markdown(r"Tolerâncias típicas: $\varepsilon = 10^{-4}$ a $10^{-6}$ pu. O teste é feito sobre os **resíduos de potência**, e não sobre as correções de estado.")

    st.markdown("#### 7. Fluxograma do algoritmo")
    _figura("fig_fluxograma_mnr", "Figura 3 — Fluxograma do fluxo de potência pelo Método de Newton-Raphson.", 620)

    st.markdown("#### 8. Características de convergência")
    _figura("fig_convergencia", "Figura 4 — Comparação qualitativa entre convergência linear e quadrática.", 660)
    st.markdown(r"""
- Próximo da solução, o erro de uma iteração é aproximadamente proporcional ao **quadrado** do
  erro anterior (**convergência quadrática**): o número de casas decimais corretas tende a dobrar
  a cada iteração.
- O número de iterações depende pouco do tamanho da rede.
- A convergência depende da **estimativa inicial**. Em SEP, o **perfil plano** ($V = 1{,}0$ pu nas
  barras PQ e $\theta = 0$, com as tensões especificadas de Slack e PV) costuma estar suficientemente
  próximo da solução.
- Em sistemas muito carregados, próximos do limite de transmissão, a Jacobiana tende à
  singularidade e o método pode divergir.
""")

    st.markdown("#### 9. Após a convergência")
    st.markdown(r"""
Com o estado conhecido, resolve-se o subsistema 2: calculam-se $P$ e $Q$ da Slack e $Q$ das
barras PV pelas equações de potência, e os fluxos e perdas em cada ramo pelas expressões do
modelo $\pi$ (linhas) e do modelo $1 : a$ (transformadores).
""")

    st.markdown("#### 10. Variantes do método")
    st.markdown(r"""
- **Newton desacoplado**: explora o fraco acoplamento entre $P$ e $V$ e entre $Q$ e $\theta$
  (consequência de $x \gg r$), desprezando os blocos $\mathbf{N}$ e $\mathbf{M}$.
- **Desacoplado rápido**: aproxima as matrizes desacopladas por matrizes constantes, montadas uma
  única vez.
- **Fluxo de carga linearizado (CC)**: considera apenas a relação entre $P$ e $\theta$, com
  tensões em 1 pu — útil em estudos de planejamento e de contingências em grande número.
""")

    _refletir([
        "Por que a estimativa inicial plana costuma funcionar bem em SEP? Em que situação ela pode falhar?",
        "Por que o $Q$ das barras PV não aparece no vetor de resíduos? Onde ele é calculado?",
        "Recalcular a Jacobiana a cada iteração tem custo. O que se ganha e o que se perde ao mantê-la fixa por algumas iterações?",
        "Observe a Figura 2: por que o bloco $\\mathbf{H}$ é sempre quadrado, mas $\\mathbf{N}$ e $\\mathbf{M}$ podem não ser?",
        "Analisando o histórico de convergência de uma simulação, como identificar que o método está na fase de convergência quadrática?",
    ])


if st.session_state.pagina == "teoria":
    render_teoria()
    st.stop()

# ======================================================
# PÁGINA: SIMULAÇÃO FC
# ======================================================
if st.session_state.pagina == "simulacao":
    st.markdown("## 🖥️ Simulação de Fluxo de Carga")

    # Parâmetros do cálculo — dentro da página de simulação, em expander
    with st.expander("⚙️ Parâmetros do Cálculo e Base do Sistema", expanded=False):
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            tol_input = st.number_input("Tolerância (Erro Máximo)", value=1e-6, format="%e", step=1e-7)
        with col_p2:
            max_iter_input = st.number_input("Máximo de Iterações", value=20, min_value=1, step=1)
        with col_p3:
            base_mva = st.number_input("Base (MVA)", value=100.0, step=10.0)
        with col_p4:
            unidade = st.selectbox("Unidade de Potência", ["MW / MVar", "p.u."])

    divisor_potencia = base_mva if unidade == "MW / MVar" else 1.0

    modo_entrada = st.radio(
        "Escolha a interface de modelagem do sistema:",
        ["Modo Tabela (Entrada Analítica)", "Modo Circuito (Diagrama Unifilar)"],
        horizontal=True
    )
    st.markdown("---")
    dados_para_calculo = None

    # ============================================================
    # MODO 1 — TABELA
    # ============================================================
    if modo_entrada == "Modo Tabela (Entrada Analítica)":
        st.markdown("Insira os parâmetros elétricos dos barramentos, linhas e transformadores. **Dica:** Os dados do Modo Circuito são sincronizados aqui automaticamente.")

        canvas_data = st.session_state.get('sync_canvas_data', None)

        if canvas_data and len(canvas_data.get('barras', [])) > 0:
            barras_dict = {"Barra": [], "Tipo": [], "V (pu)": [], "θ (graus)": [], "Pg": [], "Qg": [], "Pc": [], "Qc": [], "Bsh (pu)": []}
            for b in canvas_data['barras']:
                barras_dict["Barra"].append(b['id'])
                t = str(b['tipo']).upper()
                barras_dict["Tipo"].append("Slack" if t == "SLACK" else t)
                barras_dict["V (pu)"].append(float(b.get('v', 1.0)))
                barras_dict["θ (graus)"].append(float(b.get('theta', 0.0)))
                barras_dict["Pg"].append(float(b.get('p_ger', 0.0)))
                barras_dict["Qg"].append(float(b.get('q_ger', 0.0)))
                barras_dict["Pc"].append(float(b.get('p_carga', 0.0)))
                barras_dict["Qc"].append(float(b.get('q_carga', 0.0)))
                barras_dict["Bsh (pu)"].append(float(b.get('bsh_bus', 0.0)))
            df_barras_default = pd.DataFrame(barras_dict)

            linhas_dict = {"De": [], "Para": [], "R (pu)": [], "X (pu)": [], "Bsh_linha (pu)": []}
            for l in canvas_data.get('linhas', []):
                linhas_dict["De"].append(l['de'])
                linhas_dict["Para"].append(l['para'])
                linhas_dict["R (pu)"].append(float(l.get('r', 0.0)))
                linhas_dict["X (pu)"].append(float(l.get('x', 0.0)))
                linhas_dict["Bsh_linha (pu)"].append(float(l.get('bsh', 0.0)))
            df_linhas_default = pd.DataFrame(linhas_dict)

            trafos_dict = {"De (lado tap)": [], "Para": [], "R (pu)": [], "X (pu)": [], "Tap a (pu)": []}
            for t in canvas_data.get('transformadores', []):
                trafos_dict["De (lado tap)"].append(t['de'])
                trafos_dict["Para"].append(t['para'])
                trafos_dict["R (pu)"].append(float(t.get('r', 0.0)))
                trafos_dict["X (pu)"].append(float(t.get('x', 0.1)))
                trafos_dict["Tap a (pu)"].append(float(t.get('a', 1.0)))
            df_trafos_default = pd.DataFrame(trafos_dict)

        else:
            df_barras_default = pd.DataFrame({
                "Barra": [1, 2],
                "Tipo": ["Slack", "PQ"],
                "V (pu)": [1.06, 1.0],
                "θ (graus)": [0.0, 0.0],
                "Pg": [0.0, 0.0],
                "Qg": [0.0, 0.0],
                "Pc": [0.0, 50.0],
                "Qc": [0.0, 20.0],
                "Bsh (pu)": [0.0, 0.0]
            })
            df_linhas_default = pd.DataFrame({
                "De": [1], "Para": [2], "R (pu)": [0.05], "X (pu)": [0.1], "Bsh_linha (pu)": [0.0]
            })
            df_trafos_default = pd.DataFrame({
                "De (lado tap)": pd.Series([], dtype=int),
                "Para":          pd.Series([], dtype=int),
                "R (pu)":        pd.Series([], dtype=float),
                "X (pu)":        pd.Series([], dtype=float),
                "Tap a (pu)":    pd.Series([], dtype=float),
            })

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("Dados de Barras")
            df_barras_editado = st.data_editor(
                df_barras_default, num_rows="dynamic", use_container_width=True,
                column_config={"Tipo": st.column_config.SelectboxColumn(options=["Slack", "PV", "PQ"], required=True)}
            )
        with col_t2:
            st.subheader("Dados de Linhas")
            df_linhas_editado = st.data_editor(df_linhas_default, num_rows="dynamic", use_container_width=True)

        st.subheader("Dados de Transformadores em-Fase")
        st.caption("Coluna **'De (lado tap)'** = barra k onde o tap $a$ atua. **'Tap a'** = relação $V_k/V_m$. Use $a=1.0$ para tap nominal.")
        df_trafos_editado = st.data_editor(
            df_trafos_default, num_rows="dynamic", use_container_width=True,
            column_config={
                "Tap a (pu)": st.column_config.NumberColumn(min_value=0.01, max_value=5.0, step=0.01)
            }
        )

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("▶ EXECUTAR FLUXO DE POTÊNCIA", use_container_width=True, type="primary"):
                df_b = df_barras_editado.copy().fillna(0.0)
                df_b["V (pu)"] = df_b["V (pu)"].replace(0.0, 1.0)
                df_b["Tipo"]   = df_b["Tipo"].replace(0.0, "PQ")
                df_l = df_linhas_editado.copy().fillna(0.0)
                df_t = df_trafos_editado.copy().fillna(0.0)
                df_t["Tap a (pu)"] = df_t["Tap a (pu)"].replace(0.0, 1.0)

                barras_list = []
                for _, row in df_b.iterrows():
                    barras_list.append({
                        "id":      int(row["Barra"]),
                        "tipo":    str(row["Tipo"]).strip(),
                        "v":       float(row["V (pu)"]),
                        "theta":   float(row["θ (graus)"]),
                        "p_ger":   float(row["Pg"]),
                        "q_ger":   float(row["Qg"]),
                        "p_carga": float(row["Pc"]),
                        "q_carga": float(row["Qc"]),
                        "bsh_bus": float(row["Bsh (pu)"])
                    })
                linhas_list = []
                for _, row in df_l.iterrows():
                    linhas_list.append({
                        "de": int(row["De"]), "para": int(row["Para"]),
                        "r": float(row["R (pu)"]), "x": float(row["X (pu)"]),
                        "bsh": float(row["Bsh_linha (pu)"])
                    })
                trafos_list = []
                for _, row in df_t.iterrows():
                    trafos_list.append({
                        "de": int(row["De (lado tap)"]), "para": int(row["Para"]),
                        "r": float(row["R (pu)"]), "x": float(row["X (pu)"]),
                        "a": float(row["Tap a (pu)"])
                    })
                dados_para_calculo = {"barras": barras_list, "linhas": linhas_list, "transformadores": trafos_list}

    # ============================================================
    # MODO 2 — CIRCUITO (CANVAS)
    # ============================================================
    else:
        html_canvas = """
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; background: #cfcfcf; color: black; margin: 0; user-select: none; overflow: hidden; }
                .toolbar { background: #e0e0e0; padding: 10px; display: flex; gap: 8px; border-bottom: 2px solid #999; align-items: center; flex-wrap: wrap; }
                .btn { background: #fff; color: #333; border: 1px solid #999; padding: 8px 14px; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 13px; transition: 0.2s; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
                .btn:hover { background: #eee; }
                .btn-danger { color: #d32f2f; border-color: #d32f2f; }
                .btn-action { background: #1976d2; color: white; border-color: #115293; }
                .btn-trafo  { background: #6a1b9a; color: white; border-color: #4a148c; }
                .btn-calc   { background: #2e7d32; color: white; border-color: #1b5e20; margin-left: auto; }
                #workspace  { position: relative; height: 600px; cursor: default; background-color: #cfcfcf; }
                .component  { position: absolute; cursor: pointer; display: flex; flex-direction: column; align-items: center; transform: translate(-50%, -50%); z-index: 20; }
                .component.selected .barra-linha { box-shadow: 0 0 10px 3px #1976d2; background: #1976d2; }
                .barra-linha { background: #000; border-radius: 4px; transition: transform 0.2s; width: 8px; height: 120px; }
                .gerador-circulo { width: 44px; height: 44px; border: 2px solid #000; border-radius: 50%; background: #cfcfcf; display: flex; align-items: center; justify-content: center; font-size: 22px; font-weight: bold; }
                .carga-seta { width: 0; height: 0; border-left: 12px solid transparent; border-right: 12px solid transparent; border-top: 35px solid #000; }
                .label { font-size: 13px; position: absolute; white-space: nowrap; color: #111; font-weight: bold; background: rgba(255,255,255,0.85); padding: 5px 10px; border-radius: 6px; z-index: 30; border: 1px solid #999; box-shadow: 0 2px 5px rgba(0,0,0,0.3); pointer-events: none; }
                #wires { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 10; }
                .wire-path  { stroke: #000; stroke-width: 2.5; fill: none; pointer-events: stroke; cursor: pointer; }
                .trafo-path { stroke: #6a1b9a; stroke-width: 2.5; fill: none; pointer-events: stroke; cursor: pointer; stroke-dasharray: 8 4; }
                .wire-path.selected  { stroke: #1976d2; stroke-width: 5; }
                .trafo-path.selected { stroke: #6a1b9a; stroke-width: 5; }
                #properties-panel { position: absolute; right: 15px; top: 15px; width: 240px; background: #fff; border: 1px solid #999; box-shadow: 0 5px 20px rgba(0,0,0,0.4); padding: 18px; border-radius: 8px; display: none; z-index: 100; cursor: default; }
                #properties-panel h4 { margin: 0 0 12px 0; font-size: 15px; color: #222; border-bottom: 2px solid #1976d2; padding-bottom: 6px; }
                .prop-group { margin-bottom: 12px; }
                .prop-group label { display: block; font-size: 12px; margin-bottom: 4px; color: #444; font-weight: bold; }
                .prop-group input, .prop-group select { width: 100%; box-sizing: border-box; padding: 6px; font-size: 13px; border: 1px solid #ccc; border-radius: 4px; }
            </style>
        </head>
        <body>
            <div class="toolbar">
                <button class="btn" onclick="addBarra()">+ Barra</button>
                <button class="btn" onclick="attachComponent('gerador')">+ Gerador</button>
                <button class="btn" onclick="attachComponent('carga')">+ Carga</button>
                <button class="btn btn-action" id="btnConnect" onclick="toggleConnectMode('linha')">🔗 Conectar Linha</button>
                <button class="btn btn-trafo"  id="btnTrafo"   onclick="toggleConnectMode('trafo')">🔀 Conectar Trafo</button>
                <button class="btn btn-danger" onclick="deleteSelected()">🗑 Excluir</button>
                <button class="btn btn-calc"   onclick="exportarParaPython()">▶ CALCULAR FLUXO (DESENHO)</button>
            </div>
            <div id="workspace">
                <svg id="wires"></svg>
                <div id="properties-panel">
                    <h4 id="panel-title">Propriedades</h4>
                    <div id="panel-content"></div>
                </div>
            </div>

            <script>
                function sendMessageToStreamlit(data) {
                    window.parent.postMessage({ isStreamlitMessage: true, type: "streamlit:setComponentValue", value: data }, "*");
                }
                window.parent.postMessage({ isStreamlitMessage: true, type: "streamlit:componentReady", apiVersion: 1 }, "*");
                setInterval(() => {
                    window.parent.postMessage({ isStreamlitMessage: true, type: "streamlit:setFrameHeight", height: 650 }, "*");
                }, 500);

                let barras = [], linhas = [], transformadores = [];
                let geradores = [], cargas = [];
                let idCounter = 1;
                let selectedElement = null, selectedType = null;
                let connectMode = false, connectKind = null, connectStartBarra = null;

                const ws    = document.getElementById("workspace");
                const svg   = document.getElementById("wires");
                const panel = document.getElementById("properties-panel");
                panel.addEventListener('mousedown', e => e.stopPropagation());
                // Constantes declaradas antes de initSlack() (evita erro de TDZ em const)
                const SVG_NS = "http://www.w3.org/2000/svg";
                const fmtNum = v => String(+parseFloat(v).toFixed(4));
                const BUS_SHUNT_AO_LONGO = 30, BUS_SHUNT_DEGRAU = 40;

                function initSlack() {
                    const b = { id: idCounter++, type: 'slack', x: 150, y: 300, v: 1.06, theta: 0, bsh_bus: 0, rotState: 0, el: null };
                    barras.push(b); renderBarra(b);
                }
                initSlack();

                // Símbolo de elemento shunt SEMPRE na vertical, aterrado para baixo.
                // (x, y) = ponto de conexão no topo. val > 0 → capacitor (azul); val < 0 → reator (vermelho).
                // lado = +1 escreve o valor à direita do símbolo; -1 à esquerda.
                function shuntSymbolSVG(x, y, val, texto, lado) {
                    const isCap = val > 0;
                    const cor = isCap ? "#1976d2" : "#d32f2f";
                    const s = lado || 1;
                    let elemento;
                    if (isCap) {
                        elemento = `<path d="M ${x} ${y} L ${x} ${y+8}" stroke="#000" stroke-width="2"/>
                            <path d="M ${x-9} ${y+8} L ${x+9} ${y+8}" stroke="${cor}" stroke-width="3"/>
                            <path d="M ${x-9} ${y+13} L ${x+9} ${y+13}" stroke="${cor}" stroke-width="3"/>
                            <path d="M ${x} ${y+13} L ${x} ${y+22}" stroke="#000" stroke-width="2"/>`;
                    } else {
                        elemento = `<path d="M ${x} ${y} L ${x} ${y+4}" stroke="#000" stroke-width="2"/>
                            <path d="M ${x} ${y+4} q -8 2.5 0 5 q 8 2.5 0 5 q -8 2.5 0 5" fill="none" stroke="${cor}" stroke-width="2.5"/>
                            <path d="M ${x} ${y+19} L ${x} ${y+22}" stroke="#000" stroke-width="2"/>`;
                    }
                    const terra = `<path d="M ${x-10} ${y+22} L ${x+10} ${y+22}" stroke="#000" stroke-width="2"/>
                        <path d="M ${x-6} ${y+26} L ${x+6} ${y+26}" stroke="#000" stroke-width="2"/>
                        <path d="M ${x-2} ${y+30} L ${x+2} ${y+30}" stroke="#000" stroke-width="2"/>`;
                    const txt = `<text x="${x + s*13}" y="${y+15}" text-anchor="${s > 0 ? 'start' : 'end'}"
                        font-size="11" font-weight="bold" fill="${cor}">${texto}</text>`;
                    return elemento + terra + txt;
                }

                // Shunt de barra: desenhado na camada SVG global, ligado fisicamente à barra.
                // Barra vertical: derivação a 30 px abaixo do centro, degrau de 40 px para o lado da carga.
                // Barra horizontal: desce direto da barra, 35 px à direita do centro.
                // Geradores e cargas ficam no centro da barra (±140 px), então não há sobreposição.
                function renderBusShunt(b) {
                    if (!b.elShunt) {
                        b.elShunt = document.createElementNS(SVG_NS, "g");
                        b.elShunt.setAttribute("pointer-events", "none");
                        svg.appendChild(b.elShunt);
                    }
                    const val = parseFloat(b.bsh_bus) || 0;
                    if (val === 0) { b.elShunt.innerHTML = ""; return; }
                    const s = b.rotState || 0;
                    let html;
                    if (s === 0 || s === 2) {
                        const lado = (s === 0) ? 1 : -1;
                        const ya = b.y + BUS_SHUNT_AO_LONGO;
                        const xs = b.x + lado * BUS_SHUNT_DEGRAU;
                        html = `<path d="M ${b.x} ${ya} L ${xs} ${ya}" stroke="#000" stroke-width="2"/>`
                             + shuntSymbolSVG(xs, ya, val, fmtNum(val), lado);
                    } else {
                        const xa = b.x + BUS_SHUNT_AO_LONGO + 5;
                        html = shuntSymbolSVG(xa, b.y, val, fmtNum(val), 1);
                    }
                    b.elShunt.innerHTML = html;
                }

                function renderBarra(b) {
                    if (b.el) b.el.remove();
                    const el = document.createElement("div");
                    el.className = "component";
                    el.style.left = b.x + "px"; el.style.top = b.y + "px";
                    const g = geradores.find(x => x.barraId === b.id);
                    const c = cargas.find(x => x.barraId === b.id);
                    let net_p = (g ? g.p : 0) - (c ? c.p : 0);
                    let net_q = (g ? g.q : 0) - (c ? c.q : 0);
                    let topText = `Barra ${b.id}`, bottomText = "";
                    if (b.type === 'slack')      { topText += " (Slack)"; bottomText = `V=${b.v}∠${b.theta}°`; }
                    else if (g)                  { bottomText = `V=${g.v} | Pliq=${net_p.toFixed(2)}`; }
                    else                         { bottomText = `P=${net_p.toFixed(2)}, Q=${net_q.toFixed(2)}`; }
                    const angle = (b.rotState || 0) * 90;
                    el.innerHTML = `<div class="label" style="top:-35px;">${topText}</div>
                                    <div class="barra-linha" style="transform:rotate(${angle}deg);"></div>
                                    <div class="label" style="bottom:-35px;color:#1976d2">${bottomText}</div>`;
                    makeDraggable(el, b, 'barra');
                    ws.appendChild(el); b.el = el;
                    renderBusShunt(b);
                    if (selectedElement && selectedElement.id === b.id && selectedType === 'barra')
                        el.classList.add("selected");
                }

                function rotateBarra() {
                    if (selectedType !== 'barra') return;
                    selectedElement.rotState = ((selectedElement.rotState || 0) + 1) % 4;
                    renderBarra(selectedElement); updateAllWires();
                }

                function addBarra() {
                    const b = { id: idCounter++, type: 'PQ', x: 400, y: 300, bsh_bus: 0, rotState: 0, v: 1.0, theta: 0, el: null };
                    barras.push(b); renderBarra(b); selectElement(b, 'barra');
                }

                function attachComponent(tipo) {
                    if (selectedType !== 'barra') { alert("Selecione uma barra primeiro!"); return; }
                    const barra = selectedElement;
                    if (tipo === 'gerador') {
                        if (geradores.find(g => g.barraId === barra.id)) return;
                        const g = { id: idCounter++, barraId: barra.id, p: 0.5, q: 0.0, v: 1.04, el: null, elWire: null };
                        geradores.push(g); renderGerador(g);
                    } else if (tipo === 'carga') {
                        if (cargas.find(c => c.barraId === barra.id)) return;
                        const c = { id: idCounter++, barraId: barra.id, p: 1.0, q: 0.5, el: null, elWire: null };
                        cargas.push(c); renderCarga(c);
                    }
                    if (barra.type !== 'slack') {
                        barra.type = geradores.find(g => g.barraId === barra.id) ? 'PV' : 'PQ';
                    }
                    renderBarra(barra); updateAllWires();
                }

                function createComponentWire() {
                    const w = document.createElementNS("http://www.w3.org/2000/svg", "path");
                    w.setAttribute("stroke", "#000"); w.setAttribute("stroke-width", "2.5"); w.setAttribute("fill", "none");
                    svg.appendChild(w); return w;
                }
                function renderGerador(g) {
                    if (g.el) g.el.remove();
                    if (!g.elWire) g.elWire = createComponentWire();
                    const el = document.createElement("div"); el.className = "component";
                    el.innerHTML = `<div class="gerador-circulo">G</div>`;
                    ws.appendChild(el); g.el = el;
                    el.addEventListener("mousedown", e => { e.stopPropagation(); selectElement(g, 'gerador'); });
                }
                function renderCarga(c) {
                    if (c.el) c.el.remove();
                    if (!c.elWire) c.elWire = createComponentWire();
                    const el = document.createElement("div"); el.className = "component";
                    el.innerHTML = `<div class="carga-seta"></div>`;
                    ws.appendChild(el); c.el = el;
                    el.addEventListener("mousedown", e => { e.stopPropagation(); selectElement(c, 'carga'); });
                }
                function positionAttached(item, barra, tipo) {
                    const offset = 140; let cx = barra.x, cy = barra.y; const s = barra.rotState || 0;
                    if (s===0) { if(tipo==='gerador') cx-=offset; else cx+=offset; }
                    else if(s===1) { if(tipo==='gerador') cy-=offset; else cy+=offset; }
                    else if(s===2) { if(tipo==='gerador') cx+=offset; else cx-=offset; }
                    else if(s===3) { if(tipo==='gerador') cy+=offset; else cy-=offset; }
                    item.el.style.left = cx+"px"; item.el.style.top = cy+"px";
                    item.elWire.setAttribute("d", `M ${barra.x} ${barra.y} L ${cx} ${cy}`);
                }

                function toggleConnectMode(kind) {
                    const sameKind = connectMode && connectKind === kind;
                    connectMode = false; connectKind = null; connectStartBarra = null;
                    document.getElementById("btnConnect").style.background = "";
                    document.getElementById("btnConnect").innerText = "🔗 Conectar Linha";
                    document.getElementById("btnTrafo").style.background = "";
                    document.getElementById("btnTrafo").innerText = "🔀 Conectar Trafo";
                    if (!sameKind) {
                        connectMode = true; connectKind = kind;
                        if (kind === 'linha') {
                            document.getElementById("btnConnect").style.background = "#ff9800";
                            document.getElementById("btnConnect").innerText = "Cancelar Conexão";
                        } else {
                            document.getElementById("btnTrafo").style.background = "#ff9800";
                            document.getElementById("btnTrafo").innerText = "Cancelar Trafo";
                        }
                    }
                }

                function handleBarraClick(barra) {
                    if (!connectMode) { selectElement(barra, 'barra'); return; }
                    if (!connectStartBarra) {
                        connectStartBarra = barra; barra.el.classList.add("selected");
                    } else {
                        if (connectStartBarra.id !== barra.id) {
                            if (connectKind === 'linha') {
                                const existe = linhas.find(l =>
                                    (l.b1===barra.id && l.b2===connectStartBarra.id) ||
                                    (l.b1===connectStartBarra.id && l.b2===barra.id));
                                if (!existe) {
                                    const l = { id: idCounter++, b1: connectStartBarra.id, b2: barra.id,
                                                r: 0.05, x: 0.1, bsh: 0.0, elPath: null, elLabel: null, elSymbol: null };
                                    linhas.push(l); renderLinha(l);
                                }
                            } else if (connectKind === 'trafo') {
                                const existe = transformadores.find(t =>
                                    (t.bk===barra.id && t.bm===connectStartBarra.id) ||
                                    (t.bk===connectStartBarra.id && t.bm===barra.id));
                                if (!existe) {
                                    const t = { id: idCounter++, bk: connectStartBarra.id, bm: barra.id,
                                                r: 0.0, x: 0.1, a: 1.0, elPath: null, elLabel: null, elSymbol: null };
                                    transformadores.push(t); renderTrafo(t);
                                }
                            }
                        }
                        toggleConnectMode(connectKind); selectElement(null, null);
                    }
                }

                function renderLinha(l) {
                    const b1 = barras.find(b => b.id === l.b1);
                    const b2 = barras.find(b => b.id === l.b2);
                    if (!l.elPath) {
                        l.elPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
                        l.elPath.setAttribute("class", "wire-path");
                        l.elPath.addEventListener("mousedown", e => { e.stopPropagation(); selectElement(l, 'linha'); });
                        svg.appendChild(l.elPath);
                        l.elSymbol = document.createElementNS("http://www.w3.org/2000/svg", "g");
                        svg.appendChild(l.elSymbol);
                        // Grupo dos shunts da linha fora do grupo rotacionado → símbolos sempre verticais
                        l.elShunts = document.createElementNS(SVG_NS, "g");
                        l.elShunts.setAttribute("pointer-events", "none");
                        svg.appendChild(l.elShunts);
                        l.elLabel = document.createElement("div");
                        l.elLabel.className = "label"; l.elLabel.style.zIndex = "40";
                        ws.appendChild(l.elLabel);
                    }
                    l.elPath.setAttribute("d", `M ${b1.x} ${b1.y} L ${b2.x} ${b2.y}`);
                    const midX = (b1.x+b2.x)/2, midY = (b1.y+b2.y)/2;
                    const dx = b2.x-b1.x, dy = b2.y-b1.y;
                    const angle = Math.atan2(dy, dx)*180/Math.PI;
                    l.elSymbol.innerHTML = '';
                    l.elSymbol.setAttribute("transform", `translate(${midX},${midY}) rotate(${angle})`);
                    const bg = document.createElementNS("http://www.w3.org/2000/svg","rect");
                    bg.setAttribute("x","-25"); bg.setAttribute("y","-15"); bg.setAttribute("width","50"); bg.setAttribute("height","30"); bg.setAttribute("fill","#cfcfcf");
                    l.elSymbol.appendChild(bg);
                    let off = 0;
                    if (l.r > 0) {
                        const res = document.createElementNS("http://www.w3.org/2000/svg","path");
                        res.setAttribute("d","M -15 0 L -10 -8 L 0 8 L 10 -8 L 15 0"); res.setAttribute("fill","none"); res.setAttribute("stroke","#d32f2f"); res.setAttribute("stroke-width","2.5");
                        l.elSymbol.appendChild(res); off += 25;
                    }
                    if (l.x > 0) {
                        const ig = document.createElementNS("http://www.w3.org/2000/svg","g");
                        if (l.r > 0) ig.setAttribute("transform",`translate(${off},0)`);
                        const bgI = document.createElementNS("http://www.w3.org/2000/svg","rect");
                        bgI.setAttribute("x","-16"); bgI.setAttribute("y","-15"); bgI.setAttribute("width","32"); bgI.setAttribute("height","20"); bgI.setAttribute("fill","#cfcfcf");
                        ig.appendChild(bgI);
                        const ind = document.createElementNS("http://www.w3.org/2000/svg","path");
                        ind.setAttribute("d","M -15 0 Q -10 -15 -5 0 Q 0 -15 5 0 Q 10 -15 15 0"); ind.setAttribute("fill","none"); ind.setAttribute("stroke","#1976d2"); ind.setAttribute("stroke-width","2.5");
                        ig.appendChild(ind); l.elSymbol.appendChild(ig);
                    }
                    // Shunts do modelo π: B/2 em cada extremidade, sempre na vertical.
                    l.elShunts.innerHTML = '';
                    const bshVal = parseFloat(l.bsh) || 0;
                    const len = Math.hypot(dx, dy);
                    if (bshVal !== 0 && len > 0) {
                        const ux = dx/len, uy = dy/len;
                        // Linha íngreme (> ~35°): degrau horizontal para o lado oposto à descida da linha,
                        // senão o símbolo vertical ficaria em cima do próprio traço da linha.
                        const ingreme = Math.abs(ux) < 0.82;
                        let lado = 1;
                        if (ingreme) {
                            const dxDescendo = (uy >= 0) ? ux : -ux;
                            lado = dxDescendo > 0 ? -1 : 1;
                        }
                        const texto = "j" + fmtNum(bshVal/2);
                        // Obstáculos já desenhados (rótulos, corpo das barras, geradores, cargas) para evitar sobreposição
                        const wsR = ws.getBoundingClientRect();
                        const rotulos = Array.from(ws.querySelectorAll(".label, .barra-linha, .gerador-circulo, .carga-seta"))
                            .filter(e => e !== l.elLabel && e.offsetParent !== null)
                            .map(e => { const r = e.getBoundingClientRect();
                                        return { x1: r.left - wsR.left - 4, x2: r.right - wsR.left + 4,
                                                 y1: r.top - wsR.top - 4,   y2: r.bottom - wsR.top + 4 }; });
                        const caixaShunt = (px, py) => {
                            const sx = ingreme ? px + lado*22 : px;
                            const xt = sx + lado*(13 + 6.5*texto.length);
                            return { x1: Math.min(px, sx - 11, xt), x2: Math.max(px, sx + 11, xt),
                                     y1: py - 3, y2: py + 32 };
                        };
                        const sobreposicao = c => rotulos.reduce((acc, r) =>
                            acc + Math.max(0, Math.min(c.x2, r.x2) - Math.max(c.x1, r.x1))
                                * Math.max(0, Math.min(c.y2, r.y2) - Math.max(c.y1, r.y1)), 0);
                        // Para cada extremidade testa posições ao longo da linha (fração medida a partir da barra)
                        // e fica com a de menor sobreposição; a 1ª sem sobreposição encerra a busca.
                        const fracoes = [0.25, 0.30, 0.20, 0.35, 0.40, 0.15];
                        const posicao = (ox, oy, sgn) => {
                            let melhor = null, menor = Infinity;
                            for (const f of fracoes) {
                                const d = len * f;
                                const px = ox + sgn*ux*d, py = oy + sgn*uy*d;
                                const area = sobreposicao(caixaShunt(px, py));
                                if (area < menor) { menor = area; melhor = [px, py]; }
                                if (area === 0) break;
                            }
                            return melhor;
                        };
                        const pontos = [posicao(b1.x, b1.y, 1), posicao(b2.x, b2.y, -1)];
                        let html = "";
                        pontos.forEach(([px, py]) => {
                            let sx = px;
                            if (ingreme) {
                                sx = px + lado*22;
                                html += `<path d="M ${px} ${py} L ${sx} ${py}" stroke="#000" stroke-width="2"/>`;
                            }
                            html += `<circle cx="${px}" cy="${py}" r="2.5" fill="#000"/>`
                                  + shuntSymbolSVG(sx, py, bshVal, texto, lado);
                        });
                        l.elShunts.innerHTML = html;
                    }
                    l.elLabel.innerHTML = `Z: ${l.r}+j${l.x}`;
                    l.elLabel.style.left = midX+"px"; l.elLabel.style.top = (midY-40)+"px";
                    l.elLabel.style.transform = "translateX(-50%)";
                    if (selectedElement && selectedElement.id===l.id && selectedType==='linha')
                        l.elPath.classList.add("selected");
                    else
                        l.elPath.classList.remove("selected");
                }

                function renderTrafo(t) {
                    const bk = barras.find(b => b.id === t.bk);
                    const bm = barras.find(b => b.id === t.bm);
                    if (!t.elPath) {
                        t.elPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
                        t.elPath.setAttribute("class", "trafo-path");
                        t.elPath.addEventListener("mousedown", e => { e.stopPropagation(); selectElement(t, 'trafo'); });
                        svg.appendChild(t.elPath);
                        t.elSymbol = document.createElementNS("http://www.w3.org/2000/svg", "g");
                        svg.appendChild(t.elSymbol);
                        t.elLabel = document.createElement("div");
                        t.elLabel.className = "label"; t.elLabel.style.zIndex = "40";
                        ws.appendChild(t.elLabel);
                    }
                    t.elPath.setAttribute("d", `M ${bk.x} ${bk.y} L ${bm.x} ${bm.y}`);
                    const midX = (bk.x+bm.x)/2, midY = (bk.y+bm.y)/2;
                    const dx = bm.x-bk.x, dy = bm.y-bk.y;
                    const angle = Math.atan2(dy,dx)*180/Math.PI;
                    t.elSymbol.innerHTML = '';
                    t.elSymbol.setAttribute("transform", `translate(${midX},${midY}) rotate(${angle})`);
                    const bgT = document.createElementNS("http://www.w3.org/2000/svg","rect");
                    bgT.setAttribute("x","-35"); bgT.setAttribute("y","-20"); bgT.setAttribute("width","70"); bgT.setAttribute("height","40"); bgT.setAttribute("fill","#cfcfcf");
                    t.elSymbol.appendChild(bgT);
                    const ck = document.createElementNS("http://www.w3.org/2000/svg","circle");
                    ck.setAttribute("cx","-10"); ck.setAttribute("cy","0"); ck.setAttribute("r","12");
                    ck.setAttribute("fill","none"); ck.setAttribute("stroke","#6a1b9a"); ck.setAttribute("stroke-width","2.5");
                    t.elSymbol.appendChild(ck);
                    const cm = document.createElementNS("http://www.w3.org/2000/svg","circle");
                    cm.setAttribute("cx","10"); cm.setAttribute("cy","0"); cm.setAttribute("r","12");
                    cm.setAttribute("fill","none"); cm.setAttribute("stroke","#6a1b9a"); cm.setAttribute("stroke-width","2.5");
                    t.elSymbol.appendChild(cm);
                    const tap = document.createElementNS("http://www.w3.org/2000/svg","polygon");
                    tap.setAttribute("points","-28,-6 -22,0 -28,6");
                    tap.setAttribute("fill","#6a1b9a");
                    t.elSymbol.appendChild(tap);
                    t.elLabel.innerHTML = `🔀 T(a=${t.a}) B${t.bk}→B${t.bm}`;
                    t.elLabel.style.color = "#6a1b9a";
                    t.elLabel.style.left = midX+"px"; t.elLabel.style.top = (midY-44)+"px";
                    t.elLabel.style.transform = "translateX(-50%)";
                    if (selectedElement && selectedElement.id===t.id && selectedType==='trafo')
                        t.elPath.classList.add("selected");
                    else
                        t.elPath.classList.remove("selected");
                }

                function updateAllWires() {
                    barras.forEach(renderBusShunt);
                    linhas.forEach(renderLinha);
                    transformadores.forEach(renderTrafo);
                    geradores.forEach(g => positionAttached(g, barras.find(b => b.id===g.barraId), 'gerador'));
                    cargas.forEach(c => positionAttached(c, barras.find(b => b.id===c.barraId), 'carga'));
                }

                function makeDraggable(el, item, type) {
                    let drag = false, sx, sy;
                    el.addEventListener('mousedown', e => {
                        e.stopPropagation(); drag = true; sx = e.clientX-item.x; sy = e.clientY-item.y;
                        if (connectMode) {
                            handleBarraClick(item);
                        } else {
                            selectElement(item, 'barra');
                        }
                    });
                    document.addEventListener('mousemove', e => {
                        if (!drag) return;
                        const r = ws.getBoundingClientRect();
                        item.x = Math.max(30, Math.min(r.width-30,  e.clientX-sx));
                        item.y = Math.max(30, Math.min(r.height-30, e.clientY-sy));
                        el.style.left = item.x+"px"; el.style.top = item.y+"px"; updateAllWires();
                    });
                    document.addEventListener('mouseup', () => drag = false);
                }

                ws.addEventListener('mousedown', () => { if (!connectMode) selectElement(null, null); });

                function selectElement(item, type) {
                    selectedElement = item; selectedType = type;
                    document.querySelectorAll(".component.selected").forEach(el => el.classList.remove("selected"));
                    document.querySelectorAll(".wire-path.selected, .trafo-path.selected").forEach(el => el.classList.remove("selected"));
                    panel.style.display = item ? "block" : "none";
                    if (!item) return;

                    const content = document.getElementById("panel-content");
                    let html = "";

                    if (type === 'barra') {
                        item.el.classList.add("selected");
                        document.getElementById("panel-title").innerText = `Barra ${item.id} (${item.type})`;
                        const g = geradores.find(x => x.barraId===item.id);
                        const c = cargas.find(x => x.barraId===item.id);
                        html += `<button class="btn" style="width:100%;margin-bottom:15px;background:#f0f0f0;" onclick="rotateBarra()">↻ Girar a Barra</button>`;
                        if (item.type !== 'slack') {
                            html += `<div class="prop-group"><label>Tipo de Barra</label>
                                <select onchange="updateProp('type',this.value); selectElement(selectedElement,'barra');">
                                    <option value="PQ" ${item.type==='PQ'?'selected':''}>PQ (Carga)</option>
                                    <option value="PV" ${item.type==='PV'?'selected':''}>PV (Geração)</option>
                                </select></div>`;
                        }
                        if (item.type === 'slack') {
                            html += `<div class="prop-group"><label>Módulo V (pu)</label><input type="number" step="0.01" value="${item.v}" onchange="updateProp('v',this.value)"></div>`;
                            html += `<div class="prop-group"><label>Ângulo θ (°)</label><input type="number" step="1" value="${item.theta}" onchange="updateProp('theta',this.value)"></div>`;
                        }
                        if (g) {
                            html += `<h5 style="margin:10px 0 5px 0;color:#d32f2f;border-bottom:1px solid #ddd;">⚙️ Gerador</h5>`;
                            html += `<div class="prop-group"><label>Tensão V Fixa (pu)</label><input type="number" step="0.01" value="${g.v}" onchange="updateComponent('gerador','v',this.value)"></div>`;
                            html += `<div class="prop-group"><label>Potência Ativa Pg</label><input type="number" step="0.1" value="${g.p}" onchange="updateComponent('gerador','p',this.value)"></div>`;
                            html += `<div class="prop-group"><label>Potência Reativa Qg</label><input type="number" step="0.1" value="${g.q}" onchange="updateComponent('gerador','q',this.value)"></div>`;
                        }
                        if (c) {
                            html += `<h5 style="margin:10px 0 5px 0;color:#1976d2;border-bottom:1px solid #ddd;">🔋 Carga</h5>`;
                            html += `<div class="prop-group"><label>Carga Ativa Pc</label><input type="number" step="0.1" value="${c.p}" onchange="updateComponent('carga','p',this.value)"></div>`;
                            html += `<div class="prop-group"><label>Carga Reativa Qc</label><input type="number" step="0.1" value="${c.q}" onchange="updateComponent('carga','q',this.value)"></div>`;
                        }
                        html += `<hr style="margin:10px 0;"><div class="prop-group"><label>Shunt na Barra (pu)</label><input type="number" step="0.01" value="${item.bsh_bus}" onchange="updateProp('bsh_bus',this.value)"></div>`;

                    } else if (type === 'linha') {
                        item.elPath.classList.add("selected");
                        document.getElementById("panel-title").innerText = `Linha B${item.b1} ↔ B${item.b2}`;
                        html += `<div class="prop-group"><label>Resistência r (pu)</label><input type="number" step="0.001" value="${item.r}" onchange="updateProp('r',this.value)"></div>`;
                        html += `<div class="prop-group"><label>Reatância x (pu)</label><input type="number" step="0.001" value="${item.x}" onchange="updateProp('x',this.value)"></div>`;
                        html += `<div class="prop-group"><label>Susceptância shunt total B (pu) — B/2 em cada extremidade</label><input type="number" step="0.001" value="${item.bsh}" onchange="updateProp('bsh',this.value)"></div>`;

                    } else if (type === 'trafo') {
                        item.elPath.classList.add("selected");
                        document.getElementById("panel-title").innerText = `Trafo B${item.bk} → B${item.bm}`;
                        html += `<p style="font-size:12px;color:#6a1b9a;margin:0 0 10px 0;">▲ Barra <b>B${item.bk}</b> = lado do tap (<i>k</i>)</p>`;
                        html += `<div class="prop-group"><label>Resistência R (pu)</label><input type="number" step="0.001" value="${item.r}" onchange="updateProp('r',this.value)"></div>`;
                        html += `<div class="prop-group"><label>Reatância X (pu)</label><input type="number" step="0.001" value="${item.x}" onchange="updateProp('x',this.value)"></div>`;
                        html += `<div class="prop-group"><label>Tap a = V_k / V_m (pu)</label><input type="number" step="0.01" min="0.01" max="5.0" value="${item.a}" onchange="updateProp('a',this.value); renderTrafo(selectedElement);"></div>`;

                    } else if (type === 'gerador' || type === 'carga') {
                        selectElement(barras.find(b => b.id===item.barraId), 'barra');
                        return;
                    }

                    content.innerHTML = html;
                }

                const PROPS_TEXTO = ['type'];
                function updateProp(key, value) {
                    if (PROPS_TEXTO.includes(key)) {
                        selectedElement[key] = String(value);
                    } else {
                        const n = parseFloat(value);
                        if (!Number.isFinite(n)) { selectElement(selectedElement, selectedType); return; }
                        selectedElement[key] = n;
                    }
                    if (selectedType === 'barra') renderBarra(selectedElement);
                    if (selectedType === 'linha') updateAllWires();
                    if (selectedType === 'trafo') renderTrafo(selectedElement);
                }
                function updateComponent(compType, key, value) {
                    const n = parseFloat(value);
                    if (!Number.isFinite(n)) { selectElement(selectedElement, 'barra'); return; }
                    if (compType === 'gerador') geradores.find(x => x.barraId===selectedElement.id)[key] = n;
                    else cargas.find(x => x.barraId===selectedElement.id)[key] = n;
                    renderBarra(selectedElement); selectElement(selectedElement, 'barra');
                }

                function deleteSelected() {
                    if (!selectedElement) return;
                    if (selectedType === 'barra') {
                        if (selectedElement.type === 'slack') { alert("A Barra Slack não pode ser excluída!"); return; }
                        linhas.filter(l => l.b1===selectedElement.id || l.b2===selectedElement.id).forEach(l => {
                            if(l.elPath) l.elPath.remove(); if(l.elLabel) l.elLabel.remove(); if(l.elSymbol) l.elSymbol.remove(); if(l.elShunts) l.elShunts.remove();
                        });
                        linhas = linhas.filter(l => l.b1!==selectedElement.id && l.b2!==selectedElement.id);
                        transformadores.filter(t => t.bk===selectedElement.id || t.bm===selectedElement.id).forEach(t => {
                            if(t.elPath) t.elPath.remove(); if(t.elLabel) t.elLabel.remove(); if(t.elSymbol) t.elSymbol.remove();
                        });
                        transformadores = transformadores.filter(t => t.bk!==selectedElement.id && t.bm!==selectedElement.id);
                        geradores.filter(g => g.barraId===selectedElement.id).forEach(g => { g.el.remove(); g.elWire.remove(); });
                        geradores = geradores.filter(g => g.barraId!==selectedElement.id);
                        cargas.filter(c => c.barraId===selectedElement.id).forEach(c => { c.el.remove(); c.elWire.remove(); });
                        cargas = cargas.filter(c => c.barraId!==selectedElement.id);
                        if (selectedElement.elShunt) selectedElement.elShunt.remove();
                        selectedElement.el.remove();
                        barras = barras.filter(b => b.id!==selectedElement.id);
                    } else if (selectedType === 'linha') {
                        if(selectedElement.elPath) selectedElement.elPath.remove();
                        if(selectedElement.elLabel) selectedElement.elLabel.remove();
                        if(selectedElement.elSymbol) selectedElement.elSymbol.remove();
                        if(selectedElement.elShunts) selectedElement.elShunts.remove();
                        linhas = linhas.filter(l => l.id!==selectedElement.id);
                    } else if (selectedType === 'trafo') {
                        if(selectedElement.elPath) selectedElement.elPath.remove();
                        if(selectedElement.elLabel) selectedElement.elLabel.remove();
                        if(selectedElement.elSymbol) selectedElement.elSymbol.remove();
                        transformadores = transformadores.filter(t => t.id!==selectedElement.id);
                    }
                    selectElement(null, null); updateAllWires();
                }

                function exportarParaPython() {
                    const slacks = barras.filter(b => b.type === 'slack');
                    if (slacks.length === 0) { alert("Erro: nenhuma barra Slack encontrada."); return; }
                    const pvSemGerador = barras.filter(b => b.type==='PV' && !geradores.find(g => g.barraId===b.id));
                    if (pvSemGerador.length > 0) {
                        alert("Erro: barra(s) PV sem gerador associado: " + pvSemGerador.map(b=>`B${b.id}`).join(", "));
                        return;
                    }
                    const sistema = {
                        barras: barras.map(b => {
                            const g = geradores.find(x => x.barraId===b.id);
                            const c = cargas.find(x => x.barraId===b.id);
                            let obj = {
                                id: b.id, tipo: b.type, bsh_bus: b.bsh_bus,
                                p_ger: g ? g.p : 0, q_ger: g ? g.q : 0,
                                p_carga: c ? c.p : 0, q_carga: c ? c.q : 0
                            };
                            if (b.type === 'slack')      { obj.v = b.v; obj.theta = b.theta; }
                            else if (g)                  { obj.v = g.v; obj.theta = 0; }
                            else                         { obj.v = 1.0; obj.theta = 0; }
                            return obj;
                        }),
                        linhas: linhas.map(l => ({ de: l.b1, para: l.b2, r: l.r, x: l.x, bsh: l.bsh })),
                        transformadores: transformadores.map(t => ({ de: t.bk, para: t.bm, r: t.r, x: t.x, a: t.a }))
                    };
                    sendMessageToStreamlit(sistema);
                }
            </script>
        </body>
        </html>
        """

        component_dir = os.path.abspath("canvas_sep_component")
        os.makedirs(component_dir, exist_ok=True)
        with open(os.path.join(component_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(html_canvas)

        componente_canvas = components.declare_component("canvas_sep", path=component_dir)
        dados_canvas_bruto = componente_canvas(key="meu_canvas")

        if dados_canvas_bruto is not None:
            dados_para_calculo = dados_canvas_bruto
            st.session_state['sync_canvas_data'] = dados_canvas_bruto

    # ============================================================
    # PROCESSAMENTO DO MOTOR
    # ============================================================
    if dados_para_calculo is not None:
        st.markdown("---")

        erros_validacao = []
        tipos_barra = [str(b['tipo']).strip() for b in dados_para_calculo.get('barras', [])]
        tipos_invalidos = [str(b.get('id', '?')) for b, t in zip(dados_para_calculo.get('barras', []), tipos_barra)
                           if t.lower() not in ('slack', 'pv', 'pq')]
        if tipos_invalidos:
            erros_validacao.append(f"❌ Tipo de barra inválido na(s) barra(s) {', '.join(tipos_invalidos)}. "
                                   "Selecione a barra e redefina o tipo como PQ ou PV.")
        n_slack = sum(1 for t in tipos_barra if t.lower() == 'slack')
        if n_slack == 0:
            erros_validacao.append("❌ Nenhuma barra Slack encontrada. O sistema requer exatamente uma barra de referência.")
        elif n_slack > 1:
            erros_validacao.append(f"❌ {n_slack} barras Slack detectadas. Apenas uma é permitida.")
        if len(dados_para_calculo.get('linhas', [])) == 0 and len(dados_para_calculo.get('transformadores', [])) == 0:
            erros_validacao.append("❌ O sistema necessita de pelo menos uma linha ou transformador conectando os barramentos.")

        if erros_validacao:
            for e in erros_validacao:
                st.error(e)
            st.stop()

        with st.spinner("Solucionando pelo Método de Newton-Raphson..."):
            id_map       = {b['id']: idx   for idx, b in enumerate(dados_para_calculo['barras'])}
            id_map_label = {b['id']: idx+1 for idx, b in enumerate(dados_para_calculo['barras'])}
            tipo_map = {'slack': 'Slack', 'pv': 'PV', 'pq': 'PQ'}

            backend_buses = []
            for b in dados_para_calculo['barras']:
                tipo_real = tipo_map[str(b['tipo']).lower().strip()]
                p_liq_pu  = (b['p_ger'] - b['p_carga']) / divisor_potencia
                q_liq_pu  = 0.0 if tipo_real == 'PV' else (b['q_ger'] - b['q_carga']) / divisor_potencia
                backend_buses.append({
                    "type": tipo_real, "V": float(b.get('v', 1.0)),
                    "theta": float(b.get('theta', 0.0)),
                    "P": float(p_liq_pu), "Q": float(q_liq_pu),
                    "Bsh_bus": float(b.get('bsh_bus', 0.0))
                })

            backend_lines = []
            for l in dados_para_calculo.get('linhas', []):
                backend_lines.append({
                    "from": id_map[l['de']]+1, "to": id_map[l['para']]+1,
                    "R": float(l['r']), "X": float(l['x']), "Bsh": float(l.get('bsh', 0.0))
                })

            backend_trafos = []
            for t in dados_para_calculo.get('transformadores', []):
                backend_trafos.append({
                    "from": id_map[t['de']]+1, "to": id_map[t['para']]+1,
                    "R": float(t['r']), "X": float(t['x']), "a": float(t['a'])
                })

            try:
                Ybus = build_ybus(backend_buses, backend_lines, backend_trafos if backend_trafos else None)

                (V_final, theta_final, log_iteracoes,
                 pvpq, pq_index, P_spec, Q_spec, convergiu) = newton_raphson(
                    backend_buses, Ybus, tol=tol_input, max_iter=int(max_iter_input)
                )

                n_iter = len(log_iteracoes)
                if convergiu:
                    st.success(f"✅ O sistema convergiu em {n_iter} iteração(ões)!")
                else:
                    st.warning(f"⚠️ Limite de {int(max_iter_input)} iterações atingido sem convergência completa. Os resultados abaixo são aproximados.")

                st.markdown("#### Resultados Finais nas Barras")
                res_barras = []
                for b_orig in dados_para_calculo['barras']:
                    idx = id_map[b_orig['id']]
                    res_barras.append({
                        "Barra": id_map_label[b_orig['id']],
                        "ID Original": b_orig['id'],
                        "Tipo": str(b_orig['tipo']).upper(),
                        "Módulo |V| (pu)": f"{V_final[idx]:.4f}",
                        "Ângulo θ (°)": f"{np.degrees(theta_final[idx]):.4f}",
                    })
                st.dataframe(pd.DataFrame(res_barras), hide_index=True)
                st.markdown("---")

                st.markdown("<h2 style='text-align:center;color:#2e7d32;'>📚 Memória de Cálculo Analítica</h2>", unsafe_allow_html=True)
                st.caption(f"*(Processamento interno em p.u., S_base = {base_mva} MVA)*")

                st.markdown("### 🔹 Matriz de Admitância Nodal ($Y_{bus}$)")
                rotulos_y = [f"Barra {id_map_label[b['id']]}" for b in dados_para_calculo['barras']]
                df_ybus = formatar_ybus(Ybus)
                df_ybus.columns = rotulos_y
                df_ybus.index   = rotulos_y
                st.dataframe(df_ybus)

                st.markdown("### 🔹 Estado Inicial e Potências Injetadas Líquidas (p.u.)")
                col_ini1, col_ini2 = st.columns(2)
                with col_ini1:
                    st.markdown("**Vetor de Estado Inicial ($\\nu = 0$)**")
                    st.latex(r"V^{(0)} = "     + formatar_vetor_latex(log_iteracoes[0]['V_nu']))
                    st.latex(r"\theta^{(0)} = " + formatar_vetor_latex(log_iteracoes[0]['theta_nu']))
                with col_ini2:
                    st.markdown("**Potências Específicas**")
                    st.latex(r"P^{esp} = " + formatar_vetor_latex(P_spec))
                    st.latex(r"Q^{esp} = " + formatar_vetor_latex(Q_spec))

                st.markdown("### 🔹 Iteração $\\nu = 0$")
                dados_iter0 = log_iteracoes[0]
                col_p0, col_q0 = st.columns(2)
                with col_p0:
                    st.markdown("**Potência Ativa:**")
                    st.latex(r"P_{calc}^{(0)} = " + formatar_vetor_latex(dados_iter0['P_calc']))
                    st.latex(r"\Delta P^{(0)} = "  + formatar_vetor_latex(dados_iter0['dP']))
                with col_q0:
                    st.markdown("**Potência Reativa:**")
                    st.latex(r"Q_{calc}^{(0)} = " + formatar_vetor_latex(dados_iter0['Q_calc']))
                    st.latex(r"\Delta Q^{(0)} = "  + formatar_vetor_latex(dados_iter0['dQ']))

                st.markdown("**Teste de Convergência:**")
                st.latex(r"\max \left\{ |\Delta P|, |\Delta Q| \right\} = "
                         + f"{dados_iter0['erro']:.2e}"
                         + r" \quad \text{(Tolerância: } " + f"{tol_input:.0e}" + r"\text{)}")

                if dados_iter0['convergiu']:
                    st.success("✅ Critério de parada atendido na avaliação inicial.")
                else:
                    st.warning("⚠️ Critério não atingido. O algoritmo avança para o processo iterativo.")
                    st.markdown("---")
                    st.markdown("### 🔹 Processo Iterativo")
                    it_validas = [s for s in log_iteracoes if 'J' in s]
                    if it_validas:
                        iter_selecionada = st.selectbox(
                            "Selecione a Iteração ($\\nu$):",
                            options=[s['nu'] for s in it_validas],
                            format_func=lambda x: f"Iteração {x}  ➔  estado {x+1}"
                        )
                        dados_iter = next(s for s in it_validas if s['nu'] == iter_selecionada)
                        nu = dados_iter['nu']

                        st.markdown(f"#### 1. Derivadas Parciais — iteração {nu}")
                        col_j1, col_j2 = st.columns(2)
                        with col_j1:
                            if dados_iter['H'].size > 0:
                                st.markdown("**H ($\\partial P / \\partial \\theta$):**")
                                st.dataframe(pd.DataFrame(dados_iter['H']).map(lambda x: f"{x:.4f}"))
                            if dados_iter['M'].size > 0:
                                st.markdown("**M ($\\partial Q / \\partial \\theta$):**")
                                st.dataframe(pd.DataFrame(dados_iter['M']).map(lambda x: f"{x:.4f}"))
                        with col_j2:
                            if dados_iter['N'].size > 0:
                                st.markdown("**N ($\\partial P / \\partial V$):**")
                                st.dataframe(pd.DataFrame(dados_iter['N']).map(lambda x: f"{x:.4f}"))
                            if dados_iter['L'].size > 0:
                                st.markdown("**L ($\\partial Q / \\partial V$):**")
                                st.dataframe(pd.DataFrame(dados_iter['L']).map(lambda x: f"{x:.4f}"))

                        st.markdown(f"#### 2. Jacobiana Completa $J^{{({nu})}}$")
                        barras_pvpq = [dados_para_calculo['barras'][i]['id'] for i in pvpq]
                        barras_pq   = [dados_para_calculo['barras'][i]['id'] for i in pq_index]
                        rotulos_j   = ([f"Δθ(B{bid})" for bid in barras_pvpq] +
                                       [f"ΔV(B{bid})" for bid in barras_pq])
                        df_jacob = pd.DataFrame(dados_iter['J']).map(lambda x: f"{x:.4f}")
                        df_jacob.columns = rotulos_j
                        df_jacob.index   = rotulos_j
                        st.dataframe(df_jacob)

                        st.markdown("#### 3. Correções e Novo Estado")
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            st.markdown("**Vetor incremental $\\Delta x$:**")
                            st.latex(
                                r"\begin{bmatrix} \Delta\theta^{(" + str(nu) + r")} \\ \Delta V^{(" + str(nu) + r")} \end{bmatrix} = \begin{bmatrix} "
                                + r" \\ ".join([f"{v:.6f}" for v in dados_iter['dtheta']])
                                + r" \\ "
                                + r" \\ ".join([f"{v:.6f}" for v in dados_iter['dV']])
                                + r" \end{bmatrix}"
                            )
                        with col_d2:
                            st.markdown(f"**Novo estado ($\\nu={nu+1}$):**")
                            st.latex(r"\theta^{(" + str(nu+1) + r")} = " + formatar_vetor_latex(dados_iter['theta_prox']))
                            st.latex(r"V^{(" + str(nu+1) + r")} = "      + formatar_vetor_latex(dados_iter['V_prox']))

                st.markdown("---")
                st.markdown("### 🔹 Histórico de Convergência")
                hist_conv = []
                for s in log_iteracoes:
                    hist_conv.append({
                        "Iteração ν": s['nu'],
                        "Erro máximo": f"{s['erro']:.4e}",
                        "Convergiu?": "✅ Sim" if s['convergiu'] else "❌ Não",
                    })
                st.dataframe(pd.DataFrame(hist_conv), hide_index=True)

                st.markdown("---")
                st.markdown("### 🔹 Fluxos de Potência nos Ramos")
                from mismatch import calc_power as _calc_power
                P_f, Q_f = _calc_power(V_final, theta_final, Ybus)
                res_ramos = []

                for l in dados_para_calculo.get('linhas', []):
                    ki = id_map[l['de']]; mi = id_map[l['para']]
                    Z = complex(l['r'], l['x'])
                    if abs(Z) > 1e-12:
                        y_s  = 1.0 / Z
                        Vk   = V_final[ki] * np.exp(1j * theta_final[ki])
                        Vm   = V_final[mi] * np.exp(1j * theta_final[mi])
                        b_sh = complex(0, l.get('bsh', 0.0) / 2)
                        I_km = y_s * (Vk - Vm) + b_sh * Vk
                        I_mk = y_s * (Vm - Vk) + b_sh * Vm
                        S_km = Vk * np.conj(I_km); S_mk = Vm * np.conj(I_mk)
                        perda = S_km + S_mk
                        res_ramos.append({
                            "Ramo": f"L: B{l['de']} → B{l['para']}",
                            "P_km (pu)": f"{S_km.real:.4f}", "Q_km (pu)": f"{S_km.imag:.4f}",
                            "P_mk (pu)": f"{S_mk.real:.4f}", "Q_mk (pu)": f"{S_mk.imag:.4f}",
                            "Perda P (pu)": f"{perda.real:.4f}", "Perda Q (pu)": f"{perda.imag:.4f}",
                        })

                for t in dados_para_calculo.get('transformadores', []):
                    ki = id_map[t['de']]; mi = id_map[t['para']]
                    Z = complex(t['r'], t['x']); a = float(t['a'])
                    if abs(Z) > 1e-12:
                        y_km = 1.0 / Z
                        Vk   = V_final[ki] * np.exp(1j * theta_final[ki])
                        Vm   = V_final[mi] * np.exp(1j * theta_final[mi])
                        I_km = a**2 * y_km * Vk - a * y_km * Vm
                        I_mk = y_km * Vm - a * y_km * Vk
                        S_km = Vk * np.conj(I_km); S_mk = Vm * np.conj(I_mk)
                        perda = S_km + S_mk
                        res_ramos.append({
                            "Ramo": f"T (a={a}): B{t['de']} → B{t['para']}",
                            "P_km (pu)": f"{S_km.real:.4f}", "Q_km (pu)": f"{S_km.imag:.4f}",
                            "P_mk (pu)": f"{S_mk.real:.4f}", "Q_mk (pu)": f"{S_mk.imag:.4f}",
                            "Perda P (pu)": f"{perda.real:.4f}", "Perda Q (pu)": f"{perda.imag:.4f}",
                        })

                if res_ramos:
                    st.dataframe(pd.DataFrame(res_ramos), hide_index=True)
                    st.caption("**L** = Linha  |  **T** = Transformador  |  **P_km**: fluxo saindo de k  |  **P_mk**: fluxo saindo de m")

                st.markdown("---")
                st.markdown("### 🔹 Balanço de Potência do Sistema")
                P_ger_total = sum((b.get('p_ger', 0.0) / divisor_potencia) for b in dados_para_calculo['barras'])
                Q_ger_total = sum((b.get('q_ger', 0.0) / divisor_potencia) for b in dados_para_calculo['barras'])
                P_car_total = sum((b.get('p_carga', 0.0) / divisor_potencia) for b in dados_para_calculo['barras'])
                Q_car_total = sum((b.get('q_carga', 0.0) / divisor_potencia) for b in dados_para_calculo['barras'])
                P_slack     = float(P_f[next(i for i, b in enumerate(backend_buses) if b['type'] == 'Slack')])
                Q_slack     = float(Q_f[next(i for i, b in enumerate(backend_buses) if b['type'] == 'Slack')])
                P_perda     = sum(float(r["Perda P (pu)"]) for r in res_ramos) if res_ramos else 0.0

                col_bal1, col_bal2, col_bal3 = st.columns(3)
                with col_bal1:
                    st.metric("Geração Total P (pu)", f"{P_ger_total + P_slack:.4f}")
                    st.metric("Geração Total Q (pu)", f"{Q_ger_total + Q_slack:.4f}")
                with col_bal2:
                    st.metric("Carga Total P (pu)",   f"{P_car_total:.4f}")
                    st.metric("Carga Total Q (pu)",   f"{Q_car_total:.4f}")
                with col_bal3:
                    st.metric("Perdas P nos Ramos (pu)", f"{P_perda:.4f}")
                    st.metric("Injeção Slack P (pu)",    f"{P_slack:.4f}")

            except Exception as e:
                st.error(f"❌ Erro durante a simulação: {e}")