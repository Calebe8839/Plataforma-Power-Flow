import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import os

try:
    from ybus import build_ybus
    from newton_raphson import newton_raphson
except ImportError:
    st.error("⚠️ Arquivos do motor (ybus.py, newton_raphson.py, etc) não encontrados. Verifique se todos estão na mesma pasta.")
    st.stop()

# ======================================================
# FUNÇÕES AUXILIARES DE FORMATAÇÃO
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
st.set_page_config(page_title="Simulador SEP", layout="wide")

# ======================================================
# BARRA LATERAL
# ======================================================
with st.sidebar:
    st.header("⚙️ Parâmetros do Cálculo")
    tol_input      = st.number_input("Tolerância (Erro Máximo)", value=1e-6, format="%e", step=1e-7)
    max_iter_input = st.number_input("Máximo de Iterações", value=20, min_value=1, step=1)
    st.markdown("---")
    st.header("📊 Sistema e Unidades")
    base_mva = st.number_input("Base (MVA)", value=100.0, step=10.0)
    unidade  = st.selectbox("Unidade de Entrada de Potência", ["MW / MVar", "p.u."])

divisor_potencia = base_mva if unidade == "MW / MVar" else 1.0

st.markdown("<h2 style='text-align: center; color: #1976d2;'>Análise de Sistemas de Energia Elétrica</h2>", unsafe_allow_html=True)

# ======================================================
# ABAS PRINCIPAIS
# ======================================================
tab_teoria, tab_simulador = st.tabs(["📚 Fundamentação Teórica", "⚡ Simulador Prático"])

# ======================================================
# ABA 1 — FUNDAMENTAÇÃO TEÓRICA
# ======================================================
with tab_teoria:
    st.markdown("""
    ### 1. INTRODUÇÃO
    O crescimento dos sistemas elétricos de potência tornou indispensável o desenvolvimento de ferramentas matemáticas capazes de analisar o comportamento operacional das redes elétricas em regime permanente. Dentre essas ferramentas, o estudo do fluxo de potência possui papel central, permitindo determinar as magnitudes e ângulos das tensões nas barras do sistema, bem como os fluxos de potência ativa e reativa nas linhas de transmissão.

    A solução do problema de fluxo de carga envolve a resolução de um sistema não linear de equações algébricas. Entre os métodos numéricos empregados, o Método de Newton-Raphson (MNR) é amplamente utilizado devido à sua elevada precisão e rápida convergência, especialmente em sistemas de grande porte.

    ### 2. MODELO DE LINHA DE TRANSMISSÃO TIPO $\\pi$
    As linhas de transmissão apresentam características resistivas, indutivas e capacitivas distribuídas ao longo de sua extensão. Para análises de fluxo de potência, utiliza-se amplamente o modelo equivalente nominal tipo $\\pi$, pois este fornece boa precisão sem elevar excessivamente a complexidade computacional. A linha de transmissão conectando as barras $k$ e $m$ é representada por uma impedância série e duas susceptâncias shunt igualmente distribuídas:
    """)
    st.latex(r"Z_{km} = R_{km} + jX_{km}")
    if os.path.exists("image_517d43.png"):
        st.image("image_517d43.png", caption="Figura 1: Modelo equivalente tipo π da linha de transmissão.", width=500)
    st.markdown("""
    * A resistência $R$ representa as perdas ôhmicas da linha de transmissão.
    * A reatância $X$ representa os efeitos indutivos associados ao campo magnético criado pelos condutores.
    * A susceptância shunt $B_{sh}$ modela os efeitos capacitivos da linha em relação ao solo.

    A admitância série da linha é dada por:
    """)
    st.latex(r"Y_{km} = \frac{1}{R_{km} + jX_{km}} = G_{km} + jB_{km}")

    st.markdown("""
    ### 3. MODELO DO TRANSFORMADOR EM-FASE (EQUIVALENTE $\\pi$)
    O transformador em-fase com relação de transformação $1:a$ (onde $a = V_k/V_m$) e admitância série $y_{km} = 1/(R+jX)$ é representado pelo seguinte circuito equivalente $\\pi$:
    """)
    st.latex(r"A = a \cdot y_{km} \quad B = a(a-1) \cdot y_{km} \quad C = (1-a) \cdot y_{km}")
    st.markdown("""
    As contribuições na matriz $Y_{bus}$ são:
    """)
    st.latex(r"Y_{kk} \mathrel{+}= a^2 \cdot y_{km} \qquad Y_{mm} \mathrel{+}= y_{km} \qquad Y_{km} = Y_{mk} = -a \cdot y_{km}")
    st.markdown("""
    Para $a = 1$ (tap nominal), o modelo reduz-se à admitância série simples, equivalente a uma linha sem shunt. Para $a \\neq 1$, os shunts $B$ e $C$ têm sinais opostos, modelando o efeito do tap sobre as tensões terminais.

    ### 4. CONSTRUÇÃO DA MATRIZ DE ADMITÂNCIAS NODAIS
    A representação nodal do sistema elétrico é realizada através da matriz $Y_{bus}$. A relação fundamental da análise nodal é expressa por $I_{bus} = Y_{bus}V_{bus}$.

    Os elementos diagonais da matriz são formados pela soma das admitâncias conectadas à barra:
    """)
    st.latex(r"Y_{kk} = \sum Y_{km} + j\frac{B_{sh}}{2}")
    st.markdown("Os elementos fora da diagonal representam as conexões entre barras:")
    st.latex(r"Y_{km} = -Y_{linha}")

    st.markdown("""
    ### 5. FORMULAÇÃO DO PROBLEMA DE FLUXO DE POTÊNCIA
    A potência complexa injetada na barra $k$ é definida como $S_k = P_k + jQ_k$ e também $S_k = V_k I_k^*$. Separando as partes real e imaginária:
    """)
    st.latex(r"P_{k} = \sum_{m=1}^{NB} V_{k}V_{m} \left[ G_{km}\cos(\theta_{k}-\theta_{m}) + B_{km}\sin(\theta_{k}-\theta_{m}) \right]")
    st.latex(r"Q_{k} = \sum_{m=1}^{NB} V_{k}V_{m} \left[ G_{km}\sin(\theta_{k}-\theta_{m}) - B_{km}\cos(\theta_{k}-\theta_{m}) \right]")

    st.markdown("""
    ### 6. CLASSIFICAÇÃO DAS BARRAS
    * **Barra Slack (Referência):** São conhecidos a magnitude e o ângulo da tensão. As potências ativa e reativa são calculadas durante a solução.
    * **Barra PQ (Carga):** São especificadas as potências ativa e reativa. As incógnitas são a magnitude e o ângulo da tensão.
    * **Barra PV (Geração):** São conhecidos a potência ativa e a magnitude da tensão. As incógnitas são a potência reativa e o ângulo.

    ### 7. MÉTODO DE NEWTON-RAPHSON
    O método lineariza o sistema através da expansão em série de Taylor, resultando na formulação matricial com a Matriz Jacobiana ($J$):
    """)
    st.latex(r"\begin{bmatrix} \Delta P \\ \Delta Q \end{bmatrix} = \begin{bmatrix} H & N \\ M & L \end{bmatrix} \begin{bmatrix} \Delta \theta \\ \Delta V \end{bmatrix}")
    st.markdown("""
    As submatrizes representam as derivadas parciais:
    * $H = \\frac{\\partial P}{\\partial\\theta}$
    * $N = \\frac{\\partial P}{\\partial V}$
    * $M = \\frac{\\partial Q}{\\partial\\theta}$
    * $L = \\frac{\\partial Q}{\\partial V}$

    ### 8. ALGORITMO ITERATIVO
    1. Inicializar tensões e ângulos (perfil plano: $|V|=1$ pu, $\\theta=0$);
    2. Construir a matriz $Y_{bus}$ (linhas + transformadores + shunts de barra);
    3. Calcular potências injetadas ($P_{calc}$ e $Q_{calc}$);
    4. Determinar os resíduos $\\Delta P$ e $\\Delta Q$;
    5. Construir a matriz Jacobiana;
    6. Resolver o sistema linear para $\\Delta\\theta$ e $\\Delta V$;
    7. Atualizar as variáveis de estado;
    8. Verificar convergência;
    9. Repetir até atingir a tolerância especificada.

    ### 9. CONCLUSÃO
    A formulação matemática baseada na matriz Jacobiana permite resolver iterativamente as equações não lineares do sistema elétrico, fornecendo resultados precisos para tensões, ângulos e fluxos de potência, destacando-se pela sua elevada velocidade de convergência e robustez numérica em sistemas de grande porte.
    """)

# ======================================================
# ABA 2 — SIMULADOR PRÁTICO
# ======================================================
with tab_simulador:
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

            # Transformadores vindos do canvas
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
                df_b["V (pu)"]  = df_b["V (pu)"].replace(0.0, 1.0)
                df_b["Tipo"]    = df_b["Tipo"].replace(0.0, "PQ")
                df_l = df_linhas_editado.copy().fillna(0.0)
                df_t = df_trafos_editado.copy().fillna(0.0)
                df_t["Tap a (pu)"] = df_t["Tap a (pu)"].replace(0.0, 1.0)

                barras_list = []
                for _, row in df_b.iterrows():
                    barras_list.append({
                        "id":       int(row["Barra"]),
                        "tipo":     str(row["Tipo"]).strip(),
                        "v":        float(row["V (pu)"]),
                        "theta":    float(row["θ (graus)"]),
                        "p_ger":    float(row["Pg"]),
                        "q_ger":    float(row["Qg"]),
                        "p_carga":  float(row["Pc"]),
                        "q_carga":  float(row["Qc"]),
                        "bsh_bus":  float(row["Bsh (pu)"])
                    })

                linhas_list = []
                for _, row in df_l.iterrows():
                    linhas_list.append({
                        "de":   int(row["De"]),
                        "para": int(row["Para"]),
                        "r":    float(row["R (pu)"]),
                        "x":    float(row["X (pu)"]),
                        "bsh":  float(row["Bsh_linha (pu)"])
                    })

                trafos_list = []
                for _, row in df_t.iterrows():
                    trafos_list.append({
                        "de":   int(row["De (lado tap)"]),
                        "para": int(row["Para"]),
                        "r":    float(row["R (pu)"]),
                        "x":    float(row["X (pu)"]),
                        "a":    float(row["Tap a (pu)"])
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

                // ── Estado global ──────────────────────────────────────────
                let barras = [], linhas = [], transformadores = [];
                let geradores = [], cargas = [];
                let idCounter = 1;
                let selectedElement = null, selectedType = null;
                let connectMode = false, connectKind = null, connectStartBarra = null;

                const ws    = document.getElementById("workspace");
                const svg   = document.getElementById("wires");
                const panel = document.getElementById("properties-panel");
                panel.addEventListener('mousedown', e => e.stopPropagation());

                // ── Inicialização com barra Slack ──────────────────────────
                function initSlack() {
                    const b = { id: idCounter++, type: 'slack', x: 150, y: 300, v: 1.06, theta: 0, bsh_bus: 0, rotState: 0, el: null };
                    barras.push(b); renderBarra(b);
                }
                initSlack();

                // ── Shunt visual na barra ──────────────────────────────────
                function getBusShuntSVG(val) {
                    if (!val || val === 0) return "";
                    const isCap = val > 0;
                    const color = isCap ? "#1976d2" : "#d32f2f";
                    let symbol = isCap
                        ? `<path d="M -10 15 L 10 15" stroke="${color}" stroke-width="3"/><path d="M -10 20 L 10 20" stroke="${color}" stroke-width="3"/>`
                        : `<path d="M 0 10 Q -8 14 0 18 Q 8 22 0 26 Q -8 30 0 34" fill="none" stroke="${color}" stroke-width="2.5"/>`;
                    const yEnd = isCap ? 20 : 34; const yTerra = isCap ? 35 : 50;
                    return `<svg style="position:absolute; top:40px; left:-25px; width:50px; height:80px; pointer-events:none; overflow:visible; z-index:5;">
                        <path d="M 0 0 L 0 10" stroke="#000" stroke-width="2"/>${symbol}
                        <path d="M 0 ${yEnd} L 0 ${yTerra}" stroke="#000" stroke-width="2"/>
                        <path d="M -12 ${yTerra} L 12 ${yTerra}" stroke="#000" stroke-width="2"/>
                        <path d="M -8 ${yTerra+5} L 8 ${yTerra+5}" stroke="#000" stroke-width="2"/>
                        <path d="M -4 ${yTerra+10} L 4 ${yTerra+10}" stroke="#000" stroke-width="2"/>
                        <text x="15" y="25" font-size="12" font-weight="bold" fill="${color}">${val}</text>
                    </svg>`;
                }

                // ── Render de barra ────────────────────────────────────────
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
                                    <div class="label" style="bottom:-35px;color:#1976d2">${bottomText}</div>
                                    ${getBusShuntSVG(b.bsh_bus)}`;
                    makeDraggable(el, b, 'barra');
                    ws.appendChild(el); b.el = el;
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

                // ── Gerador / Carga ────────────────────────────────────────
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

                // ── Modo de conexão (linha ou trafo) ───────────────────────
                function toggleConnectMode(kind) {
                    const sameKind = connectMode && connectKind === kind;
                    // Desativa modo atual
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
                    if (!connectMode) return;
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
                                // Verifica se já existe trafo entre essas barras
                                const existe = transformadores.find(t =>
                                    (t.bk===barra.id && t.bm===connectStartBarra.id) ||
                                    (t.bk===connectStartBarra.id && t.bm===barra.id));
                                if (!existe) {
                                    // connectStartBarra = lado do tap (barra k)
                                    const t = { id: idCounter++, bk: connectStartBarra.id, bm: barra.id,
                                                r: 0.0, x: 0.1, a: 1.0,
                                                elPath: null, elLabel: null, elSymbol: null };
                                    transformadores.push(t); renderTrafo(t);
                                }
                            }
                        }
                        toggleConnectMode(connectKind); selectElement(null, null);
                    }
                }

                // ── Render de linha ────────────────────────────────────────
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
                    if (l.bsh > 0) {
                        const len = Math.sqrt(dx*dx+dy*dy);
                        if (len > 0) {
                            const o1 = -len/2+len*0.25, o2 = len/2-len*0.25;
                            const shuntSVG = (x) => `<path d="M ${x} 0 L ${x} 15" stroke="#000" stroke-width="2" fill="none"/>
                                <path d="M ${x-10} 15 L ${x+10} 15" stroke="#1976d2" stroke-width="3" fill="none"/>
                                <path d="M ${x-10} 20 L ${x+10} 20" stroke="#1976d2" stroke-width="3" fill="none"/>
                                <path d="M ${x} 20 L ${x} 35" stroke="#000" stroke-width="2" fill="none"/>
                                <path d="M ${x-12} 35 L ${x+12} 35" stroke="#000" stroke-width="2" fill="none"/>
                                <path d="M ${x-8} 40 L ${x+8} 40" stroke="#000" stroke-width="2" fill="none"/>
                                <path d="M ${x-4} 45 L ${x+4} 45" stroke="#000" stroke-width="2" fill="none"/>
                                <text x="${x+12}" y="25" font-size="12" font-weight="bold" fill="#1976d2">jbsh</text>`;
                            l.elSymbol.innerHTML += shuntSVG(o1) + shuntSVG(o2);
                        }
                    }
                    l.elLabel.innerHTML = `Z: ${l.r}+j${l.x}`;
                    l.elLabel.style.left = midX+"px"; l.elLabel.style.top = (midY-40)+"px";
                    l.elLabel.style.transform = "translateX(-50%)";
                    if (selectedElement && selectedElement.id===l.id && selectedType==='linha')
                        l.elPath.classList.add("selected");
                    else
                        l.elPath.classList.remove("selected");
                }

                // ── Render de transformador ────────────────────────────────
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

                    // Símbolo visual do transformador: dois círculos concêntricos no centro
                    t.elSymbol.innerHTML = '';
                    t.elSymbol.setAttribute("transform", `translate(${midX},${midY}) rotate(${angle})`);
                    const bgT = document.createElementNS("http://www.w3.org/2000/svg","rect");
                    bgT.setAttribute("x","-35"); bgT.setAttribute("y","-20"); bgT.setAttribute("width","70"); bgT.setAttribute("height","40"); bgT.setAttribute("fill","#cfcfcf");
                    t.elSymbol.appendChild(bgT);
                    // Círculo lado k (tap)
                    const ck = document.createElementNS("http://www.w3.org/2000/svg","circle");
                    ck.setAttribute("cx","-10"); ck.setAttribute("cy","0"); ck.setAttribute("r","12");
                    ck.setAttribute("fill","none"); ck.setAttribute("stroke","#6a1b9a"); ck.setAttribute("stroke-width","2.5");
                    t.elSymbol.appendChild(ck);
                    // Círculo lado m
                    const cm = document.createElementNS("http://www.w3.org/2000/svg","circle");
                    cm.setAttribute("cx","10"); cm.setAttribute("cy","0"); cm.setAttribute("r","12");
                    cm.setAttribute("fill","none"); cm.setAttribute("stroke","#6a1b9a"); cm.setAttribute("stroke-width","2.5");
                    t.elSymbol.appendChild(cm);
                    // Indicador de tap (triângulo pequeno no lado k)
                    const tap = document.createElementNS("http://www.w3.org/2000/svg","polygon");
                    tap.setAttribute("points","-28,-6 -22,0 -28,6");
                    tap.setAttribute("fill","#6a1b9a");
                    t.elSymbol.appendChild(tap);

                    // Rótulo com tap
                    t.elLabel.innerHTML = `🔀 T(a=${t.a}) B${t.bk}→B${t.bm}`;
                    t.elLabel.style.color = "#6a1b9a";
                    t.elLabel.style.left = midX+"px"; t.elLabel.style.top = (midY-44)+"px";
                    t.elLabel.style.transform = "translateX(-50%)";
                    if (selectedElement && selectedElement.id===t.id && selectedType==='trafo')
                        t.elPath.classList.add("selected");
                    else
                        t.elPath.classList.remove("selected");
                }

                // ── updateAllWires ─────────────────────────────────────────
                function updateAllWires() {
                    linhas.forEach(renderLinha);
                    transformadores.forEach(renderTrafo);
                    geradores.forEach(g => positionAttached(g, barras.find(b => b.id===g.barraId), 'gerador'));
                    cargas.forEach(c => positionAttached(c, barras.find(b => b.id===c.barraId), 'carga'));
                }

                // ── Arraste ────────────────────────────────────────────────
                function makeDraggable(el, item, type) {
                    let drag = false, sx, sy;
                    el.addEventListener('mousedown', e => {
                        e.stopPropagation(); drag = true; sx = e.clientX-item.x; sy = e.clientY-item.y;
                        handleBarraClick(item);
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

                // ── Painel de propriedades ─────────────────────────────────
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
                                <select onchange="updateProp('type',this.value); renderBarra(selectedElement);">
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
                        html += `<div class="prop-group"><label>Susceptância shunt bsh (pu)</label><input type="number" step="0.001" value="${item.bsh}" onchange="updateProp('bsh',this.value)"></div>`;

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

                function updateProp(key, value) {
                    selectedElement[key] = parseFloat(value);
                    if (selectedType === 'barra') renderBarra(selectedElement);
                    if (selectedType === 'linha') updateAllWires();
                    if (selectedType === 'trafo') renderTrafo(selectedElement);
                }
                function updateComponent(compType, key, value) {
                    if (compType === 'gerador') geradores.find(x => x.barraId===selectedElement.id)[key] = parseFloat(value);
                    else cargas.find(x => x.barraId===selectedElement.id)[key] = parseFloat(value);
                    renderBarra(selectedElement); selectElement(selectedElement, 'barra');
                }

                // ── Exclusão ───────────────────────────────────────────────
                function deleteSelected() {
                    if (!selectedElement) return;
                    if (selectedType === 'barra') {
                        if (selectedElement.type === 'slack') { alert("A Barra Slack não pode ser excluída!"); return; }
                        linhas.filter(l => l.b1===selectedElement.id || l.b2===selectedElement.id).forEach(l => {
                            if(l.elPath) l.elPath.remove(); if(l.elLabel) l.elLabel.remove(); if(l.elSymbol) l.elSymbol.remove();
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
                        selectedElement.el.remove();
                        barras = barras.filter(b => b.id!==selectedElement.id);
                    } else if (selectedType === 'linha') {
                        if(selectedElement.elPath) selectedElement.elPath.remove();
                        if(selectedElement.elLabel) selectedElement.elLabel.remove();
                        if(selectedElement.elSymbol) selectedElement.elSymbol.remove();
                        linhas = linhas.filter(l => l.id!==selectedElement.id);
                    } else if (selectedType === 'trafo') {
                        if(selectedElement.elPath) selectedElement.elPath.remove();
                        if(selectedElement.elLabel) selectedElement.elLabel.remove();
                        if(selectedElement.elSymbol) selectedElement.elSymbol.remove();
                        transformadores = transformadores.filter(t => t.id!==selectedElement.id);
                    }
                    selectElement(null, null); updateAllWires();
                }

                // ── Exportar para Python ───────────────────────────────────
                function exportarParaPython() {
                    // Validação: exatamente uma barra Slack
                    const slacks = barras.filter(b => b.type === 'slack');
                    if (slacks.length === 0) { alert("Erro: nenhuma barra Slack encontrada."); return; }

                    // Validação: barras PV precisam de gerador
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
    # PROCESSAMENTO DO MOTOR — COMUM A AMBOS OS MODOS
    # ============================================================
    if dados_para_calculo is not None:
        st.markdown("---")

        # ── Validações antes de chamar o motor ──────────────────
        erros_validacao = []

        tipos_barra = [str(b['tipo']).strip() for b in dados_para_calculo.get('barras', [])]
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
            # ── Mapeamento de IDs para índices internos 0..n-1 ──
            # A ordem da lista define os índices internos.
            # id_map: ID original da barra → índice interno (1-based para exibição)
            id_map       = {b['id']: idx for idx, b in enumerate(dados_para_calculo['barras'])}
            id_map_label = {b['id']: idx+1 for idx, b in enumerate(dados_para_calculo['barras'])}

            tipo_map = {'slack': 'Slack', 'pv': 'PV', 'pq': 'PQ'}

            backend_buses = []
            for b in dados_para_calculo['barras']:
                tipo_real      = tipo_map[str(b['tipo']).lower().strip()]
                p_liq_pu       = (b['p_ger'] - b['p_carga']) / divisor_potencia
                # Q de barras PV é incógnita — zerado aqui (motor também zera,
                # mas zeramos na fonte para clareza)
                if tipo_real == 'PV':
                    q_liq_pu = 0.0
                else:
                    q_liq_pu = (b['q_ger'] - b['q_carga']) / divisor_potencia

                backend_buses.append({
                    "type":  tipo_real,
                    "V":     float(b.get('v', 1.0)),
                    "theta": float(b.get('theta', 0.0)),
                    "P":     float(p_liq_pu),
                    "Q":     float(q_liq_pu),
                    "Bsh_bus": float(b.get('bsh_bus', 0.0))
                })

            backend_lines = []
            for l in dados_para_calculo.get('linhas', []):
                backend_lines.append({
                    "from": id_map[l['de']]  + 1,
                    "to":   id_map[l['para']]+ 1,
                    "R":    float(l['r']),
                    "X":    float(l['x']),
                    "Bsh":  float(l.get('bsh', 0.0))
                })

            backend_trafos = []
            for t in dados_para_calculo.get('transformadores', []):
                backend_trafos.append({
                    "from": id_map[t['de']]  + 1,
                    "to":   id_map[t['para']]+ 1,
                    "R":    float(t['r']),
                    "X":    float(t['x']),
                    "a":    float(t['a'])
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

                # ── Tabela de resultados ─────────────────────────
                st.markdown("#### Resultados Finais nas Barras")
                res_barras = []
                for b_orig, b_back in zip(dados_para_calculo['barras'], backend_buses):
                    idx = id_map[b_orig['id']]
                    res_barras.append({
                        "Barra":         id_map_label[b_orig['id']],
                        "ID Original":   b_orig['id'],
                        "Tipo":          str(b_orig['tipo']).upper(),
                        "Módulo |V| (pu)": f"{V_final[idx]:.4f}",
                        "Ângulo θ (°)":  f"{np.degrees(theta_final[idx]):.4f}",
                    })
                st.dataframe(pd.DataFrame(res_barras), hide_index=True)

                st.markdown("---")

                # ══════════════════════════════════════════════════
                # MEMÓRIA DE CÁLCULO EDUCACIONAL
                # ══════════════════════════════════════════════════
                st.markdown("<h2 style='text-align:center;color:#2e7d32;'>📚 Memória de Cálculo Analítica</h2>", unsafe_allow_html=True)
                st.caption(f"*(Processamento interno em p.u., S_base = {base_mva} MVA)*")

                # Ybus
                st.markdown("### 🔹 Matriz de Admitância Nodal ($Y_{bus}$)")
                rotulos_y = [f"Barra {id_map_label[b['id']]}" for b in dados_para_calculo['barras']]
                df_ybus = formatar_ybus(Ybus)
                df_ybus.columns = rotulos_y
                df_ybus.index   = rotulos_y
                st.dataframe(df_ybus)

                # Estado inicial e potências especificadas
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

                # Iteração 0
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
                        # Rótulos usando IDs originais das barras
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

            except Exception as e:
                st.error(f"❌ Erro durante a simulação: {e}")