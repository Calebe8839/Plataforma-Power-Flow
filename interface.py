import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import os

try:
    from ybus import build_ybus
    from newton_raphson import newton_raphson
except ImportError:
    st.error("⚠️ Arquivos do motor (ybus.py, newton_raphson.py, etc) não encontrados. Verifique se todos estão soltos na mesma pasta.")
    st.stop()

def formatar_vetor_latex(vec, precisao=4):
    elementos = [f"{v:.{precisao}f}" for v in vec]
    return r"\begin{bmatrix} " + r" \\ ".join(elementos) + r" \end{bmatrix}"

def formatar_ybus(ybus):
    df = pd.DataFrame(ybus)
    return df.map(lambda c: f"{c.real:.4f} {c.imag:+.4f}j")

st.set_page_config(page_title="Simulador SEP", layout="wide")

# ==========================================
# BARRA LATERAL E CONTROLES GERAIS
# ==========================================
with st.sidebar:
    st.header("⚙️ Parâmetros do Cálculo")
    tol_input = st.number_input("Tolerância (Erro Máximo)", value=0.001, format="%f", step=0.0001)
    max_iter_input = st.number_input("Máximo de Iterações", value=20, min_value=1, step=1)
    
    st.markdown("---")
    st.header("📊 Sistema e Unidades")
    base_mva = st.number_input("Base (MVA)", value=100.0, step=10.0)
    unidade = st.selectbox("Unidade de Entrada de Potência", ["MW / MVar", "p.u."])

divisor_potencia = base_mva if unidade == "MW / MVar" else 1.0

st.markdown("<h2 style='text-align: center; color: #1976d2;'>Análise de Sistemas de Energia Elétrica</h2>", unsafe_allow_html=True)

# ==========================================
# CRIAÇÃO DAS ABAS PRINCIPAIS
# ==========================================
tab_teoria, tab_simulador = st.tabs(["📚 Fundamentação Teórica", "⚡ Simulador Prático"])

# ==========================================
# ABA 1: FUNDAMENTAÇÃO TEÓRICA
# ==========================================
with tab_teoria:
    st.markdown("""
    ### 1. INTRODUÇÃO
    O crescimento dos sistemas elétricos de potência tornou indispensável o desenvolvimento de ferramentas matemáticas capazes de analisar o comportamento operacional das redes elétricas em regime permanente. Dentre essas ferramentas, o estudo do fluxo de potência possui papel central, permitindo determinar as magnitudes e ângulos das tensões nas barras do sistema, bem como os fluxos de potência ativa e reativa nas linhas de transmissão.
    
    A solução do problema de fluxo de carga envolve a resolução de um sistema não linear de equações algébricas. Entre os métodos numéricos empregados, o Método de Newton-Raphson (MNR) é amplamente utilizado devido à sua elevada precisão e rápida convergência, especialmente em sistemas de grande porte.
    
    ### 2. MODELO DE LINHA DE TRANSMISSÃO TIPO $\pi$
    As linhas de transmissão apresentam características resistivas, indutivas e capacitivas distribuídas ao longo de sua extensão. Para análises de fluxo de potência, utiliza-se amplamente o modelo equivalente nominal tipo $\pi$, pois este fornece boa precisão sem elevar excessivamente a complexidade computacional. A linha de transmissão conectando as barras $k$ e $m$ é representada por uma impedância série e duas susceptâncias shunt igualmente distribuídas:
    """)
    
    st.latex(r"Z_{km} = R_{km} + jX_{km}")
    
    if os.path.exists("image_517d43.png"):
        st.image("image_517d43.png", caption="Figura 1: Modelo equivalente tipo π da linha de transmissão.", width=500)
    
    st.markdown("""
    * A resistência $R$ representa as perdas ôhmicas da linha de transmissão, responsáveis pela dissipação de potência ativa. 
    * A reatância $X$ representa os efeitos indutivos associados ao campo magnético criado pelos condutores da linha. 
    * A susceptância shunt $B_{sh}$ modela os efeitos capacitivos da linha em relação ao solo e entre fases, contribuindo para a injeção de potência reativa no sistema.
    
    A admitância série da linha é dada por:
    """)
    st.latex(r"Y_{km} = \frac{1}{R_{km} + jX_{km}} = G_{km} + jB_{km}")

    st.markdown("""
    ### 3. CONSTRUÇÃO DA MATRIZ DE ADMITÂNCIAS NODAIS
    A representação nodal do sistema elétrico é realizada através da matriz de admitâncias nodais, conhecida como matriz $Y_{bus}$. A relação fundamental da análise nodal é expressa por $I_{bus} = Y_{bus}V_{bus}$.
    
    Os elementos diagonais da matriz são formados pela soma das admitâncias conectadas à barra:
    """)
    st.latex(r"Y_{kk} = \sum Y_{km} + j\frac{B_{sh}}{2}")
    
    st.markdown("Os elementos fora da diagonal representam as conexões entre barras:")
    st.latex(r"Y_{km} = -Y_{linha}")

    st.markdown("""
    ### 4. FORMULAÇÃO DO PROBLEMA DE FLUXO DE POTÊNCIA
    A potência complexa injetada na barra $k$ é definida como $S_k = P_k + jQ_k$ e também $S_k = V_k I_k^*$. Separando a parte real e imaginária, as equações clássicas do fluxo de potência tornam-se:
    """)
    st.latex(r"P_{k} = \sum_{m=1}^{NB} V_{k}V_{m} \left[ G_{km}\cos(\theta_{k}-\theta_{m}) + B_{km}\sin(\theta_{k}-\theta_{m}) \right]")
    st.latex(r"Q_{k} = \sum_{m=1}^{NB} V_{k}V_{m} \left[ G_{km}\sin(\theta_{k}-\theta_{m}) - B_{km}\cos(\theta_{k}-\theta_{m}) \right]")
    
    st.markdown("""
    ### 5. CLASSIFICAÇÃO DAS BARRAS
    * **Barra Slack (Referência):** São conhecidos a magnitude e o ângulo da tensão. As potências ativa e reativa são calculadas durante a solução.
    * **Barra PQ (Carga):** São especificadas as potências ativa e reativa. As incógnitas são a magnitude e o ângulo da tensão.
    * **Barra PV (Geração):** São conhecidos a potência ativa e a magnitude da tensão. As incógnitas são a potência reativa e o ângulo.
    
    ### 6. MÉTODO DE NEWTON-RAPHSON
    O método lineariza o sistema através da expansão em série de Taylor, resultando na formulação matricial com a Matriz Jacobiana ($J$):
    """)
    st.latex(r"\begin{bmatrix} \Delta P \\ \Delta Q \end{bmatrix} = \begin{bmatrix} H & N \\ M & L \end{bmatrix} \begin{bmatrix} \Delta \theta \\ \Delta V \end{bmatrix}")
    st.markdown("""
    As submatrizes representam as derivadas parciais: 
    * $H = \frac{\partial P}{\partial\theta}$
    * $N = \frac{\partial P}{\partial V}$
    * $M = \frac{\partial Q}{\partial\theta}$
    * $L = \frac{\partial Q}{\partial V}$
    
    ### 7. ALGORITMO ITERATIVO
    O algoritmo do Método de Newton-Raphson pode ser resumido nos seguintes passos estruturais:
    """)
    
    if os.path.exists("image_5be22d.png"):
        st.image("image_5be22d.png", caption="Figura 2: Estrutura do Algoritmo de Newton-Raphson.")
    else:
        st.markdown("""
        1. Inicializar tensões e ângulos;
        2. Construir a matriz $Y_{bus}$;
        3. Calcular potências injetadas ($P_{calc}$ e $Q_{calc}$);
        4. Determinar os erros $\Delta P$ e $\Delta Q$;
        5. Construir a matriz Jacobiana;
        6. Resolver o sistema linear para $\Delta \theta$ e $\Delta V$;
        7. Atualizar as variáveis de estado;
        8. Verificar convergência;
        9. Repetir o processo até atingir a tolerância especificada.
        """)
        
    st.markdown("""
    ### 8. CONCLUSÃO
    A formulação matemática baseada na matriz Jacobiana permite resolver iterativamente as equações não lineares do sistema elétrico, fornecendo resultados precisos para tensões, ângulos e fluxos de potência, destacando-se pela sua elevada velocidade de convergência e robustez numérica em sistemas de grande porte.
    """)

# ==========================================
# ABA 2: SIMULADOR PRÁTICO
# ==========================================
with tab_simulador:
    modo_entrada = st.radio(
        "Escolha a interface de modelagem do sistema:",
        ["Modo Tabela (Entrada Analítica)", "Modo Circuito (Diagrama Unifilar)"],
        horizontal=True
    )

    st.markdown("---")
    dados_para_calculo = None

    # ==========================================
    # MODO 1: TABELA
    # ==========================================
    if modo_entrada == "Modo Tabela (Entrada Analítica)":
        st.markdown("Insira os parâmetros elétricos dos barramentos e das linhas de transmissão. **Dica:** Os dados do Modo Circuito são sincronizados aqui automaticamente.")
        
        # Recupera os dados desenhados no Canvas (se existirem)
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
            
        else: # Padrão caso não haja desenho
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

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("Dados de Barras")
            df_barras_editado = st.data_editor(
                df_barras_default, num_rows="dynamic", use_container_width=True,
                column_config={"Tipo": st.column_config.SelectboxColumn(options=["Slack", "PV", "PQ"], required=True)}
            )

        with col_t2:
            st.subheader("Dados de Linhas")
            df_linhas_editado = st.data_editor(
                df_linhas_default, num_rows="dynamic", use_container_width=True
            )

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("▶ EXECUTAR FLUXO DE POTÊNCIA", use_container_width=True, type="primary"):
                df_b = df_barras_editado.copy().fillna(0.0)
                df_b["V (pu)"] = df_b["V (pu)"].replace(0.0, 1.0)
                df_b["Tipo"] = df_b["Tipo"].replace(0.0, "PQ")
                
                df_l = df_linhas_editado.copy().fillna(0.0)

                barras_list = []
                for _, row in df_b.iterrows():
                    barras_list.append({
                        "id": int(row["Barra"]),
                        "tipo": str(row["Tipo"]).strip(),
                        "v": float(row["V (pu)"]),
                        "theta": float(row["θ (graus)"]),
                        "p_ger": float(row["Pg"]),
                        "q_ger": float(row["Qg"]),
                        "p_carga": float(row["Pc"]),
                        "q_carga": float(row["Qc"]),
                        "bsh_bus": float(row["Bsh (pu)"])
                    })
                
                linhas_list = []
                for _, row in df_l.iterrows():
                    linhas_list.append({
                        "de": int(row["De"]),
                        "para": int(row["Para"]),
                        "r": float(row["R (pu)"]),
                        "x": float(row["X (pu)"]),
                        "bsh": float(row["Bsh_linha (pu)"])
                    })
                    
                dados_para_calculo = {"barras": barras_list, "linhas": linhas_list}

    # ==========================================
    # MODO 2: CIRCUITO (CANVAS)
    # ==========================================
    else:
        html_canvas = """
        <!DOCTYPE html>
        <html lang="pt-PT">
        <head>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; background: #cfcfcf; color: black; margin: 0; user-select: none; overflow: hidden; }
                .toolbar { background: #e0e0e0; padding: 10px; display: flex; gap: 8px; border-bottom: 2px solid #999; align-items: center; }
                .btn { background: #fff; color: #333; border: 1px solid #999; padding: 8px 14px; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 13px; transition: 0.2s; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
                .btn:hover { background: #eee; }
                .btn-danger { color: #d32f2f; border-color: #d32f2f; }
                .btn-action { background: #1976d2; color: white; border-color: #115293; }
                .btn-calc { background: #2e7d32; color: white; border-color: #1b5e20; margin-left: auto; }
                #workspace { position: relative; height: 600px; cursor: default; background-color: #cfcfcf; }
                .component { position: absolute; cursor: pointer; display: flex; flex-direction: column; align-items: center; transform: translate(-50%, -50%); z-index: 20; }
                .component.selected .barra-linha { box-shadow: 0 0 10px 3px #1976d2; background: #1976d2; }
                .barra-linha { background: #000; border-radius: 4px; transition: transform 0.2s; width: 8px; height: 120px; }
                .gerador-circulo { width: 44px; height: 44px; border: 2px solid #000; border-radius: 50%; background: #cfcfcf; display: flex; align-items: center; justify-content: center; font-size: 22px; font-weight: bold; }
                .carga-seta { width: 0; height: 0; border-left: 12px solid transparent; border-right: 12px solid transparent; border-top: 35px solid #000; }
                .label { font-size: 13px; position: absolute; white-space: nowrap; color: #111; font-weight: bold; background: rgba(255,255,255,0.85); padding: 5px 10px; border-radius: 6px; z-index: 30; border: 1px solid #999; box-shadow: 0 2px 5px rgba(0,0,0,0.3); pointer-events: none; }
                #wires { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 10; }
                .wire-path { stroke: #000; stroke-width: 2.5; fill: none; pointer-events: stroke; cursor: pointer; }
                .wire-path.selected { stroke: #1976d2; stroke-width: 5; }
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
                <button class="btn btn-action" id="btnConnect" onclick="toggleConnectMode()">🔗 Conectar</button>
                <button class="btn btn-danger" onclick="deleteSelected()">🗑 Excluir</button>
                <button class="btn btn-calc" onclick="exportarParaPython()">▶ CALCULAR FLUXO (DESENHO)</button>
            </div>
            <div id="workspace"><svg id="wires"></svg><div id="properties-panel"><h4 id="panel-title">Propriedades</h4><div id="panel-content"></div></div></div>

            <script>
                function sendMessageToStreamlit(data) { window.parent.postMessage({ isStreamlitMessage: true, type: "streamlit:setComponentValue", value: data }, "*"); }
                window.parent.postMessage({ isStreamlitMessage: true, type: "streamlit:componentReady", apiVersion: 1 }, "*");
                setInterval(() => { window.parent.postMessage({ isStreamlitMessage: true, type: "streamlit:setFrameHeight", height: 650 }, "*"); }, 500);

                let barras = []; let linhas = []; let geradores = []; let cargas = []; let idCounter = 1;
                let selectedElement = null; let selectedType = null; let connectMode = false; let connectStartBarra = null;
                const ws = document.getElementById("workspace"); const svg = document.getElementById("wires"); const panel = document.getElementById("properties-panel");
                panel.addEventListener('mousedown', (e) => e.stopPropagation());
                
                function initSlack() { const b1 = { id: idCounter++, type: 'slack', x: 150, y: 300, v: 1.06, theta: 0, bsh_bus: 0, rotState: 0, el: null }; barras.push(b1); renderBarra(b1); }
                initSlack();

                function getBusShuntSVG(val) {
                    if (!val || val === 0) return "";
                    const isCap = val > 0;
                    const color = isCap ? "#1976d2" : "#d32f2f";
                    let symbol = isCap 
                        ? `<path d="M -10 15 L 10 15" stroke="${color}" stroke-width="3"/><path d="M -10 20 L 10 20" stroke="${color}" stroke-width="3"/>`
                        : `<path d="M 0 10 Q -8 14 0 18 Q 8 22 0 26 Q -8 30 0 34" fill="none" stroke="${color}" stroke-width="2.5"/>`;
                    const yEnd = isCap ? 20 : 34; const yTerra = isCap ? 35 : 50;
                    return `<svg style="position:absolute; top:40px; left:-25px; width:50px; height:80px; pointer-events:none; overflow:visible; z-index: 5;">
                        <path d="M 0 0 L 0 10" stroke="#000" stroke-width="2"/>${symbol}<path d="M 0 ${yEnd} L 0 ${yTerra}" stroke="#000" stroke-width="2"/>
                        <path d="M -12 ${yTerra} L 12 ${yTerra}" stroke="#000" stroke-width="2"/>
                        <path d="M -8 ${yTerra+5} L 8 ${yTerra+5}" stroke="#000" stroke-width="2"/>
                        <path d="M -4 ${yTerra+10} L 4 ${yTerra+10}" stroke="#000" stroke-width="2"/>
                        <text x="15" y="25" font-size="12" font-weight="bold" fill="${color}" style="background:white;">${val}</text>
                    </svg>`;
                }

                function renderBarra(b) {
                    if(b.el) b.el.remove();
                    const el = document.createElement("div"); el.className = "component"; el.style.left = b.x + "px"; el.style.top = b.y + "px";
                    
                    const g = geradores.find(x => x.barraId === b.id);
                    const c = cargas.find(x => x.barraId === b.id);

                    let bottomText = ""; let topText = `Barra ${b.id}`;
                    let net_p = (g ? g.p : 0) - (c ? c.p : 0);
                    let net_q = (g ? g.q : 0) - (c ? c.q : 0);

                    if (b.type === 'slack') { 
                        topText += " (Slack)"; bottomText = `V=${b.v}∠${b.theta}°`; 
                    } else if (g) { 
                        bottomText = `V=${g.v} | Pliq=${net_p.toFixed(2)}`; 
                    } else { 
                        bottomText = `P=${net_p.toFixed(2)}, Q=${net_q.toFixed(2)}`; 
                    }
                    
                    const angle = (b.rotState || 0) * 90;
                    const shuntVisual = getBusShuntSVG(b.bsh_bus);
                    
                    el.innerHTML = `<div class="label" style="top: -35px;">${topText}</div>
                                    <div class="barra-linha" style="transform: rotate(${angle}deg);"></div>
                                    <div class="label" style="bottom: -35px; color:#1976d2">${bottomText}</div>
                                    ${shuntVisual}`;
                    makeDraggable(el, b, 'barra'); ws.appendChild(el); b.el = el;
                    if(selectedElement && selectedElement.id === b.id && selectedType === 'barra') el.classList.add("selected");
                }

                function rotateBarra() { if (selectedType !== 'barra') return; selectedElement.rotState = ((selectedElement.rotState || 0) + 1) % 4; renderBarra(selectedElement); updateAllWires(); }
                function addBarra() { const b = { id: idCounter++, type: 'PQ', x: 400, y: 300, bsh_bus: 0, rotState: 0, v: 1.0, theta: 0, el: null }; barras.push(b); renderBarra(b); selectElement(b, 'barra'); }

                function attachComponent(tipo) {
                    if (selectedType !== 'barra') { alert("Selecione uma barra primeiro!"); return; }
                    const barra = selectedElement; 
                    
                    if (tipo === 'gerador') {
                        if(geradores.find(g => g.barraId === barra.id)) return;
                        const g = { id: idCounter++, barraId: barra.id, p: 0.5, q: 0.0, v: 1.04, el: null, elWire: null }; 
                        geradores.push(g); renderGerador(g);
                    } else if (tipo === 'carga') {
                        if(cargas.find(c => c.barraId === barra.id)) return; 
                        const c = { id: idCounter++, barraId: barra.id, p: 1.0, q: 0.5, el: null, elWire: null }; 
                        cargas.push(c); renderCarga(c);
                    }
                    
                    if (barra.type !== 'slack') {
                        if (geradores.find(g => g.barraId === barra.id)) barra.type = 'PV'; else barra.type = 'PQ';
                    }
                    renderBarra(barra); updateAllWires();
                }

                function createComponentWire() { const w = document.createElementNS("http://www.w3.org/2000/svg", "path"); w.setAttribute("stroke", "#000"); w.setAttribute("stroke-width", "2.5"); w.setAttribute("fill", "none"); svg.appendChild(w); return w; }
                function renderGerador(g) { if(g.el) g.el.remove(); if(!g.elWire) g.elWire = createComponentWire(); const el = document.createElement("div"); el.className = "component"; el.innerHTML = `<div class="gerador-circulo">G</div>`; ws.appendChild(el); g.el = el; el.addEventListener("mousedown", (e) => { e.stopPropagation(); selectElement(g, 'gerador'); }); }
                function renderCarga(c) { if(c.el) c.el.remove(); if(!c.elWire) c.elWire = createComponentWire(); const el = document.createElement("div"); el.className = "component"; el.innerHTML = `<div class="carga-seta"></div>`; ws.appendChild(el); c.el = el; el.addEventListener("mousedown", (e) => { e.stopPropagation(); selectElement(c, 'carga'); }); }

                function positionAttached(item, barra, tipo) {
                    const offset = 140; 
                    let cx = barra.x, cy = barra.y; const state = barra.rotState || 0;
                    if (state === 0) { if (tipo === 'gerador') cx -= offset; else cx += offset; } 
                    else if (state === 1) { if (tipo === 'gerador') cy -= offset; else cy += offset; } 
                    else if (state === 2) { if (tipo === 'gerador') cx += offset; else cx -= offset; } 
                    else if (state === 3) { if (tipo === 'gerador') cy += offset; else cy -= offset; }
                    item.el.style.left = cx + "px"; item.el.style.top = cy + "px"; item.elWire.setAttribute("d", `M ${barra.x} ${barra.y} L ${cx} ${cy}`);
                }

                function toggleConnectMode() {
                    connectMode = !connectMode; const btn = document.getElementById("btnConnect");
                    if(connectMode) { btn.style.background = "#ff9800"; btn.innerText = "Cancelar Conexão"; connectStartBarra = null; } 
                    else { btn.style.background = ""; btn.innerText = "🔗 Conectar Barras"; }
                }

                function handleBarraClick(barra) {
                    if(connectMode) {
                        if(!connectStartBarra) { connectStartBarra = barra; barra.el.classList.add("selected"); } 
                        else {
                            if(connectStartBarra.id !== barra.id) {
                                const existe = linhas.find(l => (l.b1 === barra.id && l.b2 === connectStartBarra.id) || (l.b1 === connectStartBarra.id && l.b2 === barra.id));
                                if(!existe) {
                                    const l = { id: idCounter++, b1: connectStartBarra.id, b2: barra.id, r: 0.05, x: 0.1, bsh: 0.0, elPath: null, elLabel: null, elSymbol: null };
                                    linhas.push(l); renderLinha(l);
                                }
                            }
                            toggleConnectMode(); selectElement(null, null);
                        }
                    } else { selectElement(barra, 'barra'); }
                }

                function renderLinha(l) {
                    const b1 = barras.find(b => b.id === l.b1); const b2 = barras.find(b => b.id === l.b2);
                    if(!l.elPath) {
                        l.elPath = document.createElementNS("http://www.w3.org/2000/svg", "path"); l.elPath.setAttribute("class", "wire-path");
                        l.elPath.addEventListener("mousedown", (e) => { e.stopPropagation(); selectElement(l, 'linha'); }); svg.appendChild(l.elPath);
                        l.elSymbol = document.createElementNS("http://www.w3.org/2000/svg", "g"); svg.appendChild(l.elSymbol);
                        l.elLabel = document.createElement("div"); l.elLabel.className = "label"; l.elLabel.style.zIndex = "40"; ws.appendChild(l.elLabel);
                    }
                    
                    l.elPath.setAttribute("d", `M ${b1.x} ${b1.y} L ${b2.x} ${b2.y}`);
                    const midX = (b1.x + b2.x) / 2; const midY = (b1.y + b2.y) / 2;
                    const dx = b2.x - b1.x; const dy = b2.y - b1.y; let angle = Math.atan2(dy, dx) * 180 / Math.PI;
                    
                    l.elSymbol.innerHTML = ''; l.elSymbol.setAttribute("transform", `translate(${midX}, ${midY}) rotate(${angle})`);
                    const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect"); bg.setAttribute("x", "-25"); bg.setAttribute("y", "-15"); bg.setAttribute("width", "50"); bg.setAttribute("height", "30"); bg.setAttribute("fill", "#cfcfcf"); l.elSymbol.appendChild(bg);

                    let offsetSymbol = 0;
                    if(l.r > 0) {
                        const res = document.createElementNS("http://www.w3.org/2000/svg", "path"); res.setAttribute("d", "M -15 0 L -10 -8 L 0 8 L 10 -8 L 15 0"); res.setAttribute("fill", "none"); res.setAttribute("stroke", "#d32f2f"); res.setAttribute("stroke-width", "2.5");
                        l.elSymbol.appendChild(res); offsetSymbol += 25;
                    }
                    if(l.x > 0) {
                        const indGroup = document.createElementNS("http://www.w3.org/2000/svg", "g"); if(l.r > 0) indGroup.setAttribute("transform", `translate(${offsetSymbol}, 0)`);
                        const bgInd = document.createElementNS("http://www.w3.org/2000/svg", "rect"); bgInd.setAttribute("x", "-16"); bgInd.setAttribute("y", "-15"); bgInd.setAttribute("width", "32"); bgInd.setAttribute("height", "20"); bgInd.setAttribute("fill", "#cfcfcf"); indGroup.appendChild(bgInd);
                        const ind = document.createElementNS("http://www.w3.org/2000/svg", "path"); ind.setAttribute("d", "M -15 0 Q -10 -15 -5 0 Q 0 -15 5 0 Q 10 -15 15 0"); ind.setAttribute("fill", "none"); ind.setAttribute("stroke", "#1976d2"); ind.setAttribute("stroke-width", "2.5");
                        indGroup.appendChild(ind); l.elSymbol.appendChild(indGroup);
                    }

                    // A MÁGICA VISUAL DO SHUNT DA LINHA CORRIGIDO AQUI
                    if (l.bsh > 0) {
                        const len = Math.sqrt(dx*dx + dy*dy);
                        if(len > 0) {
                            const off1 = -len/2 + len * 0.25; 
                            const off2 = len/2 - len * 0.25; 
                            const getLocalShunt = (x) => {
                                return `<path d="M ${x} 0 L ${x} 15" stroke="#000" stroke-width="2" fill="none"/>
                                        <path d="M ${x-10} 15 L ${x+10} 15" stroke="#1976d2" stroke-width="3" fill="none"/>
                                        <path d="M ${x-10} 20 L ${x+10} 20" stroke="#1976d2" stroke-width="3" fill="none"/>
                                        <path d="M ${x} 20 L ${x} 35" stroke="#000" stroke-width="2" fill="none"/>
                                        <path d="M ${x-12} 35 L ${x+12} 35" stroke="#000" stroke-width="2" fill="none"/>
                                        <path d="M ${x-8} 40 L ${x+8} 40" stroke="#000" stroke-width="2" fill="none"/>
                                        <path d="M ${x-4} 45 L ${x+4} 45" stroke="#000" stroke-width="2" fill="none"/>
                                        <text x="${x+12}" y="25" font-size="12" font-weight="bold" fill="#1976d2">jbsh</text>`;
                            };
                            l.elSymbol.innerHTML += getLocalShunt(off1) + getLocalShunt(off2);
                        }
                    }
                    
                    l.elLabel.innerHTML = `Z: ${l.r} + j${l.x}`; l.elLabel.style.left = midX + "px"; l.elLabel.style.top = (midY - 40) + "px"; l.elLabel.style.transform = "translateX(-50%)";
                    if(selectedElement && selectedElement.id === l.id && selectedType === 'linha') l.elPath.classList.add("selected"); else l.elPath.classList.remove("selected");
                }

                function updateAllWires() { linhas.forEach(renderLinha); geradores.forEach(g => positionAttached(g, barras.find(b => b.id === g.barraId), 'gerador')); cargas.forEach(c => positionAttached(c, barras.find(b => b.id === c.barraId), 'carga')); }

                function makeDraggable(el, item, type) {
                    let isDragging = false, startX, startY;
                    el.addEventListener('mousedown', (e) => { e.stopPropagation(); isDragging = true; startX = e.clientX - item.x; startY = e.clientY - item.y; handleBarraClick(item); });
                    document.addEventListener('mousemove', (e) => {
                        if (!isDragging) return; const wsRect = ws.getBoundingClientRect();
                        item.x = Math.max(30, Math.min(wsRect.width - 30, e.clientX - startX)); item.y = Math.max(30, Math.min(wsRect.height - 30, e.clientY - startY));
                        el.style.left = item.x + "px"; el.style.top = item.y + "px"; updateAllWires();
                    });
                    document.addEventListener('mouseup', () => isDragging = false);
                }

                ws.addEventListener('mousedown', () => { if(!connectMode) selectElement(null, null); });

                function selectElement(item, type) {
                    selectedElement = item; selectedType = type;
                    document.querySelectorAll(".component.selected").forEach(el => el.classList.remove("selected")); document.querySelectorAll(".wire-path.selected").forEach(el => el.classList.remove("selected"));
                    panel.style.display = item ? "block" : "none"; if(!item) return;
                    
                    const content = document.getElementById("panel-content"); let html = "";
                    
                    if (type === 'barra') {
                        item.el.classList.add("selected"); document.getElementById("panel-title").innerText = `Barra ${item.id} (${item.type})`;
                        
                        const g = geradores.find(x => x.barraId === item.id);
                        const c = cargas.find(x => x.barraId === item.id);

                        html += `<button class="btn" style="width:100%; margin-bottom:15px; background:#f0f0f0;" onclick="rotateBarra()">↻ Girar a Barra (360º)</button>`;
                        
                        if(item.type === 'slack') {
                            html += `<div class="prop-group"><label>Módulo V (pu)</label><input type="number" step="0.01" value="${item.v}" onchange="updateProp('v', this.value)"></div>`;
                            html += `<div class="prop-group"><label>Ângulo θ (°)</label><input type="number" step="1" value="${item.theta}" onchange="updateProp('theta', this.value)"></div>`;
                        } 
                        
                        if(g) {
                            html += `<h5 style="margin:10px 0 5px 0; color:#d32f2f; border-bottom: 1px solid #ddd;">⚙️ Gerador</h5>`;
                            html += `<div class="prop-group"><label>Tensão V Fixa (pu)</label><input type="number" step="0.01" value="${g.v}" onchange="updateComponent('gerador', 'v', this.value)"></div>`;
                            html += `<div class="prop-group"><label>Potência Ativa Pg</label><input type="number" step="0.1" value="${g.p}" onchange="updateComponent('gerador', 'p', this.value)"></div>`;
                            html += `<div class="prop-group"><label>Potência Reativa Qg</label><input type="number" step="0.1" value="${g.q}" onchange="updateComponent('gerador', 'q', this.value)"></div>`;
                        }
                        if(c) {
                            html += `<h5 style="margin:10px 0 5px 0; color:#1976d2; border-bottom: 1px solid #ddd;">🔋 Carga</h5>`;
                            html += `<div class="prop-group"><label>Carga Ativa Pc</label><input type="number" step="0.1" value="${c.p}" onchange="updateComponent('carga', 'p', this.value)"></div>`;
                            html += `<div class="prop-group"><label>Carga Reativa Qc</label><input type="number" step="0.1" value="${c.q}" onchange="updateComponent('carga', 'q', this.value)"></div>`;
                        }
                        
                        html += `<hr style="margin: 10px 0;"><div class="prop-group"><label>Shunt na Barra (pu)</label><input type="number" step="0.01" value="${item.bsh_bus}" onchange="updateProp('bsh_bus', this.value)"></div>`;
                    } else if (type === 'linha') {
                        item.elPath.classList.add("selected"); document.getElementById("panel-title").innerText = `Linha (B${item.b1} ↔ B${item.b2})`;
                        html += `<div class="prop-group"><label>Resistência r (pu)</label><input type="number" step="0.01" value="${item.r}" onchange="updateProp('r', this.value)"></div>`;
                        html += `<div class="prop-group"><label>Reatância x (pu)</label><input type="number" step="0.01" value="${item.x}" onchange="updateProp('x', this.value)"></div>`;
                        html += `<div class="prop-group"><label>Susceptância bsh_linha (pu)</label><input type="number" step="0.01" value="${item.bsh}" onchange="updateProp('bsh', this.value)"></div>`;
                    } else if (type === 'gerador' || type === 'carga') { selectElement(barras.find(b => b.id === item.barraId), 'barra'); }
                    content.innerHTML = html;
                }

                function updateProp(key, value) { selectedElement[key] = parseFloat(value); if(selectedType === 'barra') renderBarra(selectedElement); if(selectedType === 'linha') updateAllWires(); }
                function updateComponent(compType, key, value) { 
                    if(compType === 'gerador') { 
                        geradores.find(x => x.barraId === selectedElement.id)[key] = parseFloat(value); 
                    } else { 
                        cargas.find(x => x.barraId === selectedElement.id)[key] = parseFloat(value); 
                    } 
                    renderBarra(selectedElement); 
                    selectElement(selectedElement, 'barra'); 
                }
                
                function deleteSelected() {
                    if(!selectedElement) return;
                    if(selectedType === 'barra') {
                        if(selectedElement.type === 'slack') { alert("A Barra Slack não pode ser excluída!"); return; }
                        linhas.filter(l => l.b1 === selectedElement.id || l.b2 === selectedElement.id).forEach(l => { if(l.elPath) l.elPath.remove(); if(l.elLabel) l.elLabel.remove(); if(l.elSymbol) l.elSymbol.remove(); });
                        linhas = linhas.filter(l => l.b1 !== selectedElement.id && l.b2 !== selectedElement.id);
                        geradores.filter(g => g.barraId === selectedElement.id).forEach(g => { g.el.remove(); g.elWire.remove(); }); geradores = geradores.filter(g => g.barraId !== selectedElement.id);
                        cargas.filter(c => c.barraId === selectedElement.id).forEach(c => { c.el.remove(); c.elWire.remove(); }); cargas = cargas.filter(c => c.barraId !== selectedElement.id);
                        selectedElement.el.remove(); barras = barras.filter(b => b.id !== selectedElement.id);
                    } else if(selectedType === 'linha') {
                        if(selectedElement.elPath) selectedElement.elPath.remove(); if(selectedElement.elLabel) selectedElement.elLabel.remove(); if(selectedElement.elSymbol) selectedElement.elSymbol.remove();
                        linhas = linhas.filter(l => l.id !== selectedElement.id);
                    }
                    selectElement(null, null); updateAllWires();
                }
                
                function exportarParaPython() {
                    const sistema = {
                        barras: barras.map(b => {
                            const g = geradores.find(x => x.barraId === b.id);
                            const c = cargas.find(x => x.barraId === b.id);
                            
                            let obj = { 
                                id: b.id, 
                                tipo: b.type, 
                                bsh_bus: b.bsh_bus,
                                p_ger: g ? g.p : 0,
                                q_ger: g ? g.q : 0,
                                p_carga: c ? c.p : 0,
                                q_carga: c ? c.q : 0
                            };
                            
                            if(b.type === 'slack') { obj.v = b.v; obj.theta = b.theta; } 
                            else if (g) { obj.v = g.v; obj.theta = 0; }
                            else { obj.v = 1.0; obj.theta = 0; }
                            
                            return obj;
                        }),
                        linhas: linhas.map(l => ({ de: l.b1, para: l.b2, r: l.r, x: l.x, bsh: l.bsh }))
                    };
                    sendMessageToStreamlit(sistema);
                }
            </script>
        </body>
        </html>
        """
        
        component_dir = os.path.abspath("canvas_sep_component")
        if not os.path.exists(component_dir):
            os.makedirs(component_dir, exist_ok=True)
        with open(os.path.join(component_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(html_canvas)

        componente_canvas = components.declare_component("canvas_sep", path=component_dir)
        dados_canvas_bruto = componente_canvas(key="meu_canvas")
        
        if dados_canvas_bruto is not None:
            dados_para_calculo = dados_canvas_bruto
            # Guardar os dados na sessão para estarem disponíveis no Modo Tabela
            st.session_state['sync_canvas_data'] = dados_canvas_bruto

    # ==========================================
    # PROCESSAMENTO DO MOTOR COMUM A AMBOS OS MODOS
    # ==========================================
    if dados_para_calculo is not None:
        st.markdown("---")
        
        unid_str = "MW" if unidade == "MW / MVar" else "pu"
        unid_q_str = "MVar" if unidade == "MW / MVar" else "pu"

        if len(dados_para_calculo.get('linhas', [])) == 0:
            st.warning("⚠️ O sistema necessita de pelo menos uma linha conectando os barramentos.")
        else:
            with st.spinner("Solucionando pelo Método de Newton-Raphson..."):
                backend_buses = []
                id_map = {b['id']: idx + 1 for idx, b in enumerate(dados_para_calculo['barras'])}
                
                for b in dados_para_calculo['barras']:
                    tipo_map = {'slack': 'Slack', 'pv': 'PV', 'pq': 'PQ'}
                    tipo_real = tipo_map[str(b['tipo']).lower().strip()]
                    
                    p_liquido_bruto = b['p_ger'] - b['p_carga']
                    q_liquido_bruto = b['q_ger'] - b['q_carga']
                    
                    p_liq_pu = p_liquido_bruto / divisor_potencia
                    q_liq_pu = q_liquido_bruto / divisor_potencia
                    
                    backend_buses.append({ 
                        "type": tipo_real, 
                        "V": float(b.get('v', 1.0)), 
                        "theta": float(b.get('theta', 0.0)), 
                        "P": float(p_liq_pu), 
                        "Q": float(q_liq_pu),
                        "Bsh_bus": float(b.get('bsh_bus', 0.0))
                    })

                backend_lines = []
                for l in dados_para_calculo['linhas']:
                    backend_lines.append({ 
                        "from": id_map[l['de']], 
                        "to": id_map[l['para']], 
                        "R": float(l['r']), 
                        "X": float(l['x']), 
                        "Bsh": float(l.get('bsh', 0.0)) 
                    })
                
                try:
                    Ybus = build_ybus(backend_buses, backend_lines)
                    V_final, theta_final, log_iteracoes, pvpq, pq_index, P_spec, Q_spec = newton_raphson(backend_buses, Ybus, tol=tol_input, max_iter=max_iter_input)
                    
                    st.success(f"✅ O sistema convergiu em {len(log_iteracoes)} iterações!" if log_iteracoes[-1]['convergiu'] else f"⚠️ Limite de {max_iter_input} iterações atingido sem convergência completa.")
                    
                    st.markdown("#### Resultados Finais nas Barras")
                    res_barras = []
                    for i, b in enumerate(dados_para_calculo['barras']):
                        res_barras.append({
                            "Barra": id_map[b['id']],
                            "Tipo": str(b['tipo']).upper(),
                            "Módulo |V| (pu)": f"{V_final[i]:.4f}",
                            "Ângulo θ (º)": f"{np.degrees(theta_final[i]):.4f}",
                        })
                    st.dataframe(pd.DataFrame(res_barras), hide_index=True)

                    st.markdown("---")
                    
                    # ==========================================
                    # SEÇÃO EDUCACIONAL: PASSO A PASSO
                    # ==========================================
                    st.markdown("<h2 style='text-align: center; color: #2e7d32;'>📚 Memória de Cálculo Analítica</h2>", unsafe_allow_html=True)
                    st.caption(f"*(Nota: O processamento interno do Newton-Raphson opera estritamente com as potências líquidas em p.u. adotando S_base = {base_mva} MVA).*")
                    
                    st.markdown("### 🔹 Matriz de Admitância Nodal ($Y_{bus}$)")
                    rotulos_y = [f"Barra {idx}" for idx in id_map.values()]
                    df_ybus = formatar_ybus(Ybus)
                    df_ybus.columns = rotulos_y
                    df_ybus.index = rotulos_y
                    st.dataframe(df_ybus)

                    st.markdown("### 🔹 Estado Inicial e Potências Injetadas Líquidas (em p.u.)")
                    col_ini1, col_ini2 = st.columns(2)
                    with col_ini1:
                        st.markdown("**Vetor de Estado Inicial ($\\nu = 0$)**")
                        st.latex(r"V^{(0)} = " + formatar_vetor_latex(log_iteracoes[0]['V_nu']))
                        st.latex(r"\theta^{(0)} = " + formatar_vetor_latex(log_iteracoes[0]['theta_nu']))
                    with col_ini2:
                        st.markdown("**Potências Específicas**")
                        st.latex(r"P^{esp} = " + formatar_vetor_latex(P_spec))
                        st.latex(r"Q^{esp} = " + formatar_vetor_latex(Q_spec))

                    st.markdown("### 🔹 Iteração $\\nu = 0$")
                    dados_iter0 = log_iteracoes[0]
                    
                    col_p0, col_q0 = st.columns(2)
                    with col_p0:
                        st.markdown("**Cálculo de Potência Ativa:**")
                        st.latex(r"P_{calc}^{(0)} = " + formatar_vetor_latex(dados_iter0['P_calc']))
                        st.latex(r"\Delta P^{(0)} = " + formatar_vetor_latex(dados_iter0['dP']))
                    with col_q0:
                        st.markdown("**Cálculo de Potência Reativa:**")
                        st.latex(r"Q_{calc}^{(0)} = " + formatar_vetor_latex(dados_iter0['Q_calc']))
                        st.latex(r"\Delta Q^{(0)} = " + formatar_vetor_latex(dados_iter0['dQ']))

                    st.markdown("**Teste de Convergência:**")
                    st.latex(r"\max \left\{ |\Delta P|, |\Delta Q| \right\} = " + f"{dados_iter0['erro']:.6f} \quad \text{{(Tolerância: }} {tol_input} \text{{)}}")
                    
                    if dados_iter0['convergiu']:
                        st.success("✅ O critério de parada foi atendido satisfatoriamente na avaliação inicial.")
                    else:
                        st.warning("⚠️ O critério de parada não foi atingido. O algoritmo avança para a avaliação da Matriz Jacobiana e o processo iterativo.")
                        st.markdown("---")
                        
                        st.markdown("### 🔹 Processo Iterativo")
                        
                        it_validas = [step for step in log_iteracoes if 'J' in step]
                        if it_validas:
                            iter_selecionada = st.selectbox(
                                "Selecione a Iteração ($\\nu$):", 
                                options=[step['nu'] for step in it_validas],
                                format_func=lambda x: f"Iteração {x} ➔ Indo para estado da iteração {x+1}"
                            )
                            
                            dados_iter = next(s for s in it_validas if s['nu'] == iter_selecionada)
                            nu = dados_iter['nu']
                            
                            st.markdown(f"#### 1. Derivadas Parciais calculadas na iteração {nu}")
                            col_j1, col_j2 = st.columns(2)
                            with col_j1:
                                if dados_iter['H'].size > 0:
                                    st.markdown("**Matriz H ($\\frac{\partial P}{\partial \\theta}$):**")
                                    st.dataframe(pd.DataFrame(dados_iter['H']).map(lambda x: f"{x:.4f}"))
                                if dados_iter['M'].size > 0:
                                    st.markdown("**Matriz M ($\\frac{\partial Q}{\partial \\theta}$):**")
                                    st.dataframe(pd.DataFrame(dados_iter['M']).map(lambda x: f"{x:.4f}"))
                            with col_j2:
                                if dados_iter['N'].size > 0:
                                    st.markdown("**Matriz N ($\\frac{\partial P}{\partial V}$):**")
                                    st.dataframe(pd.DataFrame(dados_iter['N']).map(lambda x: f"{x:.4f}"))
                                if dados_iter['L'].size > 0:
                                    st.markdown("**Matriz L ($\\frac{\partial Q}{\partial V}$):**")
                                    st.dataframe(pd.DataFrame(dados_iter['L']).map(lambda x: f"{x:.4f}"))

                            st.markdown(f"#### 2. Matriz Jacobiana Completa $J^{{({nu})}}$")
                            rotulos = [f"Δθ{idx+1}" for idx in pvpq] + [f"ΔV{idx+1}" for idx in pq_index]
                            df_jacob = pd.DataFrame(dados_iter['J']).map(lambda x: f"{x:.4f}")
                            df_jacob.columns = rotulos
                            df_jacob.index = rotulos
                            st.dataframe(df_jacob)

                            st.markdown("#### 3. Resolução do Sistema Linear e Atualização Vetorial")
                            col_d1, col_d2 = st.columns(2)
                            with col_d1:
                                st.markdown("**Vetor Incremental ($\\Delta x$):**")
                                st.latex(r"\begin{bmatrix} \Delta \theta^{(" + str(nu) + r")} \\ \Delta V^{(" + str(nu) + r")} \end{bmatrix} = \begin{bmatrix} " + 
                                         r" \\ ".join([f"{v:.4f}" for v in dados_iter['dtheta']]) + r" \\ " +
                                         r" \\ ".join([f"{v:.4f}" for v in dados_iter['dV']]) + r" \end{bmatrix}")
                            with col_d2:
                                st.markdown(f"**Novo Estado ($\\nu = {nu+1}$):**")
                                st.latex(r"\theta^{(" + str(nu+1) + r")} = " + formatar_vetor_latex(dados_iter['theta_prox']))
                                st.latex(r"V^{(" + str(nu+1) + r")} = " + formatar_vetor_latex(dados_iter['V_prox']))
                                
                except Exception as e:
                    st.error(f"❌ Erro matemático na simulação: {e}.")
