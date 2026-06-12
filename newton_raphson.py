import numpy as np
from mismatch import calc_mismatch
from jacobian import build_jacobian


def newton_raphson(buses, Ybus, tol=1e-6, max_iter=20):
    """
    Resolve o problema de fluxo de carga pelo Método de Newton-Raphson.

    Parâmetros
    ----------
    buses    : list of dict
        Lista de barras na mesma ordem usada para montar a Ybus. Cada
        dicionário deve conter:
            - "type"  : str   — "Slack", "PV" ou "PQ"
            - "V"     : float — magnitude de tensão inicial [pu]
            - "theta" : float — ângulo inicial [graus] (apenas Slack)
            - "P"     : float — potência ativa líquida especificada [pu]
                                (Slack: ignorado; PV e PQ: geração − carga)
            - "Q"     : float — potência reativa líquida especificada [pu]
                                (Slack e PV: ignorado; PQ: geração − carga)
    Ybus     : np.ndarray complexo, shape (n, n)
    tol      : float — tolerância de convergência (erro máximo no mismatch)
    max_iter : int   — número máximo de iterações

    Retorna
    -------
    V                : np.ndarray, shape (n,) — magnitudes finais [pu]
    theta            : np.ndarray, shape (n,) — ângulos finais [rad]
    historico_passos : list of dict — log completo de cada iteração
    pvpq             : list of int  — índices internos PV + PQ
    pq_index         : list of int  — índices internos PQ
    P_spec           : np.ndarray, shape (n,) — potências ativas especificadas
    Q_spec           : np.ndarray, shape (n,) — potências reativas especificadas
    convergiu        : bool — True se o critério de parada foi atingido

    Notas
    -----
    - theta é tratado internamente em radianos. A interface deve converter
      graus → radianos na entrada e radianos → graus na saída.
    - A atualização de V é feita na forma absoluta (ΔV), consistente com
      a Jacobiana não normalizada montada em build_jacobian.
    - Q_spec das barras PV é zerado internamente: Q é incógnita nessas
      barras, não uma restrição do sistema.
    - Se a Jacobiana for singular, uma exceção descritiva é lançada em vez
      de silenciosamente aplicar mínimos quadrados.
    """

    n = len(buses)

    # ------------------------------------------------------------------
    # 1. Inicialização das variáveis de estado e classificação das barras
    # ------------------------------------------------------------------
    V      = np.ones(n)
    theta  = np.zeros(n)
    P_spec = np.zeros(n)
    Q_spec = np.zeros(n)

    slack_index = None
    pv_index    = []
    pq_index    = []

    for i, b in enumerate(buses):
        tipo = str(b["type"]).strip()

        if tipo == "Slack":
            slack_index = i
            V[i]     = float(b["V"])
            theta[i] = np.radians(float(b["theta"]))
            # P e Q da Slack são calculados após convergência — não especificados

        elif tipo == "PV":
            pv_index.append(i)
            V[i]     = float(b["V"])   # magnitude fixa
            P_spec[i] = float(b["P"])
            # Q_spec[i] permanece 0.0 — Q é incógnita em barras PV;
            # não deve entrar como restrição no vetor de mismatch

        elif tipo == "PQ":
            pq_index.append(i)
            P_spec[i] = float(b["P"])
            Q_spec[i] = float(b["Q"])

        else:
            raise ValueError(
                f"Tipo de barra desconhecido: '{tipo}' (barra índice {i}). "
                "Use 'Slack', 'PV' ou 'PQ'."
            )

    # Validação: exatamente uma barra Slack é obrigatória
    if slack_index is None:
        raise ValueError(
            "Nenhuma barra Slack encontrada. "
            "O sistema requer exatamente uma barra de referência (Slack)."
        )

    # Ordem obrigatória: PV antes de PQ — deve ser idêntica à usada em
    # calc_mismatch e build_jacobian para garantir correspondência de índices.
    pvpq = pv_index + pq_index

    historico_passos = []

    # ------------------------------------------------------------------
    # 2. Loop iterativo de Newton-Raphson
    # ------------------------------------------------------------------
    for iteration in range(max_iter):

        # Salva estado no início desta iteração (antes de qualquer atualização)
        V_atual     = V.copy()
        theta_atual = theta.copy()

        # --- 2a. Cálculo do mismatch -----------------------------------
        mismatch, P_calc, Q_calc = calc_mismatch(
            V, theta, Ybus, P_spec, Q_spec, pvpq, pq_index
        )

        erro      = np.max(np.abs(mismatch))
        convergiu = erro < tol

        passo_info = {
            'nu'       : iteration,
            'V_nu'     : V_atual,
            'theta_nu' : theta_atual,
            'P_calc'   : P_calc.copy(),
            'Q_calc'   : Q_calc.copy(),
            'dP'       : (P_spec - P_calc).copy(),
            'dQ'       : (Q_spec - Q_calc).copy(),
            'mismatch' : mismatch.copy(),
            'erro'     : erro,
            'convergiu': convergiu,
        }

        # --- 2b. Teste de convergência ---------------------------------
        if convergiu:
            historico_passos.append(passo_info)
            return (V, theta, historico_passos,
                    pvpq, pq_index, P_spec, Q_spec, True)

        # --- 2c. Montagem da Jacobiana ---------------------------------
        # P_calc e Q_calc são repassados para evitar recomputação interna.
        H, N, M, L = build_jacobian(
            V, theta, Ybus, P_calc, Q_calc, pq_index, pv_index
        )
        J = np.block([[H, N], [M, L]])

        passo_info['J'] = J
        passo_info['H'] = H
        passo_info['N'] = N
        passo_info['M'] = M
        passo_info['L'] = L

        # --- 2d. Resolução do sistema linear  J · Δx = mismatch -------
        try:
            dx = np.linalg.solve(J, mismatch)
        except np.linalg.LinAlgError:
            raise RuntimeError(
                f"Jacobiana singular na iteração {iteration}. "
                "Possíveis causas: rede desconectada, ilha sem barra Slack, "
                "ou parâmetros de linha/transformador inconsistentes."
            )

        # --- 2e. Extração e aplicação das correções -------------------
        n_ang  = len(pvpq)
        dtheta = dx[:n_ang]        # correções angulares [rad]
        dV     = dx[n_ang:]        # correções de magnitude [pu] (forma absoluta)

        for idx, bus_idx in enumerate(pvpq):
            theta[bus_idx] += dtheta[idx]

        for idx, bus_idx in enumerate(pq_index):
            V[bus_idx] += dV[idx]

        # Garante que magnitudes de tensão não se tornem negativas ou nulas
        # (proteção numérica — valores fisicamente inválidos indicam divergência)
        if np.any(V <= 0):
            raise RuntimeError(
                f"Magnitude de tensão não positiva detectada na iteração "
                f"{iteration}. O método divergiu. Verifique os dados de entrada."
            )

        passo_info['dtheta']    = dtheta
        passo_info['dV']        = dV
        passo_info['V_prox']    = V.copy()
        passo_info['theta_prox']= theta.copy()

        historico_passos.append(passo_info)

    # ------------------------------------------------------------------
    # 3. Limite de iterações atingido sem convergência
    # ------------------------------------------------------------------
    return (V, theta, historico_passos,
            pvpq, pq_index, P_spec, Q_spec, False)