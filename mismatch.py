import numpy as np


def calc_power(V, theta, Ybus):
    """
    Calcula as potências ativa e reativa injetadas em cada barra.

    Utiliza a formulação retangular vetorizada da matriz Ybus, equivalente
    às equações clássicas do fluxo de carga:

        P_i = Σ_k  V_i * V_k * [ G_ik * cos(θ_i - θ_k) + B_ik * sin(θ_i - θ_k) ]
        Q_i = Σ_k  V_i * V_k * [ G_ik * sin(θ_i - θ_k) - B_ik * cos(θ_i - θ_k) ]

    Parâmetros
    ----------
    V     : np.ndarray, shape (n,) — magnitudes de tensão [pu]
    theta : np.ndarray, shape (n,) — ângulos de tensão [rad]
    Ybus  : np.ndarray complexo, shape (n, n) — matriz de admitâncias nodais

    Retorna
    -------
    P_calc : np.ndarray, shape (n,) — potência ativa calculada [pu]
    Q_calc : np.ndarray, shape (n,) — potência reativa calculada [pu]
    """
    G = Ybus.real
    B = Ybus.imag

    # Matriz de diferenças angulares θ_i - θ_k  →  shape (n, n)
    theta_diff = theta[:, None] - theta[None, :]

    # Produto V_i * V_k para todos os pares  →  shape (n, n)
    VV = V[:, None] * V[None, :]

    P_calc = np.sum(VV * (G * np.cos(theta_diff) + B * np.sin(theta_diff)), axis=1)
    Q_calc = np.sum(VV * (G * np.sin(theta_diff) - B * np.cos(theta_diff)), axis=1)

    return P_calc, Q_calc


def calc_mismatch(V, theta, Ybus, P_spec, Q_spec, pvpq, pq_index):
    """
    Calcula o vetor de resíduos (mismatch) do método de Newton-Raphson.

    O vetor de mismatch é formado apenas pelas barras e equações que
    possuem incógnitas ativas no método:

        ΔP_i = P_spec_i - P_calc_i   →  para todas as barras PV e PQ
        ΔQ_i = Q_spec_i - Q_calc_i   →  somente para barras PQ

    A barra Slack é excluída de ΔP e ΔQ (tensão e ângulo são conhecidos).
    As barras PV são excluídas de ΔQ (Q é incógnita, não restrição).

    Parâmetros
    ----------
    V       : np.ndarray, shape (n,) — magnitudes de tensão [pu]
    theta   : np.ndarray, shape (n,) — ângulos de tensão [rad]
    Ybus    : np.ndarray complexo, shape (n, n)
    P_spec  : np.ndarray, shape (n,) — potências ativas especificadas [pu]
    Q_spec  : np.ndarray, shape (n,) — potências reativas especificadas [pu]
    pvpq    : list of int — índices internos das barras PV + PQ (nesta ordem)
    pq_index: list of int — índices internos das barras PQ

    Retorna
    -------
    mismatch : np.ndarray, shape (len(pvpq) + len(pq_index),)
               Vetor [ ΔP[pvpq] ; ΔQ[pq_index] ]
    P_calc   : np.ndarray, shape (n,) — potência ativa calculada (todas as barras)
    Q_calc   : np.ndarray, shape (n,) — potência reativa calculada (todas as barras)
    """
    P_calc, Q_calc = calc_power(V, theta, Ybus)

    dP = P_spec - P_calc
    dQ = Q_spec - Q_calc

    mismatch = np.concatenate([dP[pvpq], dQ[pq_index]])

    return mismatch, P_calc, Q_calc