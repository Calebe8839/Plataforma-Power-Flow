import numpy as np


def build_jacobian(V, theta, Ybus, P_calc, Q_calc, pq_index, pv_index):
    """
    Monta a matriz Jacobiana do método de Newton-Raphson para fluxo de carga.

    A Jacobiana é particionada em quatro submatrizes:

        J = | H   N |     onde:
            | M   L |

        H = ∂P/∂θ   →  shape (n_pvpq  x  n_pvpq)
        N = ∂P/∂V   →  shape (n_pvpq  x  n_pq  )
        M = ∂Q/∂θ   →  shape (n_pq    x  n_pvpq)
        L = ∂Q/∂V   →  shape (n_pq    x  n_pq  )

    As barras PV aparecem apenas nas linhas/colunas de ∂P (H e N como coluna).
    A barra Slack é excluída de todas as submatrizes.

    Parâmetros
    ----------
    V        : np.ndarray, shape (n,) — magnitudes de tensão [pu]
    theta    : np.ndarray, shape (n,) — ângulos de tensão [rad]
    Ybus     : np.ndarray complexo, shape (n, n)
    P_calc   : np.ndarray, shape (n,) — potências ativas calculadas [pu]
               (retornadas por calc_mismatch — evita recomputação)
    Q_calc   : np.ndarray, shape (n,) — potências reativas calculadas [pu]
    pq_index : list of int — índices internos das barras PQ
    pv_index : list of int — índices internos das barras PV

    Retorna
    -------
    H, N, M, L : np.ndarray — as quatro submatrizes da Jacobiana

    Derivadas utilizadas
    --------------------
    Termos fora da diagonal (i ≠ k):
        H[i,k] =  V_i * V_k * ( G_ik * sin(θ_i-θ_k) - B_ik * cos(θ_i-θ_k) )
        N[i,k] =  V_i        * ( G_ik * cos(θ_i-θ_k) + B_ik * sin(θ_i-θ_k) )
        M[i,k] = -V_i * V_k * ( G_ik * cos(θ_i-θ_k) + B_ik * sin(θ_i-θ_k) )
        L[i,k] =  V_i        * ( G_ik * sin(θ_i-θ_k) - B_ik * cos(θ_i-θ_k) )

    Termos diagonais (i = k), expressos em forma fechada usando P_calc e Q_calc
    para evitar a dupla contagem que ocorre ao incluir m=i no somatório:
        H[i,i] = -Q_calc[i] - V_i² * B_ii
        N[i,i] =  P_calc[i] + V_i² * G_ii  (não normalizado por V_i)  [*]
        M[i,i] =  P_calc[i] - V_i² * G_ii
        L[i,i] =  Q_calc[i] - V_i² * B_ii  (não normalizado por V_i)  [*]

    [*] A Jacobiana é montada na forma não normalizada, ou seja, as colunas
        N e L correspondem a ∂P/∂|V| e ∂Q/∂|V| (não ∂P/∂(|V|/|V|)).
        A atualização no Newton-Raphson usa Δ|V| diretamente (forma absoluta),
        o que é consistente com esta escolha.
    """

    G = Ybus.real
    B = Ybus.imag

    n = len(V)

    # PV antes de PQ — ordem obrigatória para consistência com o vetor de
    # mismatch montado em calc_mismatch: [ ΔP[pvpq] ; ΔQ[pq] ]
    pvpq   = pv_index + pq_index
    npvpq  = len(pvpq)
    npq    = len(pq_index)

    # ------------------------------------------------------------------
    # Pré-computação vetorizada das diferenças angulares e produtos V_i*V_k
    # ------------------------------------------------------------------
    theta_diff = theta[:, None] - theta[None, :]   # shape (n, n)
    cos_diff   = np.cos(theta_diff)
    sin_diff   = np.sin(theta_diff)

    # Termos fora da diagonal para cada submatriz (shape n x n)
    # H_off[i,k] =  V_i*V_k*(G_ik*sin - B_ik*cos)
    # N_off[i,k] =  V_i    *(G_ik*cos + B_ik*sin)
    # M_off[i,k] = -V_i*V_k*(G_ik*cos + B_ik*sin)
    # L_off[i,k] =  V_i    *(G_ik*sin - B_ik*cos)
    VV  = V[:, None] * V[None, :]   # shape (n, n)

    H_full =  VV * (G * sin_diff - B * cos_diff)
    N_full =  V[:, None] * (G * cos_diff + B * sin_diff)
    M_full = -VV * (G * cos_diff + B * sin_diff)
    L_full =  V[:, None] * (G * sin_diff - B * cos_diff)

    # ------------------------------------------------------------------
    # Correção dos termos diagonais (forma fechada — sem dupla contagem)
    # ------------------------------------------------------------------
    # O cálculo vetorizado acima produz para i=k:
    #   H_full[i,i] = V_i² * (G_ii*sin(0) - B_ii*cos(0)) = -V_i² * B_ii
    # O valor correto é:  H[i,i] = -Q_calc[i] - V_i²*B_ii
    # Logo a correção diagonal é:  -Q_calc[i]
    #
    # Analogamente:
    #   N_full[i,i] = V_i*(G_ii*cos(0)+B_ii*sin(0)) =  V_i*G_ii
    #   Correto:      N[i,i] = P_calc[i] + V_i²*G_ii
    #   Correção:    +P_calc[i] + V_i*(V_i*G_ii - G_ii) → mais claro: substituir
    #
    # Para evitar ambiguidade, substituímos a diagonal inteira pelo valor correto:

    diag_idx = np.arange(n)

    H_full[diag_idx, diag_idx] = -Q_calc          - V**2 * B[diag_idx, diag_idx]
    N_full[diag_idx, diag_idx] =  P_calc           + V**2 * G[diag_idx, diag_idx]
    M_full[diag_idx, diag_idx] =  P_calc           - V**2 * G[diag_idx, diag_idx]
    L_full[diag_idx, diag_idx] =  Q_calc           - V**2 * B[diag_idx, diag_idx]

    # ------------------------------------------------------------------
    # Extração das submatrizes nas dimensões corretas (pvpq x pvpq, etc.)
    # ------------------------------------------------------------------
    pvpq_arr = np.array(pvpq,    dtype=int)
    pq_arr   = np.array(pq_index, dtype=int)

    H = H_full[np.ix_(pvpq_arr, pvpq_arr)]   # (n_pvpq x n_pvpq)
    N = N_full[np.ix_(pvpq_arr, pq_arr)]      # (n_pvpq x n_pq  )
    M = M_full[np.ix_(pq_arr,   pvpq_arr)]   # (n_pq   x n_pvpq)
    L = L_full[np.ix_(pq_arr,   pq_arr)]      # (n_pq   x n_pq  )

    return H, N, M, L