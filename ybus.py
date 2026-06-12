import numpy as np


def build_ybus(buses, lines, transformers=None):
    """
    Monta a matriz de admitâncias nodais Ybus.

    Parâmetros
    ----------
    buses : list of dict
        Lista de barras. Cada dicionário deve conter:
            - "id"      : int  — identificador da barra (usado apenas para
                                  rastreabilidade; o índice interno é 0..n-1
                                  na ordem em que as barras aparecem na lista)
            - "Bsh_bus" : float — susceptância shunt da barra [pu] (opcional,
                                  default 0.0)

    lines : list of dict
        Lista de linhas de transmissão (modelo π). Cada dicionário deve conter:
            - "from" : int — índice interno da barra de origem  (1-based)
            - "to"   : int — índice interno da barra de destino (1-based)
            - "R"    : float — resistência série [pu]
            - "X"    : float — reatância série  [pu]
            - "Bsh"  : float — susceptância shunt total da linha [pu]
                               (será dividida igualmente entre os dois nós)

    transformers : list of dict | None
        Lista de transformadores em-fase (modelo π equivalente, Fig. 1.5 do
        livro de referência). Cada dicionário deve conter:
            - "from" : int   — índice interno da barra k (lado do tap, 1-based)
            - "to"   : int   — índice interno da barra m (lado oposto, 1-based)
            - "R"    : float — resistência série do transformador [pu]
            - "X"    : float — reatância série do transformador   [pu]
            - "a"    : float — relação de transformação (tap); a = V_k / V_m
                               Para tap nominal use a = 1.0.

    Retorna
    -------
    Ybus : np.ndarray complexo, shape (n, n)

    Notas sobre o modelo π do transformador em-fase
    ------------------------------------------------
    Baseado na Seção 1.2.2 do livro "Fluxo de Carga em Redes de Energia
    Elétrica" (referência do projeto). Para relação de transformação 1:a
    e admitância série y_km = 1 / (R + jX):

        A =  a      * y_km   →  admitância série entre k e m
        B =  a(a-1) * y_km   →  shunt na barra k
        C = (1-a)   * y_km   →  shunt na barra m

    Contribuições na Ybus:
        Y[k,k] +=  A + B  =  a² * y_km
        Y[m,m] +=  A + C  =       y_km
        Y[k,m] -= A       = -a  * y_km
        Y[m,k] -= A       = -a  * y_km
    """

    n = len(buses)
    Ybus = np.zeros((n, n), dtype=complex)

    # ------------------------------------------------------------------
    # 1. Contribuição das Linhas de Transmissão (série + shunt π)
    # ------------------------------------------------------------------
    for line in lines:
        i = line["from"] - 1
        j = line["to"]   - 1

        # Validação de índices
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(
                f"Linha com índices fora do intervalo: from={line['from']}, "
                f"to={line['to']} (total de barras = {n})."
            )

        Z = complex(line["R"], line["X"])

        # Impedância série nula → linha ideal sem impedância é fisicamente
        # inválida no modelo π; lança erro descritivo.
        if abs(Z) < 1e-12:
            raise ValueError(
                f"Impedância série nula na linha {line['from']}→{line['to']}. "
                "Verifique os parâmetros R e X."
            )

        y_serie = 1.0 / Z
        b_shunt_metade = complex(0.0, line["Bsh"] / 2.0)

        Ybus[i, i] += y_serie + b_shunt_metade
        Ybus[j, j] += y_serie + b_shunt_metade
        Ybus[i, j] -= y_serie
        Ybus[j, i] -= y_serie

    # ------------------------------------------------------------------
    # 2. Contribuição dos Transformadores em-Fase (modelo π equivalente)
    # ------------------------------------------------------------------
    if transformers:
        for trafo in transformers:
            k = trafo["from"] - 1
            m = trafo["to"]   - 1

            # Validação de índices
            if not (0 <= k < n and 0 <= m < n):
                raise ValueError(
                    f"Transformador com índices fora do intervalo: "
                    f"from={trafo['from']}, to={trafo['to']} "
                    f"(total de barras = {n})."
                )

            Z_t = complex(trafo["R"], trafo["X"])

            if abs(Z_t) < 1e-12:
                raise ValueError(
                    f"Impedância série nula no transformador "
                    f"{trafo['from']}→{trafo['to']}. "
                    "Verifique os parâmetros R e X."
                )

            a = float(trafo["a"])   # relação de transformação (tap)

            if abs(a) < 1e-6:
                raise ValueError(
                    f"Relação de transformação 'a' inválida (≈ 0) no "
                    f"transformador {trafo['from']}→{trafo['to']}."
                )

            y_km = 1.0 / Z_t       # admitância série do transformador

            # Admitâncias do circuito π equivalente (eq. 1.15 do livro)
            #   A = a * y_km          → ramo série k–m
            #   B = a*(a-1) * y_km    → shunt na barra k
            #   C = (1-a)   * y_km    → shunt na barra m
            #
            # Contribuições na Ybus:
            #   Y[k,k] += A + B = a² * y_km
            #   Y[m,m] += A + C =      y_km
            #   Y[k,m] -= A     = -a * y_km
            #   Y[m,k] -= A     = -a * y_km

            Ybus[k, k] += (a ** 2)  * y_km
            Ybus[m, m] +=             y_km
            Ybus[k, m] -= a         * y_km
            Ybus[m, k] -= a         * y_km

    # ------------------------------------------------------------------
    # 3. Contribuição dos Shunts de Barra (capacitores / reatores de barra)
    # ------------------------------------------------------------------
    for idx, b in enumerate(buses):
        Bsh_B = b.get("Bsh_bus", 0.0)
        Ybus[idx, idx] += complex(0.0, Bsh_B)

    return Ybus