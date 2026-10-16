"""
Sistemas exemplo da plataforma Power Flow.

Cada sistema está no mesmo formato que o diagrama unifilar envia ao Python
(barras / linhas / transformadores), acrescido das coordenadas de desenho
(x, y em px do "mundo" de 2400 × 1600 do diagrama) e da rotação da barra.

Convenções (as mesmas da plataforma):
  - Potências em MW / MVAr, base de 100 MVA.
  - "bsh" das linhas = susceptância shunt TOTAL da linha (B/2 em cada extremidade).
  - Transformadores: modelo 1 : a com o tap "a" no lado "de" (barra k).
    ATENÇÃO ao importar dados do formato IEEE/MATPOWER: lá o tap τ é τ : 1 no
    lado "de", então a = 1 / τ.
  - Rotação da barra (rot): 0 = vertical (gerador à esquerda, carga à direita);
    1 = horizontal (gerador acima, carga abaixo); 2 = vertical espelhada;
    3 = horizontal espelhada (carga acima).

As constantes abaixo são carregadas uma única vez por processo do servidor e
compartilhadas por todas as sessões. Por isso, a plataforma NUNCA deve entregar
o dicionário original ao estudante: use sempre carregar_exemplo(), que devolve
uma cópia independente.
"""

import copy

EXEMPLOS = {
    # ------------------------------------------------------------------
    # Grainger & Stevenson — sistema de 4 barras
    # ------------------------------------------------------------------
    "gs4": {
        "nome": "Sistema de 4 barras (Grainger & Stevenson)",
        "fonte": "J. J. Grainger e W. D. Stevenson Jr., Power System Analysis, "
                 "McGraw-Hill, 1994, Exemplos 9.2–9.5 (pp. 337–338).",
        "descricao": "Rede de 230 kV com 4 barras, 4 linhas e 2 geradores "
                     "(Slack na barra 1 e PV na barra 4). Base de 100 MVA.",
        "base_mva": 100.0,
        "barras": [
            {"id": 1, "tipo": "slack", "v": 1.00, "theta": 0.0, "p_ger": 0.0, "q_ger": 0.0,
             "p_carga": 50.0, "q_carga": 30.99, "bsh_bus": 0.0, "x": 300, "y": 500, "rot": 0},
            {"id": 2, "tipo": "PQ", "v": 1.00, "theta": 0.0, "p_ger": 0.0, "q_ger": 0.0,
             "p_carga": 170.0, "q_carga": 105.35, "bsh_bus": 0.0, "x": 750, "y": 220, "rot": 0},
            {"id": 3, "tipo": "PQ", "v": 1.00, "theta": 0.0, "p_ger": 0.0, "q_ger": 0.0,
             "p_carga": 200.0, "q_carga": 123.94, "bsh_bus": 0.0, "x": 750, "y": 780, "rot": 0},
            {"id": 4, "tipo": "PV", "v": 1.02, "theta": 0.0, "p_ger": 318.0, "q_ger": 0.0,
             "p_carga": 80.0, "q_carga": 49.58, "bsh_bus": 0.0, "x": 1200, "y": 500, "rot": 2},
        ],
        "linhas": [
            {"de": 1, "para": 2, "r": 0.01008, "x": 0.05040, "bsh": 0.1025},
            {"de": 1, "para": 3, "r": 0.00744, "x": 0.03720, "bsh": 0.0775},
            {"de": 2, "para": 4, "r": 0.00744, "x": 0.03720, "bsh": 0.0775},
            {"de": 3, "para": 4, "r": 0.01272, "x": 0.06360, "bsh": 0.1275},
        ],
        "transformadores": [],
        # Solução publicada (usada só na verificação automática; não é exibida na plataforma).
        "referencia": {
            "V": [1.000, 0.982, 0.969, 1.020],
            "theta_graus": [0.000, -0.976, -1.872, 1.523],
        },
    },

    # ------------------------------------------------------------------
    # Stagg & El-Abiad — sistema de 5 barras
    # ------------------------------------------------------------------
    "stagg5": {
        "nome": "Sistema de 5 barras (Stagg & El-Abiad)",
        "fonte": "G. W. Stagg e A. H. El-Abiad, Computer Methods in Power System "
                 "Analysis, McGraw-Hill, 1968.",
        "descricao": "Rede com 5 barras e 7 linhas. Barra 1 Slack (1,06 pu) e barra 2 com "
                     "geração de potência especificada (40 MW e 30 MVAr), tratada como PQ. "
                     "Base de 100 MVA.",
        "base_mva": 100.0,
        "barras": [
            {"id": 1, "tipo": "slack", "v": 1.06, "theta": 0.0, "p_ger": 0.0, "q_ger": 0.0,
             "p_carga": 0.0, "q_carga": 0.0, "bsh_bus": 0.0, "x": 250, "y": 250, "rot": 0},
            {"id": 2, "tipo": "PQ", "v": 1.00, "theta": 0.0, "p_ger": 40.0, "q_ger": 30.0,
             "p_carga": 20.0, "q_carga": 10.0, "bsh_bus": 0.0, "x": 900, "y": 250, "rot": 1},
            {"id": 3, "tipo": "PQ", "v": 1.00, "theta": 0.0, "p_ger": 0.0, "q_ger": 0.0,
             "p_carga": 45.0, "q_carga": 15.0, "bsh_bus": 0.0, "x": 575, "y": 650, "rot": 2},
            {"id": 4, "tipo": "PQ", "v": 1.00, "theta": 0.0, "p_ger": 0.0, "q_ger": 0.0,
             "p_carga": 40.0, "q_carga": 5.0, "bsh_bus": 0.0, "x": 1225, "y": 650, "rot": 0},
            {"id": 5, "tipo": "PQ", "v": 1.00, "theta": 0.0, "p_ger": 0.0, "q_ger": 0.0,
             "p_carga": 60.0, "q_carga": 10.0, "bsh_bus": 0.0, "x": 1440, "y": 395, "rot": 0},
        ],
        # O livro tabela a susceptância de carregamento por extremidade (Y'/2);
        # aqui "bsh" é o total da linha, portanto o dobro do valor do livro.
        "linhas": [
            {"de": 1, "para": 2, "r": 0.02, "x": 0.06, "bsh": 0.060},
            {"de": 1, "para": 3, "r": 0.08, "x": 0.24, "bsh": 0.050},
            {"de": 2, "para": 3, "r": 0.06, "x": 0.18, "bsh": 0.040},
            {"de": 2, "para": 4, "r": 0.06, "x": 0.18, "bsh": 0.040},
            {"de": 2, "para": 5, "r": 0.04, "x": 0.12, "bsh": 0.030},
            {"de": 3, "para": 4, "r": 0.01, "x": 0.03, "bsh": 0.020},
            {"de": 4, "para": 5, "r": 0.08, "x": 0.24, "bsh": 0.050},
        ],
        "transformadores": [],
        "referencia": {
            "V": [1.060, 1.047, 1.024, 1.024, 1.018],
            "theta_graus": [0.000, -2.807, -4.997, -5.329, -6.150],
        },
    },
}


def opcoes_exemplos():
    """Lista (chave, nome) na ordem de exibição."""
    return [(chave, ex["nome"]) for chave, ex in EXEMPLOS.items()]


def carregar_exemplo(chave):
    """Cópia independente do sistema escolhido: o estudante pode editar à vontade
    sem alterar o exemplo das outras sessões."""
    ex = EXEMPLOS[chave]
    return {
        "barras": copy.deepcopy(ex["barras"]),
        "linhas": copy.deepcopy(ex["linhas"]),
        "transformadores": copy.deepcopy(ex["transformadores"]),
    }