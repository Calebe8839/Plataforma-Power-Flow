"""
Redes de sequência (positiva, negativa e zero) para estudos de curto-circuito.

Dados extras (todos OPCIONAIS no sistema; o fluxo de carga ignora-os):
  barra   : fonte_cc (bool), x1 (X''d), x2, x0, xn (reator de neutro), aterrada (bool), vbase_kv
  linha   : r0, x0   (sequência zero; padrão = 3 × r, 3 × x)
  trafo   : ligacao (ver LIGACOES), r0, x0, xn_k, xn_m   (sequência zero; padrão = r, x)
Todos os valores em pu, na base do sistema (a mesma do fluxo de carga).

Convenção de ligação do transformador: "<lado k>-<lado m>", com k = barra "de"
(lado do tap). "Yg" = estrela aterrada, "Y" = estrela isolada, "D" = delta.

Modelos de pré-falta:
  "classico"    : Vf = 1∠0, sem cargas, sem shunts, taps nominais (a = 1).
  "fluxo_carga" : Vf = tensões do fluxo de carga; cargas como impedância
                  constante; shunts e taps mantidos como no fluxo de carga.
"""
import copy
from dataclasses import dataclass, field

import numpy as np

from ybus import build_ybus

LIGACOES = ("Yg-Yg", "Yg-D", "D-Yg", "Y-D", "D-Y", "D-D", "Y-Y", "Yg-Y", "Y-Yg")

# Valores didáticos usados só quando o estudante não informa nada.
PADRAO = {"x1": 0.15, "razao_x2": 1.0, "razao_x0": 0.5, "razao_linha_zero": 3.0, "ligacao": "Yg-D"}

_EPS = 1e-12


# ----------------------------------------------------------------------
# Dados: preenchimento e validação
# ----------------------------------------------------------------------
def tem_fonte(b):
    """Slack, PV e barras PQ com geração especificada são fontes de curto-circuito."""
    if "fonte_cc" in b:
        return bool(b["fonte_cc"])
    return str(b["tipo"]).lower() != "pq" or float(b.get("p_ger", 0.0)) != 0.0


def completar_dados_cc(sistema):
    """Cópia do sistema com os campos de CC ausentes preenchidos com valores didáticos.
    Retorna (sistema_completo, avisos); 'avisos' diz o que foi preenchido, para a tela mostrar."""
    s = copy.deepcopy(sistema)
    avisos = []
    for b in s["barras"]:
        b["fonte_cc"] = tem_fonte(b)
        if tem_fonte(b):
            if "x1" not in b:
                b["x1"] = PADRAO["x1"]; avisos.append(f"Barra {b['id']}: X''d = {PADRAO['x1']} pu (valor didático)")
            b.setdefault("x2", b["x1"] * PADRAO["razao_x2"])
            b.setdefault("x0", b["x1"] * PADRAO["razao_x0"])
            b.setdefault("xn", 0.0)
            b.setdefault("aterrada", True)
    for l in s.get("linhas", []):
        if "x0" not in l:
            l["r0"] = l.get("r0", PADRAO["razao_linha_zero"] * l["r"])
            l["x0"] = PADRAO["razao_linha_zero"] * l["x"]
    for t in s.get("transformadores", []):
        if "ligacao" not in t:
            t["ligacao"] = PADRAO["ligacao"]; avisos.append(f"Trafo {t['de']}→{t['para']}: ligação {PADRAO['ligacao']} (valor didático)")
        t.setdefault("r0", t["r"]); t.setdefault("x0", t["x"])
        t.setdefault("xn_k", 0.0); t.setdefault("xn_m", 0.0)
    return s, avisos


def validar_cc(sistema):
    """Lista de mensagens de erro em português (lista vazia = pode calcular)."""
    erros = []
    ids = [b["id"] for b in sistema["barras"]]
    fontes = [b for b in sistema["barras"] if tem_fonte(b)]
    if not fontes:
        erros.append("Nenhuma fonte de curto-circuito: marque ao menos um gerador (Slack, PV ou barra com geração).")
    for b in fontes:
        if not float(b.get("x1", 0.0)) > 0:
            erros.append(f"Barra {b['id']}: informe a reatância subtransitória X''d (> 0).")
    for t in sistema.get("transformadores", []):
        if t.get("ligacao", PADRAO["ligacao"]) not in LIGACOES:
            erros.append(f"Trafo {t['de']}→{t['para']}: ligação '{t.get('ligacao')}' inválida. Use uma de: {', '.join(LIGACOES)}.")
    for el in list(sistema.get("linhas", [])) + list(sistema.get("transformadores", [])):
        if el["de"] not in ids or el["para"] not in ids:
            erros.append(f"Ramo {el['de']}→{el['para']} liga barra inexistente.")
    return erros


# ----------------------------------------------------------------------
# Montagem das redes
# ----------------------------------------------------------------------
@dataclass
class Redes:
    ids: list                      # número das barras, na ordem das matrizes
    Y1: np.ndarray
    Y2: np.ndarray
    Y0: np.ndarray
    Z1: np.ndarray
    Z2: np.ndarray
    Z0: np.ndarray                 # = inf (bloco) em barras sem caminho de sequência zero à terra
    Vf: np.ndarray                 # tensão pré-falta complexa por barra
    sem_terra_zero: list = field(default_factory=list)   # ids das barras sem caminho de sequência zero
    modelo: str = "classico"


def _rotulos_componentes(Y):
    """Rótulo de componente conexo de cada barra, pelas ligações fora da diagonal."""
    n = Y.shape[0]
    adj = np.abs(Y - np.diag(np.diag(Y))) > _EPS
    rot = -np.ones(n, dtype=int); c = 0
    for s in range(n):
        if rot[s] >= 0:
            continue
        pilha = [s]; rot[s] = c
        while pilha:
            u = pilha.pop()
            for v in np.nonzero(adj[u])[0]:
                if rot[v] < 0:
                    rot[v] = c; pilha.append(v)
        c += 1
    return rot, c


def _inversa_positiva(Y, ids, nome):
    """Z = inv(Y) exigindo ao menos uma fonte em cada ilha (senão Y é singular)."""
    try:
        return np.linalg.inv(Y)
    except np.linalg.LinAlgError:
        raise ValueError(f"Rede de sequência {nome} singular: há barras sem ligação a nenhuma fonte. "
                         "Verifique ilhas sem gerador.")


def montar_redes(sistema, modelo="classico", resultado_fc=None, divisor_potencia=100.0):
    """Monta Y e Z das três sequências. 'sistema' deve estar completo (ver completar_dados_cc)."""
    erros = validar_cc(sistema)
    if erros:
        raise ValueError(" | ".join(erros))
    classico = modelo == "classico"
    if not classico and resultado_fc is None:
        raise ValueError("O modelo 'fluxo_carga' precisa do resultado do fluxo de carga (resultado_fc).")

    barras = sistema["barras"]; n = len(barras)
    ids = [b["id"] for b in barras]; pos = {i: k for k, i in enumerate(ids)}

    # --- Sequências 1 e 2: rede passiva (mesmo build_ybus do fluxo de carga) + fontes + cargas
    buses = [{"Bsh_bus": 0.0 if classico else float(b.get("bsh_bus", 0.0))} for b in barras]
    lines = [{"from": pos[l["de"]] + 1, "to": pos[l["para"]] + 1, "R": float(l["r"]), "X": float(l["x"]),
              "Bsh": 0.0 if classico else float(l.get("bsh", 0.0))} for l in sistema.get("linhas", [])]
    trafos = [{"from": pos[t["de"]] + 1, "to": pos[t["para"]] + 1, "R": float(t["r"]), "X": float(t["x"]),
               "a": 1.0 if classico else float(t["a"])} for t in sistema.get("transformadores", [])]
    Ypas = build_ybus(buses, lines, trafos if trafos else None)

    if not classico:
        Vm = np.asarray(resultado_fc["V"], float)
        for k, b in enumerate(barras):
            S = complex(b.get("p_carga", 0.0), b.get("q_carga", 0.0)) / divisor_potencia
            Ypas[k, k] += np.conj(S) / Vm[k] ** 2          # carga como impedância constante

    # Toda ilha precisa de ao menos uma fonte; senão Y1/Y2 ficam singulares (e numericamente
    # a inversão pode "passar" com valores enormes, por isso a checagem é explícita).
    rot_p, nc_p = _rotulos_componentes(Ypas)
    for c in range(nc_p):
        idx = np.nonzero(rot_p == c)[0]
        if not any(tem_fonte(barras[i]) for i in idx):
            raise ValueError("Barra(s) " + ", ".join(str(ids[i]) for i in idx) +
                             " sem ligação a nenhuma fonte: a rede de sequência positiva fica singular. "
                             "Conecte-as a um gerador ou marque um gerador nelas.")

    Y1 = Ypas.copy(); Y2 = Ypas.copy()
    for k, b in enumerate(barras):
        if tem_fonte(b):
            Y1[k, k] += 1.0 / (1j * float(b["x1"]))
            Y2[k, k] += 1.0 / (1j * float(b["x2"]))
    Z1 = _inversa_positiva(Y1, ids, "positiva")
    Z2 = _inversa_positiva(Y2, ids, "negativa")

    # --- Sequência zero: ramos em série + caminhos à terra
    Y0 = np.zeros((n, n), dtype=complex)
    ygnd = np.zeros(n, dtype=complex)                       # admitância à terra de cada barra
    def serie(i, j, z):
        if abs(z) < _EPS:
            raise ValueError("Impedância de sequência zero nula em um ramo. Verifique r0 e x0.")
        y = 1.0 / z
        Y0[i, i] += y; Y0[j, j] += y; Y0[i, j] -= y; Y0[j, i] -= y
    for l in sistema.get("linhas", []):
        serie(pos[l["de"]], pos[l["para"]], complex(l["r0"], l["x0"]))
    for t in sistema.get("transformadores", []):
        k, m = pos[t["de"]], pos[t["para"]]
        z0 = complex(t["r0"], t["x0"]); lig = t["ligacao"]
        zk, zm = 3j * t.get("xn_k", 0.0), 3j * t.get("xn_m", 0.0)
        lado_k, lado_m = lig.split("-")
        if lado_k == "Yg" and lado_m == "Yg":
            serie(k, m, z0 + zk + zm)
        elif lado_k == "Yg" and lado_m == "D":
            ygnd[k] += 1.0 / (z0 + zk)
        elif lado_k == "D" and lado_m == "Yg":
            ygnd[m] += 1.0 / (z0 + zm)
        # demais ligações: sem caminho de sequência zero entre o trafo e a rede
    for k, b in enumerate(barras):
        if tem_fonte(b) and b.get("aterrada", True):
            ygnd[k] += 1.0 / (1j * (float(b["x0"]) + 3.0 * float(b.get("xn", 0.0))))
    Y0 += np.diag(ygnd)

    # Ilhas de sequência zero sem nenhum caminho à terra: Z0 = ∞ nessas barras
    rot, nc = _rotulos_componentes(Y0)
    Z0 = np.zeros((n, n), dtype=complex); sem_terra = []
    for c in range(nc):
        idx = np.nonzero(rot == c)[0]
        if np.any(np.abs(ygnd[idx]) > _EPS):
            Z0[np.ix_(idx, idx)] = np.linalg.inv(Y0[np.ix_(idx, idx)])
        else:
            Z0[np.ix_(idx, idx)] = np.inf
            sem_terra += [ids[i] for i in idx]

    if classico:
        Vf = np.ones(n, dtype=complex)
    else:
        Vf = np.asarray(resultado_fc["V"], float) * np.exp(1j * np.asarray(resultado_fc["theta"], float))
    return Redes(ids, Y1, Y2, Y0, Z1, Z2, Z0, Vf, sem_terra, modelo)
