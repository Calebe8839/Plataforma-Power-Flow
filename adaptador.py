"""
Adaptador entre o formato do diagrama/tabela (barras, linhas, transformadores)
e o motor de fluxo de carga (ybus.py + newton_raphson.py).

Faz exatamente o mesmo mapeamento que interface.py faz hoje, mas como função
pura: serve ao teste de regressão e à página de curto-circuito (pré-falta com
tensões do fluxo de carga). Quando interface.py passar a chamá-la, a lógica
deixa de existir em dois lugares.
"""
import numpy as np

from ybus import build_ybus
from newton_raphson import newton_raphson

_TIPOS = {"slack": "Slack", "pv": "PV", "pq": "PQ"}


def sistema_para_motor(sistema, divisor_potencia=100.0):
    """Converte o sistema para as listas esperadas pelo motor.

    divisor_potencia: Sbase (MVA) se as potências estão em MW/MVAr; 1.0 se já estão em pu.
    Retorna (buses, lines, trafos, id_map) com id_map: número da barra -> posição interna 0..n-1.
    """
    id_map = {b["id"]: i for i, b in enumerate(sistema["barras"])}
    buses = []
    for b in sistema["barras"]:
        tipo = _TIPOS[str(b["tipo"]).lower().strip()]
        buses.append({
            "type": tipo,
            "V": float(b.get("v", 1.0)),
            "theta": float(b.get("theta", 0.0)),
            "P": (b.get("p_ger", 0.0) - b.get("p_carga", 0.0)) / divisor_potencia,
            "Q": 0.0 if tipo == "PV" else (b.get("q_ger", 0.0) - b.get("q_carga", 0.0)) / divisor_potencia,
            "Bsh_bus": float(b.get("bsh_bus", 0.0)),
        })
    lines = [{"from": id_map[l["de"]] + 1, "to": id_map[l["para"]] + 1,
              "R": float(l["r"]), "X": float(l["x"]), "Bsh": float(l.get("bsh", 0.0))}
             for l in sistema.get("linhas", [])]
    trafos = [{"from": id_map[t["de"]] + 1, "to": id_map[t["para"]] + 1,
               "R": float(t["r"]), "X": float(t["x"]), "a": float(t["a"])}
              for t in sistema.get("transformadores", [])]
    return buses, lines, trafos, id_map


def resolver_fluxo_carga(sistema, divisor_potencia=100.0, tol=1e-6, max_iter=20):
    """Roda o fluxo de carga e devolve um dicionário pequeno e serializável.

    Chaves: V (pu), theta (rad), convergiu, n_iter, erro, id_map, Ybus.
    A página de simulação pode guardar isto em st.session_state para não
    recalcular a cada rerun, e a página de CC usa V e theta como pré-falta.
    """
    buses, lines, trafos, id_map = sistema_para_motor(sistema, divisor_potencia)
    Y = build_ybus(buses, lines, trafos if trafos else None)
    V, theta, log, _, _, _, _, conv = newton_raphson(buses, Y, tol=tol, max_iter=max_iter)
    return {"V": V, "theta": theta, "convergiu": bool(conv), "n_iter": len(log),
            "erro": float(log[-1]["erro"]), "id_map": id_map, "Ybus": Y}
