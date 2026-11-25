"""
Cálculo de faltas por componentes simétricas (equivalente de Thévenin na barra k).

Convenção de Zf (impedância de falta), a mesma dos livros-texto:
  3F  : Zf em cada fase.
  FT  : Zf entre a fase a e a terra.
  FF  : Zf entre as fases b e c (total).
  FFT : Zf do ponto comum das fases b-c até a terra (entra como 3·Zf na sequência zero).
Fases: a é a fase de referência das sequências. 3F = abc; FT = a-terra; FF = b-c; FFT = b-c-terra.
"""
import numpy as np

TIPOS = ("3F", "FT", "FF", "FFT")
_a = np.exp(2j * np.pi / 3)
# [Va, Vb, Vc] = A @ [V0, V1, V2]
A_FORTESCUE = np.array([[1, 1, 1], [1, _a ** 2, _a], [1, _a, _a ** 2]], dtype=complex)


def sequencias_para_fases(v012):
    """v012: (..., 3) com [0, 1, 2] na última dimensão → (..., 3) com [a, b, c]."""
    return np.asarray(v012) @ A_FORTESCUE.T


def calcular_falta(redes, barra_id, tipo, zf=0j):
    """Falta na barra 'barra_id'. Retorna um dicionário com correntes e tensões pós-falta."""
    if tipo not in TIPOS:
        raise ValueError(f"Tipo de falta inválido: {tipo}. Use {TIPOS}.")
    k = redes.ids.index(barra_id)
    Z1, Z2, Z0 = redes.Z1[k, k], redes.Z2[k, k], redes.Z0[k, k]
    Vf = redes.Vf[k]
    zf = complex(zf)
    z0_ok = np.isfinite(Z0.real)          # False: barra sem caminho de sequência zero à terra

    if tipo == "3F":
        I1 = Vf / (Z1 + zf); I2 = I0 = 0j
    elif tipo == "FT":
        I1 = Vf / (Z1 + Z2 + Z0 + 3 * zf) if z0_ok else 0j
        I2 = I0 = I1
    elif tipo == "FF":
        I1 = Vf / (Z1 + Z2 + zf); I2 = -I1; I0 = 0j
    else:  # FFT
        if z0_ok:
            Z0t = Z0 + 3 * zf
            Zp = Z2 * Z0t / (Z2 + Z0t)
            I1 = Vf / (Z1 + Zp)
            I2 = -I1 * Z0t / (Z2 + Z0t)
            I0 = -I1 * Z2 / (Z2 + Z0t)
        else:                                       # sem retorno pela terra: equivale a FF
            I1 = Vf / (Z1 + Z2); I2 = -I1; I0 = 0j

    i012 = np.array([I0, I1, I2], dtype=complex)
    iabc = sequencias_para_fases(i012)

    # Tensões pós-falta em todas as barras: V = V_pré − Z[:,k] · I (só a sequência 1 tem pré-falta)
    V1 = redes.Vf - redes.Z1[:, k] * I1
    V2 = -redes.Z2[:, k] * I2
    V0 = -redes.Z0[:, k] * I0 if I0 != 0 else np.zeros(len(redes.ids), dtype=complex)
    v012 = np.stack([V0, V1, V2], axis=1)
    vabc = sequencias_para_fases(v012)

    corrente_falta = {"3F": abs(iabc[0]), "FT": abs(iabc[0]),
                      "FF": abs(iabc[1]), "FFT": max(abs(iabc[1]), abs(iabc[2]))}[tipo]
    return {"tipo": tipo, "barra": barra_id, "zf": zf, "Vf": Vf,
            "Z1kk": Z1, "Z2kk": Z2, "Z0kk": Z0,
            "I012": i012, "Iabc": iabc, "In": 3 * I0, "I_falta_pu": float(corrente_falta),
            "V012": v012, "Vabc": vabc, "ids": list(redes.ids)}


def perfil_icc_trifasica(redes, zf=0j, s_base_mva=None):
    """Icc trifásica em TODAS as barras (diagonal da Z1). Custo: nenhum além da Zbus já calculada.
    Retorna dict com ids, Icc (pu) e, se s_base_mva for dado, MVAcc = |Vf|·Icc·Sbase."""
    icc = np.abs(redes.Vf / (np.diag(redes.Z1) + zf))
    out = {"ids": list(redes.ids), "Icc_pu": icc}
    if s_base_mva is not None:
        out["MVAcc"] = np.abs(redes.Vf) * icc * s_base_mva
    return out


def para_kA(i_pu, s_base_mva, vbase_kv):
    """Corrente em kA: Ibase = Sbase / (√3 · Vbase)."""
    return i_pu * s_base_mva / (np.sqrt(3.0) * vbase_kv)
