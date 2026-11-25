import numpy as np
import pytest

import exemplos
from core.adaptador import resolver_fluxo_carga
from core.faltas import calcular_falta, perfil_icc_trifasica, para_kA
from core.sequencia import completar_dados_cc, montar_redes, validar_cc


def radial(lig="Yg-D", gerador_aterrado=True):
    """G(bus1, X''d=X2=0.2, X0=0.05) -- linha (x=0.1, x0=0.3) -- bus2 -- trafo (x=0.08) -- bus3"""
    return {
        "barras": [
            {"id": 1, "tipo": "slack", "v": 1.0, "theta": 0.0, "fonte_cc": True, "x1": 0.2, "x2": 0.2,
             "x0": 0.05, "xn": 0.0, "aterrada": gerador_aterrado},
            {"id": 2, "tipo": "PQ"},
            {"id": 3, "tipo": "PQ"},
        ],
        "linhas": [{"de": 1, "para": 2, "r": 0.0, "x": 0.1, "r0": 0.0, "x0": 0.3}],
        "transformadores": [{"de": 2, "para": 3, "r": 0.0, "x": 0.08, "a": 1.0,
                             "ligacao": lig, "r0": 0.0, "x0": 0.08}],
    }


def redes_de(sis, **kw):
    return montar_redes(sis, **kw)


# ---------------- checagens físicas no ponto da falta (independentes das fórmulas) ----------------
def test_trifasica_valor_analitico_e_tensao_nula():
    r = redes_de(radial()); f = calcular_falta(r, 2, "3F")
    assert f["I_falta_pu"] == pytest.approx(1 / 0.3)
    k = r.ids.index(2); assert np.allclose(f["Vabc"][k], 0, atol=1e-12)
    assert abs(f["Vabc"][r.ids.index(1)][0]) == pytest.approx(1 / 3)      # V = 1 − 0,2·(1/0,3)


def test_fase_terra():
    r = redes_de(radial()); f = calcular_falta(r, 2, "FT"); k = r.ids.index(2)
    z0 = 1 / (1 / 0.35j + 1 / 0.08j)                                       # (gerador+linha) // trafo Yg-D
    assert abs(f["Z0kk"] - z0) < 1e-12
    assert f["I_falta_pu"] == pytest.approx(3 / abs(0.3j + 0.3j + z0))
    assert abs(f["Vabc"][k][0]) < 1e-12                                    # Va = 0
    assert np.allclose(f["Iabc"][1:], 0, atol=1e-12)                       # Ib = Ic = 0
    assert f["In"] == pytest.approx(f["Iabc"][0])                          # In = 3·I0 = Ia


def test_bifasica():
    r = redes_de(radial()); f = calcular_falta(r, 2, "FF"); k = r.ids.index(2)
    assert abs(f["Iabc"][0]) < 1e-12 and f["Iabc"][1] == pytest.approx(-f["Iabc"][2])
    assert f["I_falta_pu"] == pytest.approx(np.sqrt(3) / 0.6)
    assert f["Vabc"][k][1] == pytest.approx(f["Vabc"][k][2])               # Vb = Vc


def test_bifasica_terra():
    r = redes_de(radial()); f = calcular_falta(r, 2, "FFT"); k = r.ids.index(2)
    assert abs(f["Iabc"][0]) < 1e-12                                       # Ia = 0
    assert np.allclose(f["Vabc"][k][1:], 0, atol=1e-12)                    # Vb = Vc = 0
    assert f["In"] == pytest.approx(f["Iabc"][1] + f["Iabc"][2])           # In = Ib + Ic


def test_impedancia_de_falta():
    r = redes_de(radial()); k = r.ids.index(2); zf = 0.1
    f = calcular_falta(r, 2, "FT", zf)
    assert f["Vabc"][k][0] == pytest.approx(zf * f["Iabc"][0])             # Va = Zf·Ia
    f3 = calcular_falta(r, 2, "3F", zf)
    assert f3["I_falta_pu"] == pytest.approx(1 / abs(0.3j + zf))


def test_soma_das_correntes_de_fase_e_tres_i0():
    r = redes_de(radial())
    for t in ("3F", "FT", "FF", "FFT"):
        f = calcular_falta(r, 2, t, 0.05)
        assert f["Iabc"].sum() == pytest.approx(f["In"])


# ---------------- sequência zero e ligações de transformador ----------------
def test_delta_isola_sequencia_zero():
    r = redes_de(radial("Yg-D"))
    assert r.sem_terra_zero == [3]
    f = calcular_falta(r, 3, "FT")
    assert f["I_falta_pu"] == 0                                            # sem retorno pela terra
    ffT = calcular_falta(r, 3, "FFT")                                      # degenera em FF
    assert abs(ffT["In"]) < 1e-12


def test_ygd_lado_yg_ve_o_trafo_como_shunt():
    r = redes_de(radial("Yg-D")); k = r.ids.index(2)
    esperado = 1 / (1 / 0.35j + 1 / 0.08j)                                 # (linha+gerador) // trafo
    assert abs(r.Z0[k, k] - esperado) < 1e-12


def test_ygyg_propaga_sequencia_zero():
    r = redes_de(radial("Yg-Yg"))
    assert r.sem_terra_zero == []
    assert abs(r.Z0[2, 2] - 0.43j) < 1e-12                                 # 0,05 + 0,3 + 0,08


def test_neutro_com_reator_aumenta_z0():
    s = radial("Yg-Yg"); s["barras"][0]["xn"] = 0.02
    r = redes_de(s); assert abs(r.Z0[2, 2] - (0.43 + 0.06) * 1j) < 1e-12   # 3·Xn soma na sequência zero


def test_sistema_isolado_sem_corrente_fase_terra():
    r = redes_de(radial("D-D", gerador_aterrado=False))
    assert sorted(r.sem_terra_zero) == [1, 2, 3]
    assert calcular_falta(r, 2, "FT")["I_falta_pu"] == 0


# ---------------- validação e erros didáticos ----------------
def test_ilha_sem_fonte_gera_erro_claro():
    s = radial(); s["barras"] += [{"id": 4, "tipo": "PQ"}, {"id": 5, "tipo": "PQ"}]
    s["linhas"].append({"de": 4, "para": 5, "r": 0.0, "x": 0.1, "r0": 0.0, "x0": 0.3})
    with pytest.raises(ValueError, match="sem ligação a nenhuma fonte"):
        montar_redes(s)


def test_validar_cc():
    s = radial(); s["barras"][0]["x1"] = 0.0
    assert any("X''d" in e for e in validar_cc(s))
    s = radial(); s["transformadores"][0]["ligacao"] = "Z-Z"
    assert any("inválida" in e for e in validar_cc(s))


def test_Z_e_Y_sao_inversas():
    r = redes_de(radial("Yg-Yg"))
    assert np.allclose(r.Z1 @ r.Y1, np.eye(3)) and np.allclose(r.Z2 @ r.Y2, np.eye(3))
    assert np.allclose(r.Z0 @ r.Y0, np.eye(3))


# ---------------- integração com os exemplos e com o fluxo de carga ----------------
def test_exemplo_gs4_modelos_classico_e_fluxo_de_carga():
    ex = exemplos.EXEMPLOS["gs4"]; sis, avisos = completar_dados_cc(exemplos.carregar_exemplo("gs4"))
    assert avisos                                                          # dados didáticos foram anunciados
    fc = resolver_fluxo_carga(exemplos.carregar_exemplo("gs4"), ex["base_mva"])
    rc = montar_redes(sis); rf = montar_redes(sis, "fluxo_carga", fc, ex["base_mva"])
    pc = perfil_icc_trifasica(rc, s_base_mva=100); pf = perfil_icc_trifasica(rf, s_base_mva=100)
    assert np.all(pc["Icc_pu"] > 0)
    assert np.allclose(pc["Icc_pu"], pf["Icc_pu"], rtol=0.35)              # mesma ordem de grandeza
    f = calcular_falta(rf, 3, "3F"); assert np.allclose(f["Vabc"][rf.ids.index(3)], 0, atol=1e-9)
    assert f["Vf"] == pytest.approx(fc["V"][2] * np.exp(1j * fc["theta"][2]))
    assert pc["MVAcc"][0] == pytest.approx(100 * pc["Icc_pu"][0])          # |Vf| = 1 no modelo clássico


def test_para_kA():
    assert para_kA(1.0, 100.0, 230.0) == pytest.approx(0.25102, abs=1e-4)
