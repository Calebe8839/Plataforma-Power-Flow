"""Rede de segurança do fluxo de carga: o motor deve reproduzir as soluções publicadas de exemplos.py.
Rode (na raiz do projeto) antes e depois de qualquer refatoração:  pytest -q"""
import numpy as np
import pytest

import exemplos
from core.adaptador import resolver_fluxo_carga


@pytest.mark.parametrize("chave", [c for c, ex in exemplos.EXEMPLOS.items() if "referencia" in ex])
def test_fluxo_de_carga_reproduz_referencia(chave):
    ex = exemplos.EXEMPLOS[chave]
    r = resolver_fluxo_carga(exemplos.carregar_exemplo(chave), divisor_potencia=ex["base_mva"])
    assert r["convergiu"]
    ref = ex["referencia"]
    assert np.allclose(r["V"], ref["V"], atol=1e-3)
    assert np.allclose(np.degrees(r["theta"]), ref["theta_graus"], atol=2e-3)
