# Núcleo de curto-circuito — Fase 0/1

Copiar para a raiz do repositório (ao lado de ybus.py, newton_raphson.py, interface.py):

    conftest.py
    core/__init__.py  core/adaptador.py  core/sequencia.py  core/faltas.py
    tests/test_fc_regressao.py  tests/test_cc.py

Rodar na raiz:  pip install pytest && pytest -q      (esperado: 18 passed)

Nada aqui altera interface.py: o fluxo de carga segue idêntico.
Os testes de regressão do FC comparam gs4 e stagg5 com as soluções publicadas de exemplos.py.
