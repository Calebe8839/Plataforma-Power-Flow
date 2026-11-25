# Garante que a raiz do projeto (onde ficam ybus.py, newton_raphson.py...) esteja no sys.path ao rodar pytest.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
