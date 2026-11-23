"""Compatibilidade com o caminho antigo; implementação na raiz do projeto."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preparar_dataset import main

if __name__ == '__main__':
    main()
