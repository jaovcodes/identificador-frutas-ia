"""Configurações centrais do projeto (caminhos, limites e catálogo de frutas)."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Arquivos gerados pelo treinamento.
MODEL_PATH = BASE_DIR / "model" / "fruit_model.keras"
CLASSES_PATH = BASE_DIR / "model" / "classes.json"

# Validação de upload
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp"]
MAX_FILE_MB = 5
MAX_PIXELS = 25_000_000
MIN_SIDE_PX = 32

# Confiança mínima para afirmar qual é a fruta.
CONFIDENCE_TRAINED = 0.70   # modelo treinado com Fruits-360

# Catálogo das seis categorias. A chave é o identificador interno,
# que também deve aparecer em model/classes.json após o treinamento.
FRUITS = {
    "apple":      {"name": "Maçã",    "emoji": "🍎", "color": "#C8453B"},
    "banana":     {"name": "Banana",  "emoji": "🍌", "color": "#E3B505"},
    "orange":     {"name": "Laranja", "emoji": "🍊", "color": "#E8862A"},
    "lemon":      {"name": "Limão",   "emoji": "🍋", "color": "#A9B81F"},
    "pineapple":  {"name": "Abacaxi", "emoji": "🍍", "color": "#C9962B"},
    "strawberry": {"name": "Morango", "emoji": "🍓", "color": "#D6405A"},
}
