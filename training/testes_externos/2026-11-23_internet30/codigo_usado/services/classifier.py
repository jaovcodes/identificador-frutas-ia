"""Classificação das seis frutas com o modelo específico treinado."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
import settings
from utils.model_data import prepare_image

TRAINED = 'trained'

@dataclass
class Candidate:
    key: str
    label: str
    score: float

@dataclass
class Prediction:
    mode: str
    best: Candidate | None
    confident: bool
    threshold: float
    candidates: list[Candidate] = field(default_factory=list)
    raw_top: list[tuple[str, float]] = field(default_factory=list)


def trained_model_available():
    return settings.MODEL_PATH.is_file() and settings.CLASSES_PATH.is_file()


class FruitClassifier:
    def __init__(self):
        from tensorflow.keras.models import load_model
        if not trained_model_available():
            raise RuntimeError('Modelo indisponível. Execute o treinamento antes de identificar frutas.')
        meta = json.loads(settings.CLASSES_PATH.read_text(encoding='utf-8'))
        self.classes = meta['classes']
        if len(self.classes) != 6 or set(self.classes) != set(settings.FRUITS):
            raise ValueError('classes.json não corresponde às seis frutas esperadas.')
        if meta.get('preprocessing') != 'none' or meta.get('resize') != 'bilinear_stretch':
            raise ValueError('Pré-processamento incompatível. Gere novamente o modelo com treinar_modelo.py.')
        self.model = load_model(settings.MODEL_PATH, compile=False)
        if self.model.output_shape[-1] != len(self.classes):
            raise ValueError('Quantidade de saídas diferente de classes.json.')
        self.image_size = tuple(self.model.input_shape[1:3])
        self.mode, self.notice = TRAINED, None

    def predict(self, image: Image.Image):
        x = prepare_image(image, self.image_size)[np.newaxis, ...]
        probabilities = self.model(x, training=False).numpy()[0]
        if not np.isfinite(probabilities).all():
            raise ValueError('O modelo retornou valores inválidos.')
        candidates = [Candidate(self.classes[i], settings.FRUITS[self.classes[i]]['name'], float(probabilities[i]))
                      for i in np.argsort(probabilities)[::-1]]
        best = candidates[0]
        return Prediction(TRAINED, best, best.score >= settings.CONFIDENCE_TRAINED,
                          settings.CONFIDENCE_TRAINED, candidates, [(c.label, c.score) for c in candidates])
