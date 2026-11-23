"""Leitura em lotes, idêntica no treino, avaliação e aplicativo."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def prepare_image(image, size):
    image = ImageOps.exif_transpose(image).convert('RGB')
    return np.asarray(image.resize((size[1], size[0]), Image.Resampling.BILINEAR), dtype=np.float32)


def records(root, classes):
    root = Path(root)
    actual = {p.name for p in root.iterdir() if p.is_dir()}
    if actual != set(classes):
        raise ValueError(f'Classes inesperadas em {root}: {actual}')
    result = []
    for label, name in enumerate(classes):
        files = sorted(p for p in (root / name).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        if not files:
            raise ValueError(f'Classe vazia: {name}')
        result.extend((p, label) for p in files)
    return result


def batches(items, batch_size, size):
    for start in range(0, len(items), batch_size):
        part = items[start:start + batch_size]
        images = []
        for path, _ in part:
            with Image.open(path) as image:
                images.append(prepare_image(image, size))
        yield np.stack(images), np.asarray([label for _, label in part], dtype=np.int32)
