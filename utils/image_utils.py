"""Validação e leitura segura das imagens enviadas."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

import settings


class ImageValidationError(ValueError):
    """Erro com mensagem pronta para mostrar ao usuário."""


def load_image(uploaded_file) -> Image.Image:
    ext = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Formato .{ext} não é aceito. Envie JPG, JPEG, PNG ou WEBP."
        )

    if uploaded_file.size > settings.MAX_FILE_MB * 1024 * 1024:
        raise ImageValidationError(
            f"O arquivo tem mais de {settings.MAX_FILE_MB} MB. Envie uma imagem menor."
        )

    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        if image.width * image.height > settings.MAX_PIXELS:
            raise ImageValidationError("A imagem tem resolução alta demais. Reduza o tamanho e tente de novo.")
        image = ImageOps.exif_transpose(image).convert("RGB")
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("Não foi possível abrir a imagem. O arquivo pode estar corrompido.") from exc

    if min(image.size) < settings.MIN_SIDE_PX:
        raise ImageValidationError("A imagem é pequena demais para ser analisada.")
    return image
