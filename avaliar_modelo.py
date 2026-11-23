"""Avalia o modelo salvo sem usar o teste para ajustar parâmetros."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import settings
from utils.model_data import records, batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=settings.BASE_DIR / 'dataset_frutas6' / 'test')
    parser.add_argument('--model-dir', type=Path, default=settings.BASE_DIR / 'model')
    parser.add_argument('--output', type=Path, default=settings.BASE_DIR / 'training' / 'evaluation')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size deve ser positivo.')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    meta = json.loads((args.model_dir / 'classes.json').read_text(encoding='utf-8'))
    if meta.get('preprocessing') != 'none':
        raise ValueError('Esta avaliação espera normalização interna ao modelo.')
    classes = meta['classes']
    items = records(args.dataset, classes)
    model = tf.keras.models.load_model(args.model_dir / 'fruit_model.keras', compile=False)
    size = tuple(model.input_shape[1:3])
    probabilities, labels = [], []
    for x, y in batches(items, args.batch_size, size):
        probabilities.append(model(x, training=False).numpy())
        labels.append(y)
        done = sum(len(v) for v in labels)
        if done % 320 == 0 or done == len(items):
            print(f'Avaliação: {done}/{len(items)}', flush=True)
    probs, actual = np.concatenate(probabilities), np.concatenate(labels)
    predicted = probs.argmax(axis=1)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(matrix, (actual, predicted), 1)
    per_class = {}
    for i, name in enumerate(classes):
        tp, support, selected = int(matrix[i,i]), int(matrix[i].sum()), int(matrix[:,i].sum())
        precision, recall = tp / selected if selected else 0., tp / support
        per_class[name] = dict(label=settings.FRUITS[name]['name'], support=support, precision=precision,
                               recall=recall, f1=2*precision*recall/(precision+recall) if precision+recall else 0.)
    confidence = probs.max(axis=1)
    accepted = confidence >= settings.CONFIDENCE_TRAINED
    report = dict(dataset=str(args.dataset.resolve()), count=len(items), classes=classes,
                  accuracy=float(np.mean(actual == predicted)),
                  cross_entropy=float(-np.log(np.clip(probs[np.arange(len(actual)), actual], 1e-7, 1)).mean()),
                  confidence_threshold=settings.CONFIDENCE_TRAINED, coverage=float(accepted.mean()),
                  accepted_accuracy=float(np.mean(actual[accepted] == predicted[accepted])) if accepted.any() else None,
                  confusion_matrix=matrix.tolist(), per_class=per_class,
                  limitation='Fruits-360 com fundo padronizado; fotografias reais ainda não avaliadas.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (args.output / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'actual', 'predicted', 'confidence', 'accepted'])
        for (path, _), y, pred, score, accept in zip(items, actual, predicted, confidence, accepted):
            writer.writerow([str(path.relative_to(args.dataset)), classes[y], classes[pred], float(score), bool(accept)])
    # Matriz portátil, sem instalar bibliotecas gráficas adicionais.
    canvas = Image.new('RGB', (880, 660), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), 'Matriz de confusao - Fruits-360 (contagens)', fill='black')
    draw.text((20, 40), 'Linhas: classe real | Colunas: previsao', fill='black')
    for i, name in enumerate(classes):
        draw.text((160 + i*112, 80), name, fill='black')
        draw.text((15, 125 + i*80), name, fill='black')
        for j in range(len(classes)):
            shade = int(230 - 160*matrix[i,j]/max(1, matrix.max()))
            x0, y0 = 155+j*112, 110+i*80
            draw.rectangle((x0, y0, x0+108, y0+76), fill=(shade, shade, 255))
            draw.text((x0+40, y0+30), str(matrix[i,j]), fill='black')
    canvas.save(args.output / 'confusion_matrix.png')
    print(json.dumps({k:v for k,v in report.items() if k not in ('confusion_matrix',)}, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
