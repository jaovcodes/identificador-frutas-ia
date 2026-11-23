"""Confere as evidências publicadas; não executa treino nem nova inferência."""
from pathlib import Path
import csv
import hashlib
import json
import math
from collections import Counter

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads((BASE / path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    manifest = read('documentacao/manifesto_dataset_frutas6.json')
    entries = manifest['images']
    check(len(entries) == 6300, 'Manifesto deve conter 6.300 imagens.')
    check(len({r['sha256'] for r in entries}) == 6300, 'Hashes repetidos no manifesto.')
    check(len({r['destination'] for r in entries}) == 6300, 'Destinos repetidos no manifesto.')
    counts = Counter(tuple(Path(r['destination']).parts[:2]) for r in entries)
    for label in manifest['classes']:
        check(counts[('train', label)] == 800, f'Treino desequilibrado: {label}')
        check(counts[('test', label)] == 250, f'Teste desequilibrado: {label}')
    training = read('model/training.json')
    train = {p.replace('\\', '/') for p in training['train_files']}
    validation = {p.replace('\\', '/') for p in training['validation_files']}
    check(len(train) == 3840 and len(validation) == 960, 'Divisão de treino/validação incorreta.')
    check(not train & validation, 'Sobreposição de treino/validação.')
    prepared = {r['destination'] for r in entries if r['destination'].startswith('train/')}
    check(train | validation == prepared, 'Divisão não corresponde ao manifesto.')
    for label in manifest['classes']:
        check(sum(Path(p).parts[1] == label for p in train) == 640, f'Ajuste: {label}')
        check(sum(Path(p).parts[1] == label for p in validation) == 160, f'Validação: {label}')
    metrics = read('training/evaluation/metrics.json')
    matrix = metrics['confusion_matrix']
    check(sum(map(sum, matrix)) == 1500, 'Matriz Fruits-360 incompleta.')
    check(sum(matrix[i][i] for i in range(6)) == 1499, 'Acertos Fruits-360 divergentes.')
    with (BASE / 'training/evaluation/predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        predictions = list(csv.DictReader(f))
    check(len(predictions) == 1500, 'CSV Fruits-360 incompleto.')
    check(sum(r['actual'] == r['predicted'] for r in predictions) == 1499, 'CSV Fruits-360 divergente.')
    model_hash = digest(BASE / 'model/fruit_model.keras')
    for folder, expected_count in [('2026-11-23_rodada01', 6), ('2026-11-23_rodada02_completa', 11), ('2026-11-23_internet30/avaliacao', 30)]:
        path = Path('training/testes_externos') / folder
        record = read(path / 'registro.json')
        check(record['model_sha256'] == model_hash, f'Modelo divergente: {folder}')
        rows = record['results']
        check(len(rows) == expected_count, f'Contagem divergente: {folder}')
        images_dir = BASE / path / 'imagens'
        if expected_count == 30:
            images_dir = BASE / path.parent / 'selecionadas'
        for row in rows:
            image = images_dir / row['file']
            check(image.is_file(), f'Evidência ausente: {image}')
            check(digest(image) == row['sha256'], f'Hash divergente: {image}')
        if expected_count != 6:
            correct = sum(row['correct'] for row in rows)
            accepted_correct = sum(row['correct'] and row['confident'] for row in rows)
            accepted_wrong = sum(not row['correct'] and row['confident'] for row in rows)
            uncertain = sum(not row['confident'] for row in rows)
            check(math.isclose(correct / expected_count, record['top1_accuracy']), 'Acurácia divergente.')
            check((accepted_correct, accepted_wrong, uncertain) == (record['accepted_correct'], record['accepted_wrong'], record['uncertain']), 'Estados da interface divergentes.')
            check(accepted_correct + accepted_wrong + uncertain == expected_count, 'Categorias da interface incompletas.')
    prints = BASE / 'training/testes_externos/2026-11-23_rodada01/prints'
    check(len(list(prints.glob('caso_*.png'))) == 6, 'Devem existir seis prints iniciais.')
    print('OK: manifesto, divisão 3.840/960, teste de 1.500, modelo, fotos das três rodadas e seis prints conferidos.')
    print('Esta é uma auditoria dos registros; nenhuma nova inferência foi executada.')


if __name__ == '__main__':
    main()
