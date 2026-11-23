"""Prepara seis classes do Fruits-360, sem modificar o conjunto original."""
from pathlib import Path
import argparse
import hashlib
import json
import random
import re
import shutil
from PIL import Image

BASE = Path(__file__).resolve().parent
PATTERNS = {
    'apple': r'Apple(?: .+)? \d+',
    'banana': r'Banana(?: .+)? \d+',
    'orange': r'Orange \d+',
    'lemon': r'Lemon(?: Meyer)? \d+',
    'pineapple': r'Pineapple(?: Mini)? \d+',
    'strawberry': r'Strawberry \d+',
}

def validate(path):
    with Image.open(path) as im:
        if im.format != 'JPEG':
            raise ValueError('Formato diferente de JPEG')
        im.verify()
    with Image.open(path) as im:
        im.convert('RGB').load()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origem', type=Path, default=BASE.parent / 'dataset_fruits360')
    parser.add_argument('--destino', type=Path, default=BASE / 'dataset_frutas6')
    parser.add_argument('--treino', type=int, default=800)
    parser.add_argument('--teste', type=int, default=250)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    source, target = args.origem.resolve(), args.destino.resolve()
    if min(args.treino, args.teste) < 1:
        parser.error('As quantidades devem ser positivas.')
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        parser.error('Origem e destino devem ser independentes.')
    if target.exists():
        parser.error(f'Destino existente; nada foi alterado: {target}. Use outro --destino.')
    for split in ('Training', 'Test'):
        if not (source / split).is_dir():
            parser.error(f'Pasta ausente: {source / split}')
    rng = random.Random(args.seed)
    report = dict(seed=args.seed, source=str(source), classes=list(PATTERNS),
                  counts={}, folders={}, invalid=[], duplicates_skipped=0, images=[])
    seen, plan = set(), []
    # Reserva o teste primeiro e impede duplicatas exatas entre todos os conjuntos.
    for original, split, limit in [('Test', 'test', args.teste), ('Training', 'train', args.treino)]:
        report['counts'][split], report['folders'][split] = {}, {}
        for label, pattern in PATTERNS.items():
            folders = sorted(p for p in (source / original).iterdir()
                             if p.is_dir() and re.fullmatch(pattern, p.name, re.I))
            report['folders'][split][label] = [p.name for p in folders]
            files = sorted(p for folder in folders for p in folder.iterdir()
                           if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg'))
            rng.shuffle(files)
            selected = 0
            for path in files:
                try:
                    validate(path)
                    sha = digest(path)
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    report['invalid'].append(dict(path=str(path), error=str(exc)))
                    continue
                if sha in seen:
                    report['duplicates_skipped'] += 1
                    continue
                seen.add(sha)
                relative = Path(split) / label / f'{path.parent.name}__{path.name}'
                plan.append((path, relative, sha))
                selected += 1
                if selected == limit:
                    break
            if selected != limit:
                raise RuntimeError(f'{split}/{label}: apenas {selected} imagens válidas únicas; necessário {limit}.')
            report['counts'][split][label] = selected
            print(f'{split}/{label}: {selected} imagens válidas', flush=True)
    target.mkdir(parents=True, exist_ok=False)
    for path, relative, sha in plan:
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output)
        validate(output)
        if digest(output) != sha:
            raise RuntimeError(f'Cópia divergente: {output}')
        report['images'].append(dict(source=str(path.relative_to(source)),
                                     destination=relative.as_posix(), sha256=sha))
    (target / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Concluído: {len(plan)} imagens verificadas em {target}')
    print(f"Inválidas encontradas: {len(report['invalid'])}; duplicatas descartadas: {report['duplicates_skipped']}")

if __name__ == '__main__':
    main()
