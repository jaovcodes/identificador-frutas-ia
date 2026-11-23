"""Avalia fotos externas rotuladas, preservando evidências e o modelo usado."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image
import settings
from services.classifier import FruitClassifier


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imagens',type=Path,required=True)
    parser.add_argument('--rotulos',type=Path,required=True,help='CSV: file,expected,visual_group')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Destino já existe. Use outra pasta para preservar os resultados.')
    with args.rotulos.open(encoding='utf-8-sig',newline='') as f:
        cases=list(csv.DictReader(f))
    if not cases or len({r['file'] for r in cases})!=len(cases):
        parser.error('Rótulos vazios ou nomes repetidos.')
    source=args.imagens.resolve()
    for case in cases:
        path=(source/case['file']).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            parser.error(f"Arquivo inválido: {case['file']}")
        if case['expected'] not in settings.FRUITS:
            parser.error(f"Rótulo desconhecido: {case['expected']}")
    classifier=FruitClassifier()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'imagens').mkdir()
    (args.output/'modelo_baseline').mkdir()
    for path in [settings.MODEL_PATH,settings.CLASSES_PATH,settings.MODEL_PATH.parent/'training.json']:
        if path.is_file():
            shutil.copy2(path,args.output/'modelo_baseline'/path.name)
    shutil.copy2(args.rotulos,args.output/'rotulos.csv')
    rows=[]
    for case in cases:
        original=source/case['file']
        copied=args.output/'imagens'/case['file']
        copied.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(original,copied)
        row=dict(case,sha256=sha(copied),original=str(original),source_url=None)
        try:
            with Image.open(copied) as image:
                result=classifier.predict(image)
            row.update(predicted=result.best.key,confidence=result.best.score,
                       confident=result.confident,correct=result.best.key==case['expected'],
                       probabilities={c.key:c.score for c in result.candidates},error=None)
            row['status']=('acerto_aceito' if row['correct'] else 'erro_aceito') if result.confident else 'inconclusivo'
        except (OSError,ValueError) as exc:
            row.update(predicted=None,confidence=None,confident=False,correct=False,status='erro_processamento',error=str(exc))
        rows.append(row)
        print(case['file'],row['predicted'],row['confidence'],row['status'],flush=True)
    n=len(rows)
    accepted=[r for r in rows if r['confident']]
    per_class={}
    for label in settings.FRUITS:
        subset=[r for r in rows if r['expected']==label]
        per_class[label]=dict(count=len(subset),top1_correct=sum(r['correct'] for r in subset),
                              accepted_correct=sum(r['status']=='acerto_aceito' for r in subset),
                              accepted_wrong=sum(r['status']=='erro_aceito' for r in subset),
                              uncertain=sum(r['status']=='inconclusivo' for r in subset))
    classes=sorted(settings.FRUITS)
    matrix=[[sum(r['expected']==a and r['predicted']==b for r in rows) for b in classes] for a in classes]
    report=dict(timestamp=datetime.now(ZoneInfo('America/Manaus')).isoformat(),
                model_sha256=sha(settings.MODEL_PATH),threshold=settings.CONFIDENCE_TRAINED,
                count=n,top1_correct=sum(r['correct'] for r in rows),
                top1_accuracy=sum(r['correct'] for r in rows)/n,
                accepted=len(accepted),accepted_correct=sum(r['correct'] for r in accepted),
                accepted_wrong=sum(not r['correct'] for r in accepted),
                uncertain=sum(r['status']=='inconclusivo' for r in rows),
                coverage=len(accepted)/n,accepted_accuracy=sum(r['correct'] for r in accepted)/len(accepted) if accepted else None,
                per_class=per_class,confusion_classes=classes,confusion_matrix=matrix,results=rows,
                limitations=['Amostra pequena de conveniência, sem estimativa de desempenho geral.',
                             'Rótulos por inspeção visual; URLs e licenças não fornecidas.',
                             'Arquivos com o mesmo visual_group são variantes da mesma foto; não independentes.',
                             'Inclui falhas anteriores; não somar esta rodada à anterior.',
                             'Uso reservado para regressão; não usar no treino nem como teste final independente após ajustes.'])
    (args.output/'registro.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (args.output/'resultados.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['file','expected','predicted','confidence','confident','correct','status','visual_group','sha256','error'],extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    print('RESUMO',json.dumps({k:report[k] for k in ['count','top1_correct','top1_accuracy','accepted_correct','accepted_wrong','uncertain']},ensure_ascii=False))

if __name__=='__main__':
    main()
