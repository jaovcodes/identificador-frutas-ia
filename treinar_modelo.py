"""Treina uma cabeça de seis classes sobre a MobileNetV2 congelada."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import argparse
import json
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
import settings
from utils.model_data import records, batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=settings.BASE_DIR / 'dataset_frutas6')
    parser.add_argument('--output', type=Path, default=settings.BASE_DIR / 'model')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs e batch-size devem ser positivos.')
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / f).exists() for f in ('fruit_model.keras', 'classes.json', 'training.json')):
        parser.error('Já existem resultados. Use outro --output para preservar o modelo anterior.')
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    classes = sorted(settings.FRUITS)
    items = records(args.dataset / 'train', classes)
    rng = np.random.default_rng(args.seed)
    train_indices, val_indices = [], []
    for label in range(len(classes)):
        indices = np.array([i for i, (_, y) in enumerate(items) if y == label])
        rng.shuffle(indices)
        nval = max(1, int(len(indices) * .2))
        if len(indices) - nval < 1:
            raise ValueError('São necessárias pelo menos duas imagens por classe.')
        val_indices.extend(indices[:nval])
        train_indices.extend(indices[nval:])
    start = time.monotonic()
    print('GPUs detectadas:', tf.config.list_physical_devices('GPU'), flush=True)
    # Reutiliza os pesos 224 do ImageNet já presentes no cache local.
    base = tf.keras.applications.MobileNetV2(weights='imagenet')
    base.trainable = False
    # 160 px reduz o custo em CPU; a base convolucional aceita entrada dinâmica.
    image_size = (160, 160)
    dynamic_base = tf.keras.applications.MobileNetV2(include_top=False, weights=None,
                                                   input_shape=(*image_size, 3), pooling='avg')
    dynamic_base.set_weights(base.get_weights()[:-2])
    del base
    dynamic_base.trainable = False
    inputs = tf.keras.Input(shape=(*image_size, 3))
    features = dynamic_base(tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs), training=False)
    extractor = tf.keras.Model(inputs, features)
    # Apenas vetores de 1280 valores ficam na RAM, nunca todas as imagens.
    vectors, labels = [], []
    for x, y in batches(items, args.batch_size, image_size):
        vectors.append(extractor(x, training=False).numpy())
        labels.append(y)
        done = sum(len(v) for v in vectors)
        if done % 320 == 0 or done == len(items):
            print(f'Extração: {done}/{len(items)}', flush=True)
    x, y = np.concatenate(vectors), np.concatenate(labels)
    head = tf.keras.Sequential([
        tf.keras.Input(shape=(1280,)),
        tf.keras.layers.Dropout(.25),
        tf.keras.layers.Dense(len(classes), activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(.0001)),
    ], name='fruit_head')
    head.compile(optimizer=tf.keras.optimizers.Adam(.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = head.fit(x[train_indices], y[train_indices], validation_data=(x[val_indices], y[val_indices]),
                       batch_size=32, epochs=args.epochs, verbose=2,
                       callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)])
    model = tf.keras.Model(inputs, head(features), name='frutas6')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.save(args.output / 'fruit_model.keras')
    meta = dict(classes=classes, preprocessing='none', resize='bilinear_stretch', image_size=list(image_size))
    (args.output / 'classes.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    report = dict(seed=args.seed, train_count=len(train_indices), validation_count=len(val_indices),
                  validation_method='20% por classe, aleatória; imagens sequenciais podem ser semelhantes',
                  test_used_for_training=False, augmentation='Dropout na cabeça; sem aumento das imagens',
                  history=history.history, best_epoch=int(np.argmin(history.history['val_loss'])) + 1,
                  seconds=time.monotonic() - start,
                  tensorflow=tf.__version__, devices=[str(d) for d in tf.config.list_physical_devices()],
                  train_files=[str(items[i][0].relative_to(args.dataset)) for i in train_indices],
                  validation_files=[str(items[i][0].relative_to(args.dataset)) for i in val_indices])
    (args.output / 'training.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Modelo salvo em', args.output, flush=True)

if __name__ == '__main__':
    main()
