import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import pandas as pd
import numpy as np

import os
import sys
import json
from datetime import datetime

with open("saved_model/vocab.json", "r") as f:
    vocab = json.load(f)


def build_model() -> keras.models.Model:
    input_tf = layers.Input(shape=(None,))
    imd = input_tf
    imd = layers.Embedding(input_dim=len(vocab)+1, output_dim=8)(imd)
    imd = layers.LSTM(units=32)(imd)
    output_tf = layers.Dense(units=1, activation=keras.activations.sigmoid)(imd)

    model = keras.models.Model(inputs=input_tf, outputs=output_tf)

    return model


def build_noise_model(lstm_units=32) -> keras.models.Model:
    input_tf = layers.Input(shape=(None,))
    imd = input_tf
    imd = layers.Embedding(input_dim=len(vocab) + 1, output_dim=8)(imd)
    imd = layers.LSTM(units=lstm_units * 2)(imd)
    output_tf = layers.Dense(units=1, activation=keras.activations.sigmoid)(imd)

    model = keras.models.Model(inputs=input_tf, outputs=output_tf)

    return model


def get_indices(sequences):
    data = []
    for index, sequence in enumerate(sequences):
        tokens = [vocab[token] for token in sequence[:]]
        data.append(tokens)
    return np.array(data)


def load_sup_data(path):
    raw_data = pd.read_csv(path, header=None, delimiter=" ").values
    sequences, train_labels = raw_data[:, 0], np.array(raw_data[:, 1], dtype=np.float)
    sequences = get_indices(sequences)
    return sequences, train_labels


def load_unsup_data(path):
    sequences = pd.read_csv(path, header=None).values
    sequences = get_indices(sequences)
    return sequences


def train_baseline(train_sequences, train_labels):
    model = build_model()
    model.compile(
        optimizer="adam",
        loss=keras.losses.binary_crossentropy,
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.AUC()
        ]
    )
    model.fit(
        train_sequences,
        train_labels,
        batch_size=32,
        epochs=120,
        callbacks=tf.keras.callbacks.TensorBoard(
            log_dir="log/base_line",
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            update_freq='epoch',
            profile_batch=2,
            embeddings_freq=0,
            embeddings_metadata=None,
        )
    )

    return model


def generate_pseudo_label(sequences, model: keras.models.Model):
    # Sample part of sequence
    # TODO

    # Generate pseudo label
    pseudo_labels = model.predict(sequences, batch_size=32)

    # Sample new part of sequence
    # TODO

    return sequences, pseudo_labels


def train_student(train_sequences, train_label):
    model = build_noise_model()
    logdir = "log/{}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    # Train student model
    # TODO

    return model


def run():
    train_sequences, train_labels = load_sup_data("sup_raw_data/input.train")
    test_data = load_sup_data("sup_raw_data/input.ind.test")

    teacher_model = train_baseline(train_sequences, train_labels)

    # Evaluate baseline model
    # TODO

    for i in range(10):
        pseudo_sequences, pseudo_labels = generate_pseudo_label()
        student_model = train_student(pseudo_sequences, pseudo_labels)
        # Evaluate student model
        # TODO


if __name__ == "__main__":
    run()






