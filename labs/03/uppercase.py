#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=50, type=int,
                    help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default="300",
                    type=str, help="Hidden layer configuration.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")
parser.add_argument("--window", default=7, type=int,
                    help="Window size to use.")
args = parser.parse_args()
args.hidden_layers = [int(hidden_layer)
                      for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

# Fix random seeds
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# Create logdir name
args.logdir = os.path.join("logs", "{}-{}-{}".format(
    os.path.basename(__file__),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
              for key, value in sorted(vars(args).items())))
))

# Load data
uppercase_data = UppercaseData(args.window, args.alphabet_size)

regularizer = None  
embedding_dim = 80
dropout_rate = 0.5
ensemble_size = 5


def create_model():
    input_layer = tf.keras.layers.Input(shape=(args.window*2+1,))
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=args.alphabet_size, output_dim=embedding_dim, trainable=False)(input_layer)
    flat_layer = tf.keras.layers.Flatten()(embedding_layer)

    def create_sub_model(model_num):
        dense_layer = tf.keras.layers.Dense(args.hidden_layers[0], kernel_regularizer=regularizer,
                                            bias_regularizer=regularizer, activation=tf.keras.activations.tanh)(flat_layer)
        dropout_layer = tf.keras.layers.Dropout(dropout_rate)(dense_layer)

        for hidden_layer in args.hidden_layers[1:]:
            dense_layer = tf.keras.layers.Dense(hidden_layer, kernel_regularizer=regularizer,
                                                bias_regularizer=regularizer, activation=tf.keras.activations.tanh)(dropout_layer)
            dropout_layer = tf.keras.layers.Dropout(dropout_rate)(dense_layer)
        out_layer = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name=f"out_{model_num}")(dropout_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=out_layer)
        model.compile(tf.keras.optimizers.Adam(0.005), loss=[tf.losses.BinaryCrossentropy(label_smoothing=0.1)],metrics = [tf.metrics.BinaryAccuracy()])
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.logdir +f"_model_{model_num}", histogram_freq=1)
        model.fit(uppercase_data.train.data["windows"], uppercase_data.train.data["labels"], batch_size=args.batch_size, epochs=args.epochs, callbacks=[tensorboard_callback], validation_split = 0.05)
        return out_layer
    models = [create_sub_model(i) for i in range(ensemble_size)]
    out_layer = tf.keras.layers.average(models, name="average")
    model = tf.keras.Model(inputs=input_layer, outputs=out_layer)
    model.compile(tf.keras.optimizers.Adam(), tf.losses.BinaryCrossentropy(),None) # TF needs all of it
    return model
model = create_model()


preds = model.predict(uppercase_data.test.data["windows"])

with open("uppercase_test.txt", "w", encoding="utf-8") as out_file:
    for x in zip(preds, uppercase_data.test.text):
        if x[0]>0.5:
            out_file.write(x[1].upper())
        else:
            out_file.write(x[1])
        # TODO: Generate correctly capitalized test set.

    
