#!/usr/bin/env python3
import argparse

import numpy as np
import tensorflow as tf
from mnist import MNIST

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default="200",
                    type=str, help="Hidden layer configuration.")
parser.add_argument("--models", default=3, type=int, help="Number of models.")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")
args = parser.parse_args()
args.hidden_layers = [int(hidden_layer)
                      for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

# Fix random seeds
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)
m = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),
    ] + [tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu) for hidden_layer in args.hidden_layers] + [
        tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),
    ])

# Load data
mnist = MNIST()
input_layer = tf.keras.Input(
    shape=(MNIST.H, MNIST.W, MNIST.C,))
flat_layer = tf.keras.layers.Flatten(trainable=False)


flat_layer = flat_layer(input_layer)
# Create models
models = []
averaging_layers = []
for model in range(args.models):
    if args.recodex:
        tf.keras.utils.get_custom_objects(
        )["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42 + model)
    dense_layer = tf.keras.layers.Dense(
        args.hidden_layers[0], activation=tf.nn.relu)(flat_layer)
    for hidden_layer in args.hidden_layers[1:]:
        dense_layer = tf.keras.layers.Dense(
            hidden_layer, activation=tf.nn.relu)(dense_layer)
    out_layer = tf.keras.layers.Dense(
        MNIST.LABELS, activation=tf.nn.softmax, name=f"model_{model}_out")(dense_layer)
    models.append(tf.keras.Model(inputs=input_layer, outputs=out_layer))
    models[-1].compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy())

    print("Training model {}: ".format(model + 1), end="", flush=True)
    models[-1].fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs, verbose=0
    )
    if model != 0:
        avg_layer = tf.keras.layers.average(
            [out for out in map(lambda x: x.output, models)], name=f"average_{model}_out")
        averaging_layers.append(avg_layer)
    print("Done")


ensemble_model = tf.keras.Model(inputs=input_layer, outputs=averaging_layers + [
                                out for out in map(lambda x: x.output, models)])
ensemble_model.compile(metrics=[tf.metrics.SparseCategoricalAccuracy() for _ in range(
    args.models*2-1)], optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy())  # Because tensorflow needs optimizer and loss even when they are not used
stats = ensemble_model.evaluate(x=mnist.dev.data["images"], y=[mnist.dev.data["labels"] for _ in range(args.models*2-1)])

with open("mnist_ensemble.out", "w") as out_file:
    for model in range(args.models):

        def get_metric_index(name):
            return ensemble_model.metrics_names.index(name)

        individual_accuracy = stats[get_metric_index(
            f"model_{model}_out_sparse_categorical_accuracy")]

        ensemble_accuracy = individual_accuracy if model == 0 else stats[get_metric_index(
            f"average_{model}_out_sparse_categorical_accuracy")]

        # Print the results.
        print("{:.2f} {:.2f}".format(100 * individual_accuracy,
                                     100 * ensemble_accuracy), file=out_file)
