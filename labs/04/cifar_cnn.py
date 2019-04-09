#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from re import compile, match
from mnist import MNIST
from functools import reduce
from cifar10 import CIFAR10
# The neural network model
# Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR without following lines, solution from https://github.com/tensorflow/tensorflow/issues/24496#issuecomment-464909727
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class Network(tf.keras.Model):
    def __init__(self, args):
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        def split_params(params):
            split = compile(r"(R-\[[^\]]+\]|[^,]+),?")
            groups = split.findall(params)
            return groups

        def create_layer(desc):
            if desc[0] == "F":
                return tf.keras.layers.Flatten()
            elif desc[0:2] == "DR":
                return tf.keras.layers.Dropout(0.5)
            elif desc[0] == "D":
                size = int(desc[2:])
                return tf.keras.layers.Dense(size, tf.keras.activations.relu)
            elif desc[0] == "M":
                regex =  compile(r"M-(.+)-(.+)")
                match_ob = regex.match(desc)
                return tf.keras.layers.MaxPool2D(int(match_ob[1]),int(match_ob[2]))
                
            elif desc[0:2] == "CB":
                regex = compile(r"CB-(.+)-(.+)-(.+)-(.+)")
                match_ob = regex.match(desc)
                conv = tf.keras.layers.Conv2D(int(match_ob[1]),int(match_ob[2]),int(match_ob[3]),match_ob[4],use_bias=False)
                batch_norm = tf.keras.layers.BatchNormalization()
                activ = tf.keras.layers.Activation(tf.keras.activations.relu)
                layers = [conv,batch_norm,activ]
                return lambda x: apply_layers([x]+ layers)
            elif desc[0] == "C":
                regex = compile(r"C-(.+)-(.+)-(.+)-(.+)")
                match_ob = regex.match(desc)
                return tf.keras.layers.Conv2D(int(match_ob[1]),int(match_ob[2]),int(match_ob[3]),match_ob[4], activation=tf.keras.activations.relu)
            elif desc[0] == "R":
                layers = process_params(desc[3:-1])
                return lambda x: tf.keras.layers.add([x,layers(x)])
                
        def apply_layers(layers):
            return reduce(lambda beg, layer: layer(beg), layers)
        
        def process_params(params):
            layers_desc = split_params(params)
            layers =  [create_layer(desc) for desc in layers_desc]
            return(lambda x: apply_layers([x] + layers))


        network_ctor = process_params(args.cnn)
        hidden = network_ctor(inputs)        # Add the final output layer
        outputs = tf.keras.layers.Dense(
            MNIST.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)

        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy")],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(
            args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, mnist, args):
        self.fit(
            mnist.train.data["images"], mnist.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(
                mnist.dev.data["images"], mnist.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )

    def test(self, mnist, args):
        test_logs = self.evaluate(
            mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size)
        self.tb_callback.on_epoch_end(1, dict(
            ("val_test_" + metric, value) for metric, value in zip(self.metrics_names, test_logs)))
        return test_logs[self.metrics_names.index("accuracy")]


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50,
                        type=int, help="Batch size.")
    parser.add_argument("--cnn", default=None, type=str,
                        help="CNN architecture.")
    parser.add_argument("--epochs", default=30, type=int,
                        help="Number of epochs.")
    parser.add_argument("--recodex", default=False,
                        action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--threads", default=1, type=int,
                        help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects(
        )["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = CIFAR10()

    # Create the network and train
    network = Network(args)
    network.train(mnist, args)

    # Compute test set accuracy and print it
    accuracy = network.test(mnist, args)
    with open("mnist_cnn.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)
