#!/usr/bin/env python3
#e99f6192-a850-11e7-a937-00505601122b
#ddb5d1e7-ace9-11e7-a937-00505601122b

import numpy as np
import tensorflow as tf

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset
import math



tf.config.gpu.set_per_process_memory_growth(True)

class Network:
    def __init__(self, args, num_words, num_tags, num_chars):
        # TODO(we): Implement a one-layer RNN network. The input
        # `word_ids` consists of a batch of sentences, each
        # a sequence of word indices. Padded words have index 0.
        word_ids = tf.keras.Input(shape=(None,), dtype='int32')
        charseqs = tf.keras.Input(shape=(None,))
        charseq_ids = tf.keras.Input(shape=(None,), dtype='int32')
        self._epochs = args.epochs
        # TODO: Apart from `word_ids`, RNN CLEs utilize two more
        # inputs, `charseqs` containing unique words in batches (each word
        # being a sequence of character indices, padding characters again
        # have index 0) and `charseq_ids` with the same shape as `word_ids`,
        # but with indices pointing into `charseqs`.

        # TODO: Embed the characters in `charseqs` using embeddings of size
        # `args.cle_dim`, masking zero indices. Then, pass the embedded characters
        # through a bidirectional GRU with dimension `args.cle_dim`, concatenating
        # results in different dimensions.

        chars_embedding = tf.keras.layers.Embedding(
            num_chars, args.cle_dim, mask_zero=False)(charseqs)
        gru_cell = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(args.cle_dim, dropout = 0.5))(chars_embedding)

        # Then, copy the computed embeddings of unique words to the correct sentence
        # positions. To that end, use `tf.gather` operation, which is given a matrix
        # and a tensor of indices, and replace each index by a corresponding row
        # of the matrix. You need to wrap the `tf.gather` in `tf.keras.layers.Lambda`
        # because of a bug [fixed 6 days ago in the master], so the call shoud look like
        # `tf.keras.layers.Lambda(lambda args: tf.gather(*args))(...)`
        # TODO(we): Embed input words with dimensionality `args.we_dim`, using
        # `mask_zero=True`.
        gather = tf.keras.layers.Lambda(lambda args: tf.gather(*args))([gru_cell, charseq_ids])
        # gather = tf.gather(gru_cell, charseq_ids)
        # TODO: Concatenate the WE and CLE embeddings (in this order).
        words_embedding = tf.keras.layers.Embedding(num_words, args.we_dim, mask_zero=False)(word_ids)
        con = tf.keras.layers.Concatenate()([words_embedding, gather])

        # TODO(we): create specified `args.rnn_cell` rnn cell (lstm, gru) with
        # dimension `args.rnn_cell_dim` and apply it in a bidirectional way on
        # the embedded words, concatenating opposite directions.
        cell = tf.keras.layers.LSTM
     

        recur_cell = cell(args.rnn_cell_dim, return_sequences=True, dropout = 0.5)
        bidir_cell = tf.keras.layers.Bidirectional(recur_cell)(con)



        recur_cell = cell(args.rnn_cell_dim, return_sequences=True, dropout = 0.5)
        bidir_cell = tf.keras.layers.Bidirectional(recur_cell)(bidir_cell)


        # TODO(we): Add a softmax classification layer into `num_tags` classes, storing
        # the outputs in `predictions`.
        predictions = tf.keras.layers.Dense(
            num_tags, activation=tf.keras.activations.softmax)(bidir_cell)

        self.model = tf.keras.Model(
            inputs=[word_ids, charseq_ids, charseqs], outputs=predictions)

        # TODO: Create an Adam optimizer in self._optimizer
        self._optimizer = tf.keras.optimizers.Adam(0.0005)
        # TODO: Create a suitable loss in self._loss
        self._loss = tf.keras.losses.SparseCategoricalCrossentropy()
        # TODO: Create two metrics in self._metrics dictionary:
        #  - "loss", which is tf.metrics.Mean()
        #  - "accuracy", which is suitable accuracy
        self._metrics = {
            "loss": tf.metrics.Mean(),
            "accuracy": tf.keras.metrics.SparseCategoricalAccuracy()}
        self._writer = tf.summary.create_file_writer(
            args.logdir, flush_millis=10 * 1000)

    @tf.function(input_signature=[[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 3,
                                  tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def train_batch(self, inputs, tags):
        # TODO: Generate a mask from `tags` containing ones in positions
        # where tags are nonzero (using `tf.not_equal`).
        mask = tf.not_equal(tags, tf.constant(0, dtype=tags.dtype))

        with tf.GradientTape() as tape:
            probabilities = self.model(inputs, training=True)
            # TODO: Compute `loss` using `self._loss`, passing the generated
            # tag mask as third parameter.
            loss = self._loss(tags, probabilities, mask)
            gradients = tape.gradient(loss, self.model.variables)
            grads, _ = tf.clip_by_global_norm(
                    gradients, 2)
            self._optimizer.apply_gradients(
                zip(grads, self.model.variables))

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                metric.reset_states()
                if name == "loss":
                    metric(loss)
                else:
                    # TODO: Update the `metric` using gold `tags` and generated `probabilities`,
                    # passing the tag mask as third argument.
                    metric(tags, probabilities, mask)

                tf.summary.scalar("train/{}".format(name),
                                  metric.result(), step=None)

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            self.train_batch([batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids, batch[dataset.FORMS].charseqs],
                             batch[dataset.TAGS].word_ids)

    @tf.function(input_signature=[[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 3,
                                  tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def evaluate_batch(self, inputs, tags):
        # TODO: Again generate a mask from `tags` containing ones in positions
        # where tags are nonzero (using `tf.not_equal`).
        mask = tf.not_equal(tags, tf.constant(0, dtype=tags.dtype))
        probabilities = self.model(inputs, training=False)
        # TODO: Compute `loss` using `self._loss`, passing the generated
        # tag mask as third parameter.
        loss = self._loss(tags, probabilities, mask)

        for name, metric in self._metrics.items():
            if name == "loss":
                metric(loss)
            else:
                # TODO: Update the `metric` using gold `tags` and generated `probabilities`,
                # passing the tag mask as third argument.
                metric(tags, probabilities, mask)

    def evaluate(self, dataset, dataset_name, args):
        for metric in self._metrics.values():
            metric.reset_states()
        for batch in dataset.batches(args.batch_size):
            # TODO: Evaluate the given match, using the same inputs as in training.
            self.evaluate_batch([batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids,
                                 batch[dataset.FORMS].charseqs], batch[dataset.TAGS].word_ids)

        metrics = {name: metric.result()
                   for name, metric in self._metrics.items()}
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(
                    "{}/{}".format(dataset_name, name), value, step=None)

        return metrics  
    def train(self, dataset,eval_dataset, args):
        for epoch in range(self._epochs):
            self.train_epoch(dataset,args)
            self._new_epoch_callback()
            self.evaluate(eval_dataset, "eval",args)
            print(f"Epoch {epoch+1} done.")

    def predict(self, dataset, args):
        # TODO: Predict method should return a list, each element corresponding
        # to one sentence. Each sentence should be a list/np.ndarray
        # containing _indices_ of chosen tags (not the logits/probabilities).
        prediction = []

        batch_i = 0
        for batch in dataset.batches(args.batch_size):
            batch_i += 1
            print("\rPredicting batch {}/{}".format(batch_i, math.ceil(dataset.size()/args.batch_size)), end="")
            prediction += self.predict_batch([batch[0].word_ids, batch[0].charseq_ids, batch[0].charseqs])
            # print(prediction)
        print()

        return prediction

    # @tf.function(input_signature=[[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 3])
    def predict_batch(self, inputs):

        prediction  = []
        for pred in self.model(inputs, training=False):
            sentence = []
            for word in pred:
                sentence.append(np.argmax(word))
            prediction.append(sentence)
    #        print(prediction)
        return prediction

    def _new_epoch_callback(self):
        pass


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=12, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--cle_dim", default=192, type=int,
                        help="CLE embedding dimension.")
    parser.add_argument("--we_dim", default=384, type=int,
                        help="Word embedding dimension.")
    parser.add_argument("--rnn_cell_dim", default=384   ,
                        type=int, help="RNN cell dimension.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # Create the network and train
    network = Network(args,num_words=len(
                          morpho.train.data[morpho.train.FORMS].words),
                      num_tags=len(morpho.train.data[morpho.train.TAGS].words),
                      num_chars=len(morpho.train.data[morpho.train.FORMS].alphabet))
    network.train(morpho.train,morpho.dev,  args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    out_path = "tagger_competition_test.txt"
    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for i, sentence in enumerate(network.predict(morpho.test, args)):
            for j in range(len(morpho.test.data[morpho.test.FORMS].word_strings[i])):
                print(morpho.test.data[morpho.test.FORMS].word_strings[i][j],
                      morpho.test.data[morpho.test.LEMMAS].word_strings[i][j],
                      morpho.test.data[morpho.test.TAGS].words[sentence[j]],
                      sep="\t", file=out_file)
            print(file=out_file)
