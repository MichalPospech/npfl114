!pip install tensorflow-gpu==2.0.0-alpha0
#!/usr/bin/env python3
# this is code copypasted from Colab notebook, so it won't work as it is
#e99f6192-a850-11e7-a937-00505601122b
#ddb5d1e7-ace9-11e7-a937-00505601122b
import numpy as np
import tensorflow as tf

# from modelnet import ModelNet

# The neural network model

def eval_write(args, epoch):
    prediction = network.model.predict(modelnet.test.data["voxels"], batch_size=args.batch_size)
    with open(os.path.join(args.logdir, "modelnet_" + str(epoch+1) + "_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in prediction:
            print(np.argmax(probs), file=out_file)  
    np.save(os.path.join(args.logdir, "modelnet_" + str(epoch+1) + "_test.npy"), prediction)

    prediction = network.model.predict(modelnet.dev.data["voxels"], batch_size=args.batch_size)
    np.save(os.path.join(args.logdir, "modelnet_" + str(epoch+1) + "_dev.npy"), prediction)


class Network:
    def __init__(self, modelnet, args):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input((32, 32, 32, 1)))

        self.model.add(tf.keras.layers.Conv3D(16, 5, padding='same', activation='relu'))
        self.model.add(tf.keras.layers.MaxPool3D())
        self.model.add(tf.keras.layers.Conv3D(32, 5, padding='same', activation='relu'))
        self.model.add(tf.keras.layers.MaxPool3D())
        self.model.add(tf.keras.layers.Conv3D(64, 5, padding='same', activation='relu'))
        self.model.add(tf.keras.layers.MaxPool3D())
        self.model.add(tf.keras.layers.Conv3D(128, 5, padding='same', activation='relu'))
        self.model.add(tf.keras.layers.MaxPool3D())
        self.model.add(tf.keras.layers.Conv3D(256, 5, padding='same', activation='relu'))
        self.model.add(tf.keras.layers.MaxPool3D())
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(10, activation='softmax'))
        
        self.model.summary()

        self.tb_callback = tf.keras.callbacks.TensorBoard(
            args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None
        self.tb_callback.on_epoch_end = lambda epoch, logs: eval_write(args, epoch)

        self.model.compile(loss=tf.losses.sparse_categorical_crossentropy,
                      optimizer=tf.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])

    def train(self, modelnet, args):
        print(args.batch_size)
        print(args.epochs)
        self.model.fit(modelnet.train.data['voxels'], modelnet.train.data['labels'],
                batch_size=args.batch_size, epochs=args.epochs,
                validation_data=(modelnet.dev.data['voxels'], modelnet.dev.data['labels']),
                shuffle=True,
                callbacks=[self.tb_callback])

    def predict(self, dataset, args):
        # TODO: Predict method should return a list/np.ndarray of
        # label probabilities from the test set
        return self.model.predict(modelnet.test.data['voxels'], batch_size=args.batch_size)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--batch_size", default=256, # previously None
#                         type=int, help="Batch size.")
#     parser.add_argument("--modelnet", default=32, type=int,
#                         help="ModelNet dimension.")
#     parser.add_argument("--epochs", default=500,
#                         type=int, help="Number of epochs.")
#     parser.add_argument("--threads", default=4, type=int,
#                         help="Maximum number of threads to use.")
#     args = parser.parse_args()
    args = type('test', (object,), {})()
    args.batch_size = 256
    args.modelnet = 32
    args.epochs = 200
    args.threads = 4

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
#     tf.config.threading.set_inter_op_parallelism_threads(args.threads)
#     tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name    
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename("test"),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    modelnet = ModelNet(args.modelnet)

    # Create the network and train
    network = Network(modelnet, args)
    network.train(modelnet, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    out_path = "3d_recognition_test.txt"
    if os.path.isdir(args.logdir):
        out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for probs in network.predict(modelnet.test, args):
            print(np.argmax(probs), file=out_file)
