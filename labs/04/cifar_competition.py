#!/usr/bin/env python3
# ddb5d1e7-ace9-11e7-a937-00505601122b
# e99f6192-a850-11e7-a937-00505601122b


#I tried implementating resnet or Pyramidnet however I was unable to make it work well, therefore I found a implementation of Resnet which made it to 85% val. acc. Then I put it in a ensemble with Dan
import numpy as np
import tensorflow as tf
from functools import reduce
from cifar10 import CIFAR10


# Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR without following lines, solution from https://github.com/tensorflow/tensorflow/issues/24496#issuecomment-464909727
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def eval_test(args, epoch):
    print(f"Epoch {epoch} done")
    with open(os.path.join(args.logdir, "cifar_competition_test_" + str(epoch) + ".txt"), "w", encoding="utf-8") as out_file:
        probs = network.predict(cifar.test.data["images"])
        for prob in probs:
            print(np.argmax(prob), file=out_file)              
    np.save(os.path.join(args.logdir, "cifar_competition_test_" + str(epoch) + ".npy"), probs)
    dev_probs =  network.predict(cifar.dev.data["images"])
    np.save(os.path.join(args.logdir, "cifar_competition_dev_" + str(epoch) + ".npy"), dev_probs)

# The neural network model


#Taken from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.shape[1]
        w = img.shape[2]

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        img = img * mask

        return img
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

class ResnetModel(tf.keras.Model):
    def __init__(self, args):

        from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
        from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
        from tensorflow.keras.callbacks import ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.regularizers import l2





        def lr_schedule(epoch):
            """Learning Rate Schedule
            Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
            Called automatically every epoch as part of callbacks during training.
            # Arguments
                epoch (int): The number of epochs
            # Returns
                lr (float32): learning rate
            """
            lr = 1e-3
            if epoch > 180:
                lr *= 0.5e-3
            elif epoch > 160:
                lr *= 1e-3
            elif epoch > 120:
                lr *= 1e-2
            elif epoch > 80:
                lr *= 1e-1
            print('Learning rate: ', lr)
            return lr


        def resnet_layer(inputs,
                        num_filters=16,
                        kernel_size=3,
                        strides=1,
                        activation='relu',
                        batch_normalization=True,
                        conv_first=True):
            """2D Convolution-Batch Normalization-Activation stack builder
            # Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization
                conv_first (bool): conv-bn-activation (True) or
                    bn-activation-conv (False)
            # Returns
                x (tensor): tensor as input to the next layer
            """
            conv = Conv2D(num_filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))

            x = inputs
            if conv_first:
                x = conv(x)
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
            else:
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
                x = conv(x)
            return x


       
        def resnet_v2(input_shape, depth, num_classes=10):
            """ResNet Version 2 Model builder [b]
            Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
            bottleneck layer
            First shortcut connection per layer is 1 x 1 Conv2D.
            Second and onwards shortcut connection is identity.
            At the beginning of each stage, the feature map size is halved (downsampled)
            by a convolutional layer with strides=2, while the number of filter maps is
            doubled. Within each stage, the layers have the same number filters and the
            same filter map sizes.
            Features maps sizes:
            conv1  : 32x32,  16
            stage 0: 32x32,  64
            stage 1: 16x16, 128
            stage 2:  8x8,  256
            # Arguments
                input_shape (tensor): shape of input image tensor
                depth (int): number of core convolutional layers
                num_classes (int): number of classes (CIFAR10 has 10)
            # Returns
                model (Model): Keras model instance
            """
            if (depth - 2) % 9 != 0:
                raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
            # Start model definition.
            num_filters_in = 16
            num_res_blocks = int((depth - 2) / 9)

            inputs = Input(shape=input_shape)
            # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
            x = resnet_layer(inputs=inputs,
                            num_filters=num_filters_in,
                            conv_first=True)

            # Instantiate the stack of residual units
            for stage in range(3):
                for res_block in range(num_res_blocks):
                    activation = 'relu'
                    batch_normalization = True
                    strides = 1
                    if stage == 0:
                        num_filters_out = num_filters_in * 4
                        if res_block == 0:  # first layer and first stage
                            activation = None
                            batch_normalization = False
                    else:
                        num_filters_out = num_filters_in * 2
                        if res_block == 0:  # first layer but not first stage
                            strides = 2    # downsample

                    # bottleneck residual unit
                    y = resnet_layer(inputs=x,
                                    num_filters=num_filters_in,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=activation,
                                    batch_normalization=batch_normalization,
                                    conv_first=False)
                    y = resnet_layer(inputs=y,
                                    num_filters=num_filters_in,
                                    conv_first=False)
                    y = resnet_layer(inputs=y,
                                    num_filters=num_filters_out,
                                    kernel_size=1,
                                    conv_first=False)
                    if res_block == 0:
                        # linear projection residual shortcut connection to match
                        # changed dims
                        x = resnet_layer(inputs=x,
                                        num_filters=num_filters_out,
                                        kernel_size=1,
                                        strides=strides,
                                        activation=None,
                                        batch_normalization=False)
                    x = tf.keras.layers.add([x, y])

                num_filters_in = num_filters_out

            # Add classifier on top.
            # v2 has BN-ReLU before Pooling
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = AveragePooling2D(pool_size=8)(x)
            y = Flatten()(x)
            outputs = Dense(num_classes,
                            activation='softmax',
                            kernel_initializer='he_normal')(y)

            # Instantiate model.
            return (inputs, outputs)

        tf.keras.backend.set_image_data_format("channels_first")

        inputs, outputs = resnet_v2(input_shape=(3,32,32), depth=47)    
        super().__init__(inputs=inputs, outputs=outputs)   

        self.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=lr_schedule(0)),
                    metrics=['accuracy'])
        

    def train(self, cifar, args):
        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
        self.tb_callback = tf.keras.callbacks.TensorBoard( args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None
        self.tb_callback.on_epoch_end = lambda epoch, logs: eval_test(args, epoch)
        images = cifar.train.data["images"]
        cutout = Cutout(1, 12)
        images = np.array([cutout(image) for image in images])
        self.fit(images,tf.keras.utils.to_categorical(cifar.train.data["labels"]),
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(
                cifar.dev.data["images"], tf.keras.utils.to_categorical(cifar.dev.data["labels"])),
            callbacks=[self.tb_callback, lr_reducer, lr_scheduler],
        )






if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50,
                        type=int, help="Batch size.")
    parser.add_argument("--epochs", default=30, type=int,
                        help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int,
                        help="Maximum number of threads to use.")

    args = parser.parse_args()
    args.block_sizes = [int(x) for x in args.block_sizes.split(",")]
    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    cifar = CIFAR10()

    tf.keras.backend.set_image_data_format("channels_first")
  

    input_layer = tf.keras.Input([CIFAR10.C, CIFAR10.H, CIFAR10.W])

    tf.keras.backend.set_image_data_format("channels_first")
    network = ResnetModel(args)
    print(network.summary())

    network.train( cifar, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=out_file)
