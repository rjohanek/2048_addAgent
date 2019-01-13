# @Author: voldikss
# @Date: 2019-01-13 10:10:30
# @Last Modified by: voldikss
# @Last Modified time: 2019-01-13 10:10:33

from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Flatten, MaxPooling2D, Dropout
from keras.layers import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model


def build_model(width, height, depth, num_filters=128):
    inputs = Input((width, height, depth))
    conv1 = Conv2D(num_filters, (2, 1), kernel_initializer="he_uniform", padding="same")(inputs)
    conv2 = Conv2D(num_filters, (1, 2), kernel_initializer="he_uniform", padding="same")(inputs)
    conv11 = Conv2D(num_filters, (2, 1), kernel_initializer="he_uniform", padding="same")(inputs)
    conv12 = Conv2D(num_filters, (1, 2), kernel_initializer="he_uniform", padding="same")(inputs)
    conv21 = Conv2D(num_filters, (2, 1), kernel_initializer="he_uniform", padding="same")(inputs)
    conv22 = Conv2D(num_filters, (1, 2), kernel_initializer="he_uniform", padding="same")(inputs)

    conv1 = LeakyReLU(alpha=0.3)(conv1)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    conv11 = LeakyReLU(alpha=0.3)(conv11)
    conv12 = LeakyReLU(alpha=0.3)(conv12)
    conv21 = LeakyReLU(alpha=0.3)(conv21)
    conv22 = LeakyReLU(alpha=0.3)(conv22)

    conv1 = MaxPooling2D()(conv1)
    conv2 = MaxPooling2D()(conv2)
    conv11 = MaxPooling2D()(conv11)
    conv12 = MaxPooling2D()(conv12)
    conv21 = MaxPooling2D()(conv21)
    conv22 = MaxPooling2D()(conv22)

    hidden = concatenate([Flatten()(conv1),
                          Flatten()(conv2),
                          Flatten()(conv11),
                          Flatten()(conv12),
                          Flatten()(conv21),
                          Flatten()(conv22)])

    x = BatchNormalization()(hidden)
    x = Activation('relu')(x)
    #
    for width in [512, 256]:
        x = Dense(width, kernel_initializer="he_uniform")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Dropout(0.3)(x)
    outputs = Dense(4, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.summary()
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model
