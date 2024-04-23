from keras.layers import Input, Conv2D, MaxPooling2D, Dense, concatenate, Flatten, \
    BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model


def cnn_block(input_shape):
    input_layer = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=(2, 2))(input_layer)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, kernel_size=(2, 2))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, kernel_size=(2, 2))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    x = Dense(128)(x)
    x = Activation('relu')(x)

    x = Dense(64)(x)
    x = Activation('relu')(x)

    return Model(inputs=input_layer, outputs=x)


def architecture(input_shape):
    cnn_block1 = cnn_block(input_shape)
    cnn_block2 = cnn_block(input_shape)

    merged = concatenate([cnn_block1.output, cnn_block2.output])

    x = Dense(32)(merged)
    x = Activation('relu')(x)

    x = Dense(16)(x)
    x = Activation('relu')(x)

    x = Dense(8)(x)
    x = Activation('relu')(x)

    x = Dense(4)(x)
    x = Activation('relu')(x)

    output = Dense(1)(x)

    return Model(inputs=[cnn_block1.input, cnn_block2.input], outputs=output)
