from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, GlobalAveragePooling2D, concatenate
from tensorflow.keras.models import Model


def cnn_block(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(2, 2), activation="relu")(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(2, 2), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(2, 2), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, kernel_size=(2, 2), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(512, kernel_size=(2, 2), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=input_layer, outputs=x)


def architecture(input_shape):
    cnn_block1 = cnn_block(input_shape)
    cnn_block2 = cnn_block(input_shape)

    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)

    output_from_block1 = cnn_block1(input1)
    output_from_block2 = cnn_block2(input2)

    merged = concatenate([output_from_block1, output_from_block2])

    x = Dense(512, activation='relu')(merged)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(1)(x)

    return Model(inputs=[input1, input2], outputs=output)
