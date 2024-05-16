from keras.layers import Input, Conv2D, MaxPooling2D, Dense, concatenate, Activation, GlobalAveragePooling2D, Conv1D, MaxPooling1D, Dropout, Flatten
from keras.models import Model


def cnn_block(input_layer):
    x = Conv2D(32, kernel_size=(2, 2), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(2, 2), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(2, 2), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    return x


def gcc_phat_block(input_layer):
    x = Dense(32, activation='relu')(input_layer)
    return x


def architecture(input_shape):
    mag_input_layer = Input(shape=(150, 349, 2))
    phase_input_layer = Input(shape=(150, 349, 2))
    gcc_input_layer = Input(shape=(66,))
    mag_model = cnn_block(mag_input_layer)
    # phase_model = cnn_block(phase_input_layer)
    # gcc_model = gcc_phat_block(gcc_input_layer)
    # merged = concatenate([mag_model])

    output = Dense(1)(mag_model)

    return Model(inputs=[mag_input_layer, phase_input_layer, gcc_input_layer], outputs=output)
