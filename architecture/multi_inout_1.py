from keras import layers as l
from keras import models as m


def cnn_block(input_layer):
    x = l.Conv2D(32, kernel_size=(2, 2), activation='relu', padding='same')(input_layer)
    x = l.MaxPooling2D(pool_size=(2, 3))(x)
    x = l.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same')(x)
    x = l.MaxPooling2D(pool_size=(2, 3))(x)
    x = l.Conv2D(128, kernel_size=(2, 2), activation='relu', padding='same')(x)
    x = l.MaxPooling2D(pool_size=(2, 2))(x)
    x = l.Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same')(x)
    x = l.MaxPooling2D(pool_size=(2, 2))(x)
    x = l.GlobalAveragePooling2D()(x)
    return x


def gcc_phat_block(input_layer):
    x = l.Dense(32, activation='relu')(input_layer)
    return x


def architecture(input_shape):
    mag_input_layer = l.Input(shape=input_shape, name='in_mag')
    # phase_input_layer = l.Input(shape=input_shape, name='in_phase')
    # gcc_input_layer = l.Input(shape=(66,), name='in_gcc')

    mag_model = cnn_block(mag_input_layer)
    # phase_model = cnn_block(phase_input_layer)
    # gcc_model = gcc_phat_block(gcc_input_layer)
    merged = l.concatenate([mag_model])

    x = l.Dense(64, activation='relu')(merged)
    x = l.Dense(32, activation='relu')(x)
    x = l.Dense(16, activation='relu')(x)

    output_width = l.Dense(1, name='out_width')(x)
    output_location = l.Dense(1, name='out_location')(x)

    return m.Model(inputs=[mag_input_layer], outputs=[output_width, output_location])
