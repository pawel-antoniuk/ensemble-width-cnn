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


def architecture(input_shape):
    # input
    mag_input_layer = l.Input(shape=input_shape, name='in_mag')
    
    # inner layers
    mag_model = cnn_block(mag_input_layer)    
    x = l.Dense(128, activation='relu')(mag_model)
    x = l.Dense(64, activation='relu')(x)
    x = l.Dense(32, activation='relu')(x)
    x = l.Dense(16, activation='relu')(x)
    
    # output
    output_width = l.Dense(1, name='out_width')(x)
    output_location = l.Dense(1, name='out_location')(x)

    return m.Model(inputs=[mag_input_layer], outputs=[output_width, output_location])
