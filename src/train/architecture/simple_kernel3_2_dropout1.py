import tensorflow
from keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout


def architecture(input_shape):
    return tensorflow.keras.Sequential([
        tensorflow.keras.Input(shape=input_shape),
        Conv2D(16, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dropout(0.25),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
