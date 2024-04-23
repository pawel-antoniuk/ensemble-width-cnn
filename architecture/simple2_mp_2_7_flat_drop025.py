import tensorflow
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


def architecture(input_shape):
    return tensorflow.keras.Sequential([
        tensorflow.keras.Input(shape=input_shape),
        Conv2D(32, kernel_size=(2, 2), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(2, 2), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(2, 2), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, kernel_size=(2, 2), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(512, kernel_size=(2, 2), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.25),
        Dense(256, activation='relu'),
        Dropout(0.25),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
