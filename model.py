import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout

def build_conv3d_model(input_shape, num_classes):
    model = Sequential([
        Conv3D(32, kernel_size=(3, 3, 3), activation="relu", input_shape=input_shape),
        MaxPooling3D(pool_size=(2, 2, 2)),
        
        Conv3D(64, kernel_size=(3, 3, 3), activation="relu"),
        MaxPooling3D(pool_size=(2, 2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Model summary
model = build_conv3d_model((20, 64, 64, 3), 5)