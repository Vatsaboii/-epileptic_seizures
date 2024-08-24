import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, Xception, DenseNet121, MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
train_dir = '/Users/srivatsapalepu/seizure/output_images'
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  
    rotation_range=20, 
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    zoom_range=0.2,  
    horizontal_flip=True,  
)
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

def create_compile_model(base_model):
    base_model.trainable = False 
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

models_dict = {
    'VGG16': VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3)),
    'ResNet50': ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3)),
    'InceptionV3': InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3)),
    'Xception': Xception(weights='imagenet', include_top=False, input_shape=(150, 150, 3)),
    'DenseNet121': DenseNet121(weights='imagenet', include_top=False, input_shape=(150, 150, 3)),
    'MobileNetV2': MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3)),
    'Custom_CNN': None
}
accuracy_dict = {}
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
for model_name, base_model in models_dict.items():
    if model_name == 'Custom_CNN':
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model = create_compile_model(base_model)

    print(f"Training {model_name}...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        callbacks=[early_stopping])
    best_val_acc = max(history.history['val_accuracy'])
    accuracy_dict[model_name] = best_val_acc

accuracy_df = pd.DataFrame(list(accuracy_dict.items()), columns=['Model', 'Validation Accuracy'])
accuracy_df = accuracy_df.sort_values(by='Validation Accuracy', ascending=False)
print("\nModel Accuracy Comparison:")
print(accuracy_df)
