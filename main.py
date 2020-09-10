from tensorflow.keras.layers import Dense, Dropout, Conv2D, UpSampling2D, Flatten, Input, MaxPooling2D
from tensorflow.keras.activations import softmax, sigmoid
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
import numpy as np
import sys
import tensorflow
print(tensorflow.__version__)

data_gen = ImageDataGenerator(rescale=1./255, validation_split=.3, rotation_range=90, horizontal_flip=True, vertical_flip=True)
train_data_gen = data_gen.flow_from_directory(directory="C:/Users/akash/Downloads/keras tips and tricks/classify objects in video/Data/",
                             color_mode='rgb',
                             classes=['baseball_images', 'hockey_images'],
                             class_mode='categorical',
                             batch_size=32,
                             shuffle=False,
                             interpolation='nearest',
                             subset='training')

test_data_gen = data_gen.flow_from_directory(directory="C:/Users/akash/Downloads/keras tips and tricks/classify objects in video/Data/",
                                             target_size=(256,256),
                                             color_mode='rgb',
                                             classes=['baseball_images', 'hockey_images'],
                                             class_mode='categorical',
                                             batch_size=32,
                                             shuffle=False, interpolation='nearest',
                                             subset='validation')

print("Training class info:\n Image shape: {}\n Class Indices: {}\n Class Mode: {}\n Data Format: {}\n".format(
    train_data_gen.image_shape, train_data_gen.class_indices, train_data_gen.class_mode, train_data_gen.data_format))
print(train_data_gen.num_classes)
print(train_data_gen.n)

# We will use transfer learning to accelerate our training and model accuracy.
# We are going to use ResNet 50 Layer model
input_tensor = Input(shape=(256,256,3))
baseline_model = ResNet50(weights='imagenet', include_top=False, input_tensor= input_tensor)
head_model = baseline_model.output
head_model = MaxPooling2D(pool_size=(7,7))(head_model)
head_model = Flatten(name='flatten')(head_model)
head_model = Dense(512, activation='relu')(head_model)
head_model = Dense(100, activation='relu')(head_model)
head_model = Dropout(0.4)(head_model)
head_model = Dense(train_data_gen.num_classes, activation='softmax')(head_model)

model = Model(inputs=baseline_model.input, outputs=head_model)

for layer in baseline_model.layers:
    layer.trainable = False


model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data_gen, epochs=10, steps_per_epoch=train_data_gen.n//train_data_gen.batch_size, validation_data=test_data_gen, verbose=1,
                    validation_steps=test_data_gen.n//test_data_gen.batch_size)

import matplotlib.pyplot as plt

plt.plot(history.history['val_acc'])
plt.plot(history.history['accuracy'])