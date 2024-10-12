# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image

# constants
directory_path = "/Users/sasha/Desktop/Year 3/MAC RoboMaster Code/CV Team Challenge/train"
batch_size = 32

# seemed like every image was 224x224
img_height = 224
img_width = 224
seed = 42

# Parse data for classification
armor_plate_class = {}

# files are named with these conventions
n = 73
m = 14
for i in range(0, n):
    for j in range(0, m):

        # parsing for filenames
        i_string = f"0{i}" if i < 10 else f"{i}" 
        j_string = f"0{j}" if j < 10 else f"{j}"  
        imageFile = directory_path + f"/0000{i_string}_0{j_string}.png"
        textFile = directory_path + f"/0000{i_string}_0{j_string}.txt"

        # make sure files exist before parsing
        if os.path.exists(imageFile) and os.path.exists(textFile):
            with open(f'{textFile}', 'r') as file:

                # looked at the text files and realized the first (and only character) should represent the class
                binary_class = int(file.read(1)) # take this in and cast to integer
                armor_plate_class[imageFile] = binary_class # keep track of every image - class

# convert dictionary into two lists for easier use
image_paths = list(armor_plate_class.keys())
labels = list(armor_plate_class.values())

# function for loading, resizing, and normalizing images
def process_image(image_path, label):
    
    # open image up (final output is PIL Image) and convert tensor (image_path) into a numpy array (resize in case any images are different sizes)
    img = Image.open(image_path.numpy().decode()).resize((img_height, img_width))

    # converts PIL image (img) now into a numpy array and normalize for RGB
    img = np.array(img) / 255.0 

    # return image as expected types for tensorflow
    return img.astype(np.float32), np.array(label, dtype=np.int32)

# function for calling process_image and preparing data for tensorflow specifically
def process_image_tf(image_path, label):

    # run process_image and receieve expected tensor
    img, label = tf.py_function(func=process_image, inp=[image_path, label], Tout=[tf.float32, tf.int32])
    
    # # shape information is lost with tensorfication, so explicitly reset
    img.set_shape((img_height, img_width, 3))
    label.set_shape(())
    
    # return image and label, now prepared for model
    return img, label


#create a Tensorflow dataset from our image paths and labels and apply preprocessing to them
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(process_image_tf)

# split dataset into training (80%) and validation (20%)
val_size = int(0.2 * len(image_paths)) 

# divide datasets into batches of 32, shuffle all elements, cache for improved efficiency and prefetch for optimizied pipeline performance
train_ds = dataset.skip(val_size).batch(batch_size).shuffle(n * m).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = dataset.take(val_size).batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# model uses 3 convolution layers (standard for image processing) with RELU activation and pooling layers to reduce complexity
# output with sigmoid (generally good for binary classification output layer)
model = Sequential([
    Input(shape=(img_height, img_width, 3)), 
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid') 
])

# compile with adam optimizer (since its usually the best) and compute for cross entropy loss
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

# save final checkpoint
checkpoint_callback = ModelCheckpoint(
    filepath='final_model_checkpoint.keras', 
    save_best_only=False,  
    save_weights_only=False, 
    mode='auto',  
    verbose=1  # keep progress bar
)


# Print model summary
print(model.summary())

# 10 epochs is generally good enough for a task like this
epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[checkpoint_callback])

# plotting everything
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Loss')

plt.show()
