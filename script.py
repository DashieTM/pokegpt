# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow.keras as keras
import keras.layers as layers
from sklearn.model_selection import KFold 



# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Sklearn for Confusion Matrix
from sklearn.metrics import confusion_matrix

print(tf.__version__)


train_dir = 'datasets/dataset_training/'
test_dir = 'datasets/dataset_testing/'
batch_size = 50
img_height = 72
img_width = 72

training_ds, validation_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_width), # Some preprocessing happening here, resizing the images
    subset="both",
    seed=24,
    validation_split=0.2
)


testing_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_width) # Some preprocessing happening here, resizing the images
)

class_names = training_ds.class_names
num_classes = len(training_ds.class_names)
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in training_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

train_dir = 'datasets/dataset_training/'
batch_size = 50
img_height = 72
img_width = 72

o_training_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_width), # Some preprocessing happening here, resizing the images
)

train_images = np.concatenate(list(o_training_ds.map(lambda x, y: x)))
train_labels = np.concatenate(list(o_training_ds.map(lambda x, y: y)))

val_images = np.concatenate(list(testing_ds.map(lambda x, y: x)))
val_labels = np.concatenate(list(testing_ds.map(lambda x, y: y)))

inputs = np.concatenate((train_images, val_images), axis=0)
targets = np.concatenate((train_labels, val_labels), axis=0)

splits = 5
kfold = KFold(n_splits=5, shuffle=True)
num_epochs = 1
iteration = 0
scores = [None] * splits
models = [None] * splits

print("kfold setup done")


print("Starting kfold model evaltuation")
for train, test in kfold.split(inputs, targets):
    optimized_model = keras.Sequential([
        layers.Rescaling(1./255), # Some preprocessing happening here, normalizing the data
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(num_classes, activation='softmax')
    ])
     
    print(f"fitting model: {iteration}")
    
    optimized_model.compile(
        optimizer='adam', 
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc']
    )
    
    optimized_history = optimized_model.fit(
        inputs[train],
        targets[train],
        validation_data=validation_ds,
        epochs=num_epochs
    )
    
    models[iteration] = optimized_model
    scores[iteration] = optimized_model.evaluate(inputs[test], targets[test], verbose=0)
    iteration += 1

print(scores)

best_model_score = 0 
best_model = 0
iteration = 0
for selected_model in scores: 
    if selected_model[1] > best_model_score:
        best_model_score = selected_model[1]
        best_model = iteration 
    iteration += 1

optimized_model = models[best_model]

optimized_history = optimized_model.fit(
    training_ds,
    validation_data=validation_ds,
    epochs=num_epochs
)

