# El set de datos se puede descargar aquí  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import shutil
import cv2
from keras._tf_keras.keras.applications.vgg16 import VGG16
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import regularizers, losses, metrics
from tensorflow.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import seaborn as sns
from sklearn import metrics



# Obtiene el directorio de trabajo actual y lo asigna a data_dir
data_dir = os.getcwd()

# Construye rutas a los directorios de entrenamiento
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
val_dir = os.path.join(data_dir, 'val')

# Define rutas para las imágenes de entrenamiento normales y de neumonía.
train_normal_dir = os.path.join(train_dir, 'NORMAL')
train_pneu_dir = os.path.join(train_dir, 'PNEUMONIA')

# Define rutas para las imágenes de prueba normales y de neumonía.
test_normal_dir = os.path.join(test_dir, 'NORMAL')
test_pneu_dir = os.path.join(test_dir, 'PNEUMONIA')

# Define rutas para las imágenes de validación normales y de neumonía.
val_normal_dir = os.path.join(val_dir, 'NORMAL')
val_pneu_dir = os.path.join(val_dir, 'PNEUMONIA')


# Lista los archivos de las imágenes de entrenamiento
list_train_pneu = os.listdir(train_pneu_dir)
list_train_normal = os.listdir(train_normal_dir)

# Lista los archivos de las imágenes de prueba
list_test_pneu = os.listdir(test_pneu_dir)
list_test_normal = os.listdir(test_normal_dir)

# Lista los archivos de las imágenes de validación
list_val_pneu = os.listdir(val_pneu_dir)
list_val_normal = os.listdir(val_normal_dir)


# Genera un gráfico de barras que compara la cantidad de imágenes de neumonía y normales en el conjunto de entrenamiento.
plt.bar([1, 2], np.array([len(list_train_pneu), len(list_train_normal)]), color=['gray', 'purple'], alpha=0.5)
plt.title('Neumonia vs normal training')
plt.xticks([1, 2], ('Neumonia', 'normal'))
plt.ylabel('Count')
pd.DataFrame({'percentage': np.array([len(list_train_pneu), len(list_train_normal)]) / sum([len(list_train_pneu), len(list_train_normal)])})
plt.show()

# Selecciona aleatoriamente una muestra de imágenes de neumonía para igualar el número de imágenes normales, tanto para entrenamiento como para prueba.
train_path = train_pneu_dir
train_files = os.listdir(train_path)
train_down = random.sample(train_files, len(list_train_normal))

test_path = test_pneu_dir
test_files = os.listdir(test_path)
test_down = random.sample(test_files, len(list_test_normal))

# # Crea directorios para almacenar los datos de entrenamiento y prueba balanceados.
os.mkdir('train_down')
os.mkdir('test_down')

# Define rutas para estos nuevos directorios.
train_down_dir = os.path.join(data_dir, 'train_down')
test_down_dir = os.path.join(data_dir, 'test_down')

# Crea subdirectorios para neumonía y normales en los nuevos directorios balanceados.
os.mkdir(os.path.join(train_down_dir, 'PNEUMONIA'))
os.mkdir(os.path.join(train_down_dir, 'NORMAL'))

os.mkdir(os.path.join(test_down_dir, 'PNEUMONIA'))
os.mkdir(os.path.join(test_down_dir, 'NORMAL'))

# Copia los archivos de neumonía y normales en los directorios balanceados.
train_pneu_down_dir = os.path.join(train_down_dir, 'PNEUMONIA')
for i in train_down:
    shutil.copy(os.path.join(train_pneu_dir, i), train_pneu_down_dir)

train_normal_down_dir = os.path.join(train_down_dir, 'NORMAL')
for i in list_train_normal:
    shutil.copy(os.path.join(train_normal_dir, i), train_normal_down_dir)

test_pneu_down_dir = os.path.join(test_down_dir, 'PNEUMONIA')
for i in test_down:
    shutil.copy(os.path.join(test_pneu_dir, i), test_pneu_down_dir)

test_normal_down_dir = os.path.join(test_down_dir, 'NORMAL')
for i in list_test_normal:
    shutil.copy(os.path.join(test_normal_dir, i), test_normal_down_dir)


# Aplica corrección gamma a una imagen y la muestra.
gamma = 3
filename = os.path.join(train_normal_dir, list_train_normal[1])
img = cv2.imread(filename)
corrected_image = np.power(img, gamma)
plt.imshow(img)

# Muestra una cuadrícula de 9 imágenes de neumonía redimensionadas.
for i in range(9):
    plt.subplot(330 + 1 + i)
    filename = os.path.join(train_pneu_dir, list_train_pneu[i])
    img = cv2.imread(filename)
    img = cv2.resize(img, (200, 200))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.show()

# Crea un generador de datos de entrenamiento con varias técnicas de aumento.
datagen_train = ImageDataGenerator(rescale=1.0/255.0,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.15,
                                   fill_mode='nearest',
                                   zoom_range=0.2,
                                   brightness_range=[0.9, 1.1]
                                   )

# Crea generadores de datos de prueba y validación solo con reescalado.
datagen_test = ImageDataGenerator(rescale=1.0/255.0)
datagen_val = ImageDataGenerator(rescale=1.0/255.0)

# Crea iteradores para los conjuntos de datos de entrenamiento, prueba y validación.
batch_size = 64

train_it = datagen_train.flow_from_directory(train_down_dir,
                                             class_mode='categorical',
                                             batch_size=batch_size,
                                             shuffle=True,
                                             target_size=(224, 224),
                                             color_mode="rgb", seed=42)

test_it = datagen_test.flow_from_directory(test_down_dir,
                                           class_mode='categorical',
                                           batch_size=batch_size,
                                           shuffle=True, target_size=(224, 224),
                                           color_mode="rgb",
                                           seed=42)

val_it = datagen_val.flow_from_directory(val_dir, batch_size=1,
                                         class_mode='categorical',
                                         shuffle=False,
                                         target_size=(224, 224),
                                         color_mode="rgb",
                                         seed=42)

# Carga la base convolucional de VGG16 preentrenada.
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
print(conv_base.summary())

# Crea un modelo secuencial con la base convolucional y dos capas de clasificación.
model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

# Compila y muestra un resumen del modelo.
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("model compiled")
print(model.summary())

# Configura callbacks para reducción de la tasa de aprendizaje, detención temprana y guardado del mejor modelo.
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1,
    mode='auto',
    min_lr=0.000001)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    mode='auto')
model_checkpoint = ModelCheckpoint(
    filepath='weights.weights.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='auto')

# Calcula los pasos por época y entrena el modelo.
steps_per_epoch = train_it.samples // batch_size
validation_steps = test_it.samples // batch_size

history = model.fit(train_it,
                    epochs=40,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=test_it,
                    validation_steps=validation_steps,
                    callbacks=[reduce_lr, early_stopping, model_checkpoint])

# Carga los pesos del mejor modelo guardado y guarda el modelo final.
model.load_weights('weights.weights.h5')
model.save('pneumonia-chest-x-ray-cnn.h5')

# Grafica la pérdida y precisión del entrenamiento y validación a lo largo de las épocas.
def summarize_diagnostics(history):
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.show()
    plt.close()

summarize_diagnostics(history)

val_it.reset()

pred = model.predict(val_it, steps=val_it.n, verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)

labels = (val_it.class_indices)
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames = val_it.filenames
results = pd.DataFrame({"Filename": filenames, "Predictions": predictions})

# Crea una matriz de confusión.
def conf_matrix(matrix, pred):
    class_names = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

cnf_matrix = metrics.confusion_matrix(val_it.classes, predicted_class_indices)
conf_matrix(cnf_matrix, val_it.classes)




















