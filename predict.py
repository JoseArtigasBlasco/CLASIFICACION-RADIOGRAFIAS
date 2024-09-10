
from keras._tf_keras.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras._tf_keras.keras.models import load_model

# Definir las dimensiones de la imagen
img_height = 200
img_width = 200

# Cargar el modelo guardado
model = load_model('pneumonia-chest-x-ray-cnn.h5')

# Función para cargar y procesar la imagen
def prepare_image(img_path, img_height, img_width):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Ruta de la imagen a predecir
img_path = 'dataset/imagenes_predic/normal/Normal-1.png'

# Preparar la imagen
img_array = prepare_image(img_path, img_height, img_width)

# Hacer la predicción
prediction = model.predict(img_array)

# Interpretar la predicción
if prediction[0][0] > 0.5:
    print(f"La imagen {img_path}: Neumonía")
else:
    print(f"La imagen {img_path}: Normal")

# Mostrar la imagen con la predicción
plt.imshow(image.load_img(img_path))
plt.title(f"Predicción: {'Neumonía' if prediction[0][0] > 0.5 else 'Normal'}")
plt.show()

print(prediction)