### IMPORTACIÓN DE LIBRERIAS PARA PODER MOVERNOS ENTRE DIRECTORIOS DENTRO DE NUESTRO S.O.

import sys
import os

### IMPORTARCIÓN DE TENSOR FLOW Y SUS EXTENSIONES

import tensorflow as tf

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator  # Nos va a ayudar al procesamientos de las imágenes
from tensorflow.python.keras import optimizers # Los optimizadores de entrenamiento
from tensorflow.python.keras.models import Sequential # Librería que nos permite realizar redes neuronales secuenciales
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D # Usadas en nuetras capas
from tensorflow.python.keras import backend as K # Para poder terminar una sesión de Keras que se esté ejecutando

tf.compat.v1.disable_eager_execution()

K.clear_session() # Matamos las posibles sesiones de Keras existentes en nuestro equipo



### UBICACIÓN DE LAS IMÁGENES DE ENTRENAMIENTO Y TEST

data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'



### PARÁMETROS A USAR

epocas = 20 # Número de veces que vamos a repetir el set de datos
altura, anchura = 150, 150 # El tamaño que le vamos a dar a nuestras imágenes en píxeles
batch_size = 32 # El tamaño que le vamos a dar a nuestras imágenes en píxeles

pasos = 1000 # En número de veces que se repetirá la reproducción de una imagen por época
validation_steps = 300 # Al final de cada época se recorrerán 300 pasos para comprobación

filtros_Convolucion_1 = 32 # La profuncidad que van a tener nuestras imágenes en la 1º convolución
filtros_Convolucion_2 = 64 # La profuncidad que van a tener nuestras imágenes en la 2º convolución

tamano_filtro_1 = (3,3) # Tamaño del filtro con N altura y N anchura, en el 1º filtro
tamano_filtro_2 = (2,2) # Tamaño del filtro con N altura y N anchura, en el 2º filtro

tamano_pool = (2, 2) # Tamaño del Max Pooling con N altura y N anchura

clases = 5 # Los tipos de envases que separamos

lr = 0.0004 # El learning read



### PREPARAMOS LAS IMÁGENES

entrenamiento_datagen = ImageDataGenerator(
    rescale = 1./255, # Cada píxel tiene un rango de 0 a 155, con este calculo lo reducimos entre 0 y 1
    shear_range = 0.2, # Inclina la imagen
    zoom_range = 0.2, # Realiza un zoom en alguna imágenes
    horizontal_flip = True # Invierte una imágenes como en espejo
)

test_datagen = ImageDataGenerator(
    rescale = 1. / 255
)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(  # Procesa nuestro set de imágenes de entrenamiento
    data_entrenamiento,
    target_size = (altura, anchura),
    batch_size = batch_size,
    class_mode = 'categorical'
)

validacion_generador = test_datagen.flow_from_directory( # Procesa nuestro set de imágenes de test
    data_validacion,
    target_size = (altura, anchura),
    batch_size = batch_size,
    class_mode = 'categorical'
)



### CREAMOS LA RED CNN

cnn = Sequential() # Indicamos que nuestra red es secuencial formada por varias capas apiladas entre ellas

cnn.add( # PRIMERA CAPA
    Convolution2D( # Capa de convolución
        filtros_Convolucion_1, 
        tamano_filtro_1, 
        padding = "same", # El filtro de esquinas
        input_shape = (altura, anchura, 3), # Las características de la imágenes que vamos a entregar (altura, anchura y numero de canales RGB=red,green,blue)
        activation = 'relu' # El tipo de función de activación
    )
) 

cnn.add( # SEGUNDA CAPA
    MaxPooling2D( # Capa de Max Pooling a continuación de la de convolución
        pool_size = tamano_pool
    )
)

cnn.add( # TERCERA CAPA
    Convolution2D( # Otra capa de convolución
        filtros_Convolucion_2, 
        tamano_filtro_2, 
        # El input_shape solo lo usamos en la primera capa de convolución (en este caso no lo usamos)
        padding = "same"
    )
)

cnn.add( # CUARTA CAPA
    MaxPooling2D( # Otra capa de Max Pooling a continuación de la de convolución
        pool_size = tamano_pool
    )
)



### FASES DE CLASIFICACIÓN DE LA CCN:

cnn.add( # QUITA CAPA
    Flatten() # Aplana las imágenes que son muy profundas pero muy pequeñas tras el anterior proceso
)

cnn.add( # SEXTA CAPA
    Dense( # Crea una capa de neuronas en conexión con la capa anterior
        256, # El número de neuronas
        activation = 'relu'
    )
)

cnn.add( # SÉPTIMA CAPA
    Dropout(0.5) # Apaga el 50% de las neuronas de la capa anterior en cada paso para evitar sobreajustes, y que nuestra red no aprenda un camino expecifico para clasificar y así ser más versatil
)

cnn.add( # OCTAVA CAPA
    Dense(
        clases, # Las N clases de imágenes que hemos indicado al principio
        activation = 'softmax' # Indica el porcentaje de acierto, asumiento que el que tenga mayor categoría numérica es el correcto
    )
)

cnn.compile(
    loss = 'categorical_crossentropy', # NOVENA CAPA, capa para optimizar nuestro algoritmo,  La función de perdida, para comprender cómo de acertado está procesando
    optimizer = optimizers.Adam(lr=lr), # El optimizador, con el learning read inidicado al principio
    metrics = ['accuracy'] # La métrica, el porcentaje de imágenes bien clasificadas
)



### ENTRENAR NUESTRO ALGORITMO

cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch = pasos, # Número de pasos en cada una de las épocas
    epochs = epocas, # Número de épocas
    validation_data = validacion_generador,
    validation_steps = validation_steps # Los últimos pasos de validación antes de pasar a la siguiente época
)



### COMPROBACIONES

print(entrenamiento_generador.class_indices) ## Imprimimos por pantalla los pesos para luego usarlos y configurar la salida en predict.py



### GUARDADO DEL MODELO

target_dir = './modelo/' # Donde guardar nuestro modelo ya entrenado para usos futuros

if not os.path.exists(target_dir): # Si no existe el directorio "modelo" lo genera
    os.mkdir(target_dir)

cnn.save('./modelo/modelo.h5') # Guarda la estructura de nuestro modelo

cnn.save_weights('./modelo/pesos.h5') # Guarda los pesos de cada una de las capas