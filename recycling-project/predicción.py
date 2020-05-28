### IMPORTAMOS LAS LIBRERÍAS

import numpy as np
import tensorflow as tf
from tensorflow import keras 
from keras.initializers import glorot_uniform 
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model



### TAMAÑO DE LAS IMÁGENES DE ENTRADA

altura, anchura = 150, 150



### UBICACIÓN DEL MODELO YA ENTRENADO

modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'



### CARGAMOS LOS ARCHIVOS NECESARIOS PARA REUTILIZAR LA CNN YA ENTRENADA

cnn = load_model(modelo) # Carga del Modelo 
cnn.load_weights(pesos_modelo) # Carga de los Pesos



### FUNCIÓN DE PREDICCIÓN

def predict(file): # Recibe el nombre de la imagen
  x = load_img(file, target_size=(altura, anchura)) # Cargamos la imagen con la altura y anchura ya indicada
  x = img_to_array(x) # Cargamos el array que representa nuestra imagen
  x = np.expand_dims(x, axis=0) # En la primera dimensión (eje cero) añadimos otra dimensión extra, para procesar nuestra información sin problemas
  
  array = cnn.predict(x) # LLamamos a nuestra CNN para que haga la predicción sobre nuestra imágen (ahora denominada "x")
    # Este array contiene 2 dimensiones => [1,0,0], en el cual el algoritmo piensa que la posición del "1" es la correcta
  
  resultado = array[0] # Este siguiente paso solo utiliza la dimensión 0 del array anterior, que es la que nos interesa
  
  respuesta = np.argmax(resultado)# Alberga la posción de valor más alto dentro de array, pongamos un ejemplo:
    # [1,0,0] respuesta será igual a 0
    # [0,1,0] respuesta será igual a 1
    # [0,0,1] respuesta será igual a 2

  if respuesta == 0:
    print('Es un brick, va al contenedor amarillo.')
  elif respuesta == 1:
    print('Es cartón, va al contenedor del cartón.')
  elif respuesta == 2:
    print('Es una lata de conserva, va al contenedor amarillo.')
  elif respuesta == 3:
    print('Es papel, va al contenedor del papel.')
  elif respuesta == 4:
    print('Es una lata de refresco, va al contenedor amarillo.')
  return respuesta



### INDICAMOS LA IMAGEN QUE QUEREMOS PREDECIR

predict('aswefdasdf.jpg')