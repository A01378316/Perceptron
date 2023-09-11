#Kathia Bejarano Zamora
#A01378316
#------PERCEPTRON------

#Importamos librerias 
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

#********************* IMPORTACIÓN Y TRATAMIENTO DE MI DATASET ******************************

#Importo mis datos (set de datos obtenidos de Kaggle)
dataSet = pd.read_csv('/Users/kathbejarano/Desktop/IA/Perceptron/dsPerceptron.csv')

#Lo convertimos en array
dataSet = np.array(dataSet)

#Resultado obtenido que será nuestro punto de referencia para poder entrenar a nuestro Perceptron 
clases = dataSet[:,-1]

#Elimino mi ultima y primer columna para no confundir mis clases porque
#la primer columan sólo es la iteración y la última columna son los resultados esperados
dataSet=dataSet[:,1:-1]

#Defino mi "train_set"
train= dataSet[0:5,:]

#Defino mi "validate set"
validate = dataSet[6:9,:]

#Defino mi "test_set"
prueba = dataSet[10:13,:]



#*****************************CREACIÓN DEL MODELO*****************************************
#Activaciones
#Recordando la función base del perceptron: w1*x1 + w2*x2 +  .... + wn*xn
#La función de activación desplegará un 0 o un 1 dependiendo el caso 
#Creo mi función "activación" dada la función del perceptron

def activacion(pesos,x,b):
    z = pesos * x
    if z.sum() + b > 0:
        return 1 
        
    else:
        return 0 
#-------------------------------------------------------------------------------------------------
#Entrenamietno de mi Perceptron

#Defino mis pesos de manera aleatoria (son los que se irán ajustando)
pesos = np.random.uniform(-1,1, size = 3)
#Definimos mi umbral (bias)
b = np.random.uniform (-1,1)
#Actualización de mi peso (cada ajuste será de 0.02)
tasa_aprendizaje = 0.02 
#Mecanismo de aprendizaje, ¿cuántas veces se repite?
epocas = 100
#Incializo un contador para ver cuantas veces repitió el proceso
ctdr = 0

#Inicializo una lista para almcenar el error
error_acumulado = []

#Inicializo mis variables que me ayudarán a obtener mis métricas de evaluación 
Vpositivo = 0 #Contabilizara los Positivos que fueron correctos
Fpositivo = 0#Contabilizara los Positivos que son incorrectos
Vnegativo = 0#Contabilizara los Negativos que son correctos
Fnegativo = 0#Contabilizara los Negativos que son incorrectos

#Ciclo de control por epoca
for epoca in range(epocas):
    errorT = 0
    ctdr = ctdr + 1

    for i in range(len(dataSet)):
        prediccion = activacion(pesos, dataSet[i], b)
        error = clases[i] - prediccion
        errorT += error ** 2
        # Actualizar las métricas de acuerdo a las predicciones, ir evaluando y añadiendo a las listas según si fue predicción buena o mala
        if clases[i] == 1 and prediccion == 1:
            Vpositivo += 1
        elif clases[i] == 0 and prediccion == 1:
            Fpositivo += 1
        elif clases[i] == 0 and prediccion == 0:
            Vnegativo += 1
        elif clases[i] == 1 and prediccion == 0:
            Fnegativo += 1
        for k in range(len(pesos)):
            pesos[k] += tasa_aprendizaje * dataSet[i][k] * error
            b += tasa_aprendizaje * error
    #Calculo el error en porcentaje
    errorP = errorT * 100 / 100  
    #En la lista creada vamos añadiendo los porcentajes, un histórico 
    error_acumulado.append(errorP) 
    print(ctdr, "Porcentaje de error:", errorP, "%")
    #Mi ciclo se detiene cuando ya haya aprendido y no tenga errores
    if errorT == 0.0:
        break

#Calcular métricas de evaluación, en este caso utilizaremos 3
#Calculo de la precisión VP/totalV
precision = Vpositivo / (Vpositivo + Fpositivo)
#Calculo del recall VP/VP + FN
recall = Vpositivo / (Vpositivo + Fnegativo)
#Exactitud TotalV/tamaño de dataset
accuracy = (Vpositivo + Vnegativo) / len(dataSet)

#Impresión de mis métricas
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
    
#Prueba Final en donde correre finalmente mi test
for j in range (len(prueba)): 
    print("Los datos usados son:", prueba[j,:])
    print("Intentos: ", ctdr,"/100")
    print("Resultado:", activacion(pesos,prueba[j,:], b))

#***************************VISUALIZADOR DE MI PERCEPTRON DE MANERA GRÁFICA*********************************
#Creo mi figura para que se pueda desplegar en una misma ventana, en este caso presentaré dos gráficas
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Curva de aprendizaje
axs[0].plot(range(1, ctdr + 1), error_acumulado)
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Error Percentage')
axs[0].set_title('Curva de Aprendizaje')
axs[0].grid(True)

# Métricas de evaluación
labels = ['Precision', 'Recall', 'Accuracy']
values = [precision, recall, accuracy]
axs[1].bar(labels, values)
axs[1].set_ylabel('Valor')
axs[1].set_title('Métricas de Evaluación')
axs[1].set_ylim(0, 1)  # Asegurar que el eje vertical está en el rango [0, 1] para las métricas de porcentaje

#Mostrar mis gráficas
plt.tight_layout()
plt.show()


   
