#Kathia Bejarano ZAmora
#A01378316
#------PERCEPTRON------
import numpy as np
import csv
import pandas as pd

#Importo mis datos
dataSet = pd.read_csv('/Users/kathbejarano/Desktop/IA/Perceptron/dsPerceptron.csv')
dataSet = np.array(dataSet)
print(dataSet)

#Resultado obtenido que será nuestro punto de referencia para poder entrenar a nuestro Perceptron 
clases = dataSet[:,-1]

#Elimino mi ultima y primer columna para no confundir mis clases
dataSet=dataSet[:,1:-1]

#Defino los datos que introducire como prueba final
prueba = dataSet[10:13,:]
#-------------------------------------------------------------------------------------------------
#Activaciones

#Recordando la función 
# w1*x1 + w2*x2 +  .... + wn*xn
#La función de activación desplegará un 0 o un 1 dependiendo el caso 
def activacion(pesos,x,b):
    z = pesos * x
    if z.sum() + b > 0:
        return 1 
        
    else:
        return 0 
#-------------------------------------------------------------------------------------------------
#Entrenamietno de mi Perceptron
pesos = np.random.uniform(-1,1, size = 3)
#definimos umbral
b = np.random.uniform (-1,1)
#Actualización de mi peso
tasa_aprendizaje = 0.02 
#Mecanismo de aprendizaje, ¿cuántas veces se repite?
epocas = 100
ctdr = 0
#Ciclo de control por epoca
for epoca in range(epocas):
    errorT = 0
    ctdr = ctdr + 1
    for i in range (len(dataSet)):
        prediccion = activacion(pesos,dataSet[i],b)
        error = clases[i] - prediccion #El error debe buscar ser 0
        errorT += error**2 #Buscamos el error absoluto 
        for k in range(len(pesos)):
            pesos[k] += tasa_aprendizaje * dataSet[i][k]*error
            b += tasa_aprendizaje * error
    print(ctdr,"Porcentaje de error:",errorT*100/100, "%")
    if errorT == 0.0:
            break
    
#Prueba Final
for j in range (len(prueba)): 
    print("Los datos usados son:", prueba[j,:])
    print("Intentos: ", ctdr,"/100")
    print("Resultado:", activacion(pesos,prueba[j,:], b))
    
   
