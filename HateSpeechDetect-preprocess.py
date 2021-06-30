# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1twmP15Z3CsDttjyIhuoXTD3662f9esT6

## **Pre-procesamiento y análisis léxico de los datos**

## **Carga del dataset**
"""

import pandas as pd
datapath = "http://eadfh.mdp.edu.ar/json/reddit_comments.json"
data = pd.read_json(datapath)

"""Se filtrarán únicamente las columnas necesarias

Body: cuerpo del mensaje

is_hate: etiqueta (1(si) - 0 (no))
"""

dataFilter = data.loc[:, data.columns.isin(['body', 'is_hate'])]
print(dataFilter)

"""### **Cálculo de estadísticas**

*Calcular estadísticas referedidas a las diferentes características seleccionadas. Por ejemplo, término más frecuente, término más frecuente por clase, cantidad de verbos, sustantivos, ...*

### **Distribución de la feature is_hate**

Distribución de la clase hate sobre el total de la muestra
"""

import matplotlib.pyplot as plt
print(dataFilter['is_hate'].value_counts())
plt.pie(dataFilter['is_hate'].value_counts() , autopct='%1.0f%%' , colors=['#CC3366','#6699FF'] , 
        labels=['Not Hate','Hate'] , explode =[0, 0.1] ,textprops=dict(color='black', fontsize=15 , fontweight='bold'))
plt.title('Is Hate', fontsize = 25 , fontweight='bold' );

"""Descarga de las librerías necesarias para tokenizar"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
 
nltk.download('punkt')

"""Como primera acción se realizó una tokenización de los términos presentes en la columna body por clase (is hate / is not hate).
Esta tokenización se realiza previa al preprocesamiento para poder obetener una comparación entre la aplicación de estas tareas sobre los datos procesados y los datos crudos.
"""

result_is_hate = []
result_is_not_hate=[]
 
for index, row in dataFilter.iterrows():
  if row['is_hate'] == 1.0:
    aux = word_tokenize(row['body'])
    for word in aux:
      result_is_hate.append(word)
  else:
      aux = word_tokenize(row['body'])
      for word in aux:
        result_is_not_hate.append(word)
 
print("Cantidad de tokens que pertenecen a la clase is_hate ", len(result_is_hate))
 
print("Cantidad de tokens que pertenecen a la clase is_not_hate ", len(result_is_not_hate))

"""Como segunda acción se realizó una tokenización de las filas presentes en la columna body por clase (is hate / is not hate).
Esta tokenización se realiza previa al preprocesamiento para poder obtener una comparación entre la aplicación de estas tareas sobre los datos procesados y los datos crudos.
"""

result_is_hate = []
result_is_not_hate=[]
 
for index, row in dataFilter.iterrows():
  if row['is_hate'] == 1.0:
    aux = sent_tokenize(row['body'])
    result_is_hate.append(aux)
  else:
      aux = sent_tokenize(row['body'])
      result_is_not_hate.append(aux)
 
print("Cantidad de tokens que pertenecen a la clase is_hate ", len(result_is_hate))
 
print("Cantidad de tokens que pertenecen a la clase is_not_hate ", len(result_is_not_hate))

"""### **Pre-procesamiento**

*Selecccionar diferentes alternativas de pre-procesamiento y aplicar a las características seleccionadas. Para cada alternativa aplicada, explicar brevemente por qué fue utilizada.*

Como primer tarea de preprocesamiento, se eliminan las filas con datos nulos.
"""

#Eliminar nulls (NaN)
print(dataFilter.isnull().sum())
dataFilter.dropna(inplace=True)
print(dataFilter.isnull().sum())

"""Descarga de las librerías necesarias para preprocesar"""

import nltk
import re
import string
from nltk.corpus import stopwords
 
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))

"""Creación de la función clean_text que incluye:


*   Eliminación de caracteres especiales
*   Eliminación de partes de links como http:// o www
*   Eliminación de dígitos
*   Traspaso a lower case
*   Eliminación de saltos de linea
*   Stop Words
*   Stemming









"""

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

"""Aplicación de la función clean_text sobre la feature body"""

dataFilter['body'] = dataFilter['body'].apply(clean_text)

"""Impresión de los resultados de la aplicación de clean_text"""

dataFilter.head()

"""### **Re-cálculo de las estadísticas del dataset**

*Recalcular las estadísticas previamente definidas.*

Realización de la misma aplicación de técnicas de tokenización (Reutilización de código)
"""

result_is_hate = []
result_is_not_hate=[]
 
for index, row in dataFilter.iterrows():
  if row['is_hate'] == 1.0:
    aux = word_tokenize(row['body'])
    for word in aux:
      result_is_hate.append(word)
  else:
      aux = word_tokenize(row['body'])
      for word in aux:
        result_is_not_hate.append(word)
 
print("Cantidad de tokens que pertenecen a la clase is_hate ", len(result_is_hate))
 
print("Cantidad de tokens que pertenecen a la clase is_not_hate ", len(result_is_not_hate))

result_is_hate = []
result_is_not_hate=[]
 
for index, row in dataFilter.iterrows():
  if row['is_hate'] == 1.0:
    aux = sent_tokenize(row['body'])
    result_is_hate.append(aux)
  else:
      aux = sent_tokenize(row['body'])
      result_is_not_hate.append(aux)
 
print("Cantidad de tokens que pertenecen a la clase is_hate ", len(result_is_hate))
 
print("Cantidad de tokens que pertenecen a la clase is_not_hate ", len(result_is_not_hate))

"""### **Resultados**
Al recalcular tanto los términos como las frases incluidas en el corpus analizado, se detecta que la tokenización con un preprocesamiento, reduce la cantidad de datos para trabajar con el dataset, eliminando caracteres y texto no necesario.

En cuanto a la reducción de términos se detecta que se reduce para cada clase casi un 50% de los términos.

Con respecto a las frases la reducción de encuentra dada por la eliminación de filas nulas, con una dismunución de 10 filas para la clase is_not_hate
"""
