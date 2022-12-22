#Se va a proceder a Representar los datos correspondientes a 2D
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#En primer lugar, exportamos los valores que hemos generado
df = pd.read_csv('C:/Users/juanj/OneDrive/Escritorio/Practica3/TwoDim/log.csv')
#Creamos el mapa de calor

plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)
plt.title("Correlación entre las variables", fontsize = 22)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()