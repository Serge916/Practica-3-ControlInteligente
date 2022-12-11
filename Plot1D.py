#Programa para plotear distintos valores de ruido.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Cargamos los datos que se han recogido en el csv generado
df = pd.read_csv('C:/Users/juanj/OneDrive/Escritorio/Practica-3/log.csv')

#Configuramos el mapa de calor
plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(df.corr(), xticklabels = df.corr().columns, yticklabels = df.corr().columns, cmap = 'RdYlGn', center = 0, annot = True)

#Decoración argumento
plt.title("Correlación entre las variables", fontsize = 22)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()




