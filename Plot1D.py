#Programa para plotear distintos valores de ruido.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


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

#Representación de la posición en función de K.
sns.jointplot(data=df, x='x', y='K')

#Decoración del argumento
plt.title("Relación entre las variables x y K")
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

#Representación Figura 3
sns.pairplot(df, hue = "x", palette='coolwarm')
plt.show()



