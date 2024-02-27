#                                                     CURSO PYTHON
# Nombres y Apellidos: Andrea Dahmar Pruna Moreno
# Fecha: 2-02-2024
#                                                    TRABAJO FINAL
# Variable Clave y Población Objetivo:
# variable_clave = "serv_hig"
# poblacion_objetivo = region == "Costa"

# Importamos las librerías
import pandas as pd # importamos la libreria pandas
import numpy as np # importamos la libreria numpy
import sklearn  # importamos el paquete "sklearn" para dividir conjuntos de datos y realizar validación cruzada
import statsmodels.api as sm # importamos para visualizar nuestros datos y resultados.
import matplotlib.pyplot as plt # importamos el módulo pyplot de la biblioteca 

# Importamos funciones necesarias para el análisis estadístico
from sklearn.model_selection import train_test_split, KFold  # importamos para preprocesar nuestros datos
from sklearn.preprocessing import StandardScaler   # importamos métricas para evaluar el rendimiento de nuestros modelos
from sklearn.metrics import accuracy_score # importamos para realizar análisis estadísticos más detallados y estimación de modelos
from sklearn.linear_model import LogisticRegressionCV # importamos la clase que implementa la regresión logística con validación cruzada
from sklearn.model_selection import cross_val_score  # importamos la función se utiliza para evaluar un estimador mediante validación cruzada
from statsmodels.discrete.discrete_model import Logit # importamos para ajustar un modelo de regresión logística
from statsmodels.tools import add_constant  # importamos la funció para agregar una columna constante de unos a una matriz de datos
from sklearn.model_selection import KFold  # importamos la clase para dividir los datos en k pliegues para validación cruzada

# Preparación de los datos
# Cargamos la base de datos
dataframe = pd.read_csv("sample_endi_model_10p.txt", sep=";")
# Eliminar filas con valores nulos en la columna "dcronica" y filtrar los datos para incluir solo niños con sexo masculino y que tengan valores válidos en la columna 'serv_hig'
costa_serv_hig = dataframe[(dataframe['region'] == 'Costa') & (dataframe['serv_hig'].isin(['Alcantarillado', 'Excusado/pozo', 'Letrina/no tiene']))]
# Calcular el conteo de niños por cada categoría de la variable 'serv_hig'
conteo_serv_hig = costa_serv_hig['serv_hig'].value_counts()
# Mostrar los resultados
print("Conteo de niños por categoría de 'serv_hig':")
print(conteo_serv_hig)

# Eliminar filas con valores no finitos en las columnas especificadas
columnas_con_nulos = ['dcronica', 'region', 'n_hijos', 'tipo_de_piso', 'espacio_lavado', 'categoria_seguridad_alimentaria', 'quintil', 'categoria_cocina', 'categoria_agua', 'serv_hig']
datos_limpios = dataframe.dropna(subset=columnas_con_nulos)

# Comprobar si hay valores no finitos después de la eliminación
print("Número de valores no finitos después de la eliminación:")
print(datos_limpios.isna().sum())
# Convertir la variable categórica serv_hig en binaria
# Convertir la variable categórica serv_hig en binaria
datos_limpios.loc[:, 'serv_hig_binario'] = datos_limpios['serv_hig'].apply(lambda x: 1 if x == 'Alcantarillado' else 0)# Filtrar los datos para incluir solo niños con sexo masculino y que tengan valores válidos en la columna 'serv_hig'
costa_serv_hig = datos_limpios[(datos_limpios['region'] == 'Costa') & (datos_limpios['serv_hig_binario'] == 1)]

# Seleccionar las variables relevantes
variables = ['n_hijos', 'region', 'sexo', 'condicion_empleo', 'serv_hig_binario']

# Filtrar los datos para las variables seleccionadas y eliminar filas con valores nulos en esas variables
for i in variables:
    datos_costa_serv_hig = costa_serv_hig[~costa_serv_hig[i].isna()]

# Agrupar los datos por sexo y tipo de servicio de higiene y contar el número de niños en cada grupo
conteo_ninos_por_servicio_higiene = costa_serv_hig.groupby(["region", "serv_hig_binario"]).size()
print("Conteo de niños por categoría de 'serv_hig':")
print(conteo_ninos_por_servicio_higiene)

# Definir las variables categóricas y numéricas
variables_categoricas = ['region', 'sexo', 'condicion_empleo']
variables_numericas = ['n_hijos']

# Crear un transformador para estandarizar las variables numéricas
transformador = StandardScaler()

# Crear una copia de los datos originales
datos_escalados = datos_limpios.copy()

# Estandarizar las variables numéricas
datos_escalados[variables_numericas] = transformador.fit_transform(datos_escalados[variables_numericas])

# Convertir las variables categóricas en variables dummy
datos_dummies = pd.get_dummies(datos_escalados, columns=variables_categoricas, drop_first=True)
# Seleccionar las variables predictoras (X) y la variable objetivo (y)
X = datos_dummies[['n_hijos', 'sexo_Mujer', 
                   'condicion_empleo_Empleada', 'condicion_empleo_Inactiva', 'condicion_empleo_Menor a 15 años']]
y = datos_dummies["serv_hig_binario"]
# Definir los pesos asociados a cada observación
weights = datos_dummies['fexp_nino']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Asegurar que todas las variables sean numéricas
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

# Convertir las variables a tipo entero
variables = X_train.columns
for i in variables:
    X_train[i] = X_train[i].astype(int)
    X_test[i] = X_test[i].astype(int)

y_train = y_train.astype(int)

# Ajustar el modelo de regresión logística
modelo = sm.Logit(y_train, X_train)
result = modelo.fit()
print(result.summary())

# Extraemos los coeficientes y los almacenamos en un DataFrame
coeficientes = result.params
df_coeficientes = pd.DataFrame(coeficientes).reset_index()
df_coeficientes.columns = ['Variable', 'Coeficiente']

# Creamos una tabla pivote para una mejor visualización
df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
df_pivot.reset_index(drop=True, inplace=True)

# Realizamos predicciones en el conjunto de prueba
predictions = result.predict(X_test)
# Convertimos las probabilidades en clases binarias
predictions_class = (predictions > 0.5).astype(int)
# Comparamos las predicciones con los valores reales
comparacion = (predictions_class == y_test)
# Definir el número de folds para la validación cruzada
kf = KFold(n_splits=100)
accuracy_scores = []  # Lista para almacenar los puntajes de precisión de cada fold
df_params = pd.DataFrame()  # DataFrame para almacenar los coeficientes estimados en cada fold

# Iterar sobre cada fold
for train_index, test_index in kf.split(X_train):
    # Dividir los datos en conjuntos de entrenamiento y prueba para este fold
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    weights_train_fold, weights_test_fold = weights_train.iloc[train_index], weights_train.iloc[test_index]
    
    # Ajustar un modelo de regresión logística en el conjunto de entrenamiento de este fold
    log_reg = sm.Logit(y_train_fold, X_train_fold)
    result_reg = log_reg.fit(method='bfgs', maxiter=1000)

    # Extraer los coeficientes y organizarlos en un DataFrame
    coeficientes = result_reg.params
    df_coeficientes = pd.DataFrame(coeficientes).reset_index()
    df_coeficientes.columns = ['Variable', 'Coeficiente']
    df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
    df_pivot.reset_index(drop=True, inplace=True)
    
    # Realizar predicciones en el conjunto de prueba de este fold
    predictions = result_reg.predict(X_test_fold)
    predictions = (predictions >= 0.5).astype(int)
    
    # Calcular la precisión del modelo en el conjunto de prueba de este fold
    accuracy = accuracy_score(y_test_fold, predictions)
    accuracy_scores.append(accuracy)
    
    # Concatenar los coeficientes estimados en este fold en el DataFrame principal
    df_params = pd.concat([df_params, df_pivot], ignore_index=True)

# Calcular la precisión promedio de la validación cruzada
mean_accuracy = np.mean(accuracy_scores)
print(f"Precisión promedio de validación cruzada: {mean_accuracy}")
# Calcular la precisión promedio
precision_promedio = np.mean(accuracy_scores)

# Crear el histograma
plt.hist(accuracy_scores, bins=30, edgecolor='black')

# Añadir una línea vertical en la precisión promedio
plt.axvline(precision_promedio, color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la precisión promedio
plt.text(precision_promedio - 0.1, plt.ylim()[1] - 0.1, f'Precisión promedio: {precision_promedio:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

# Configurar el título y etiquetas de los ejes
plt.title('Histograma de Accuracy Scores')
plt.xlabel('Accuracy Score')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()
# Crear el histograma de los coeficientes para la variable "n_hijos"
plt.hist(df_params["n_hijos"], bins=30, edgecolor='black')

# Calcular la media de los coeficientes para la variable "n_hijos"
media_coeficientes_n_hijos = np.mean(df_params["n_hijos"])

# Añadir una línea vertical en la media de los coeficientes
plt.axvline(media_coeficientes_n_hijos, color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la media de los coeficientes
plt.text(media_coeficientes_n_hijos - 0.1, plt.ylim()[1] - 0.1, f'Media de los coeficientes: {media_coeficientes_n_hijos:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

# Configurar título y etiquetas de los ejes
plt.title('Histograma de Beta (N Hijos)')
plt.xlabel('Valor del parámetro')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()