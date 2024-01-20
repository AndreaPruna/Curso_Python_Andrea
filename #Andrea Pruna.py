#Variable de texto
mi_variable = "Hola soy Andrea y esta es mi primera práctica en python"
print(mi_variable)
#Vector con mis números favoritos
mis_numfav=[7,16,21,28,29]
print(mis_numfav)
#Diccionario de información de uno de mis libros favoritos
mi_diccionario = {"Título":"After", "Género":"Novela Rosa", "Publicación":"2014"}
print(mi_diccionario)
#Existen 3 tipos de diccionarios enteros, flotantes(decimales), numeros complejos(entera con parte imaginaria), boolean(logico verdadero o falso).

########
#Declarar un vector de enteros, señalando el número de hermanos que tengo y que se repita 7
##NUMERICA
números_hermanos = [1]*7
print(números_hermanos)
#Declaramos un vector de flotantes, que se repita 5 veces el valor de mi altura
vector_altura = [1.58]*5
print(vector_altura)
##siempre se asigna algo es decir "entero" se le asigna el vector flotantes y asi debe ser todo 
diccionario = {"entero" : números_hermanos, "flotante" : vector_altura, "complejo" : vector_altura}
print(diccionario)

########
##CADENAS NOMBRES DE PROGRAMACION, ETIQUETAS
cadena_simple = "EstudiO economía"
cadena_doble = ["Estoy en sexto semestre", "Me gusta mucho"]
print(cadena_doble)

#Dataframe es la estructura mas importante en pandas tiene dos dimensiones de datos etiquetados; rows(observaciones); columnas(variables)
### DataFrame
#La liberia pandas ayuda a trabajar con dataframme
##Importamos la libreria a usar
import pandas as pd
# Creamos una DataFrame con datos de ventas mensuales
datos_ventas = {
    'Vendedor': ['Juan', 'Maria', 'Carlos', 'Ana'],
    'Producto_A (ventas)': [50, 30, 40, 25],
    'Producto_B (ventas)': [20, 40, 30, 25],
    'Producto_C (ventas)': [10, 15, 20, 30]
}

df_ventas = pd.DataFrame(datos_ventas)
print(df_ventas)

#LIBRERIAS
import pandas as pd
### no valio poner "DATA" nates de ventas porque ya estoy en la carpeta de data
imp_sri=  pd.read_excel("ventas_SRI.xlsx")
print(imp_sri)

#la lectura de pandas se utilizan las funciones read_*
