# Databricks notebook source
# MAGIC %md #Actividad Advanced
# MAGIC 
# MAGIC --Cada compañero dirá un número. Los guardarás en un diccionario, junto con el nombre de tu pareja.
# MAGIC --Luego imprimirán los valores del diccionario (nombre de la persona y número que dijo) (Usando un bucle for)
# MAGIC --Al final imprimirán dos mensajes, mostrando el número más grande, y el número más pequeño que dijeron, sin el nombre del socio, sólo el número.

# COMMAND ----------

import random

dic_nom_par = {}

Nombre = ["César","Gaby","Felipe","Sofía"]
Partner = ["Olivia","David","José Emilio","Georgina"]

#Asigna un número aleatorio a tu compañero. Los guardarás en un diccionario, junto con el nombre de tu pareja.



for i in range(len(Nombre)):
    dic_nom_par[Nombre[i]] = {'Pareja': Partner[i], 'NoRandom': random.randint(0,10)}
#print(dic_nom_par)



#Luego imprimirán los valores del diccionario (nombre de la persona y número que dijo) (Usando un bucle for)
numeros = []
keys = list(dic_nom_par.keys())

for i in range(len(keys)):
    norandom = dic_nom_par[keys[i]]['NoRandom']
    numeros.append(norandom)
    print(f'El nombre es {keys[i]} y su número aleatorio es {norandom}')
   




 #Al final imprimirán dos mensajes, mostrando el número más grande, y el número más pequeño que dijeron, sin el nombre del socio, sólo el número.
print('\n')
print(f'El número más alto fue {max(numeros)} y el más pequeño {min(numeros)}')





# COMMAND ----------

Partner[1]

# COMMAND ----------


