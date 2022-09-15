# Databricks notebook source
# MAGIC %md #Actividad Básica
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
    dic_nom_par[[i]] = {'Partner': Partner[i], 'NumRandom': random.randint(0,33)}

        #print(dic_nom_par)

#Luego imprimirán los valores del diccionario (nombre de la persona y número que dijo) (Usando un bucle for)
num = []
indices = list(dic_nom_par.keys())

for i in range(len(indices)):
    norandom = dic_nom_par[indices[i]]['NumRandom']
    num.append(norandom)
    print(f'El nombre es {indices[i]} y su número aleatorio es {norandom}') 

#Al final imprimirán dos mensajes, mostrando el número más grande, y el número más pequeño que dijeron, sin el nombre del socio, sólo el número.
print('\n')
print(f'El número más alto fue {max(num)} y el más pequeño {min(num)}')



