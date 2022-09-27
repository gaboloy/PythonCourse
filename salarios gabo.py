# Databricks notebook source
import pandas as pd

dict_1={
    "Nombre":["Ringo","John","Paul","George","Yoko"],
    "Edad":[45,34,42,38,47],
    "Salario":[12000,14000,13000,11000,10000],
    "Genero":["M","M","M","M","F"]
}

print (dict_1)

df1=pd.DataFrame(dict_1)

print(df1)

df1.describe()



# COMMAND ----------

df1.max()[2]-df1.min()[2]
