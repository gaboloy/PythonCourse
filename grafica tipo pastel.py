# Databricks notebook source
#Crear y guardar una gráfica de pay del año más reciente existente en la base de datos.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path='/dbfs/mnt/dpo/AI_Factory/MonterreyDigitalHub/Modulo 12 - Actividad Competidores Olimpicos/athlete_events.csv'
df_Olimpicos = pd.read_csv(path)
df_Olimpicos.head()

dataframe = df_Olimpicos[df_Olimpicos["Year"]==df_Olimpicos["Year"].max()]
dataframe = dataframe.dropna(subset=["Medal"])
dataframe.head()

dataframe1 = dataframe.groupby(["NOC"])["Medal"].count().sort_values(ascending=False).reset_index()

dataframe1["nuevovalor"]=np.where(dataframe1["Medal"]>np.mean(dataframe1["Medal"]),dataframe1["NOC"],"Otros")

#dataframe1["nuevovalor"].value_counts().plot(kind="pie")

dataframe1=dataframe1.groupby(["nuevovalor"])["Medal"].sum().sort_values(ascending=False).reset_index()
#dataframe1

dataframe1=dataframe1[dataframe1["nuevovalor"]!="Otros"]

#dataframe1
print(dataframe1)

dataframe1["Medal"].plot(kind="pie")

# COMMAND ----------

np.mean(dataframe1["Medal"])



# COMMAND ----------


