# Databricks notebook source
#Estoy trabajando con el avanzado

#Crear un programa en Visual Studio que me permita saber cuál es el competidor más veterano que ha recibido medalla para los NOC´s MEX, COL y ARG (oro, plata o bronce) 
#Crear un programa en Visual Studio que me permita saber cuál es el competidor más joven que ha recibido medalla de oro para los NOC´s USA y CAN
#Encuentra al competidor más ganador de la historia en medallas totales, medallas de oro, medallas de plata y medallas de broce para los NOC´s USA, CHN y RUS. Crea un csv con toda su información.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path='/dbfs/mnt/dpo/AI_Factory/MonterreyDigitalHub/Modulo 12 - Actividad Competidores Olimpicos/athlete_events.csv'
df_Olimpicos = pd.read_csv(path)
df_Olimpicos.head()

# COMMAND ----------

filtera = ["ARG", "MEX", "COL"]
dfr = df_Olimpicos [df_Olimpicos ["NOC"].isin(filtera)]
dfr = dfr.dropna(subset=["Medal"])
dfr2 = dfr.groupby(["NOC"])["Age"].transform (max)==dfr["Age"]
dfr[dfr2].sort_values(by=["Age"])



# COMMAND ----------

filterb = ["USA","CAN"]
filterc = ["Gold"]
dfr3 = df_Olimpicos [df_Olimpicos ["NOC"].isin(filterb) & df_Olimpicos ["Medal"].isin(filterc)]
dfr4 = dfr3.groupby(["NOC"])["Age"].transform (min)==dfr3["Age"]
dfr3[dfr4].sort_values(by=["Age"])


# COMMAND ----------

filterd = ["USA","CHN","RUS"]
dfr5 = df_Olimpicos [df_Olimpicos ["NOC"].isin(filterd)]
dfr5 = dfr5.dropna(subset=["Medal"])
dfr5.head()
dfr6 = dfr5.groupby(["Name","NOC"])["Medal"].count().sort_values(ascending=False).head(5).reset_index()
dfr6

dfr6.to_csv("jeuxolympiques.csv")
