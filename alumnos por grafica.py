# Databricks notebook source
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/dbfs/mnt/dpo/AI_Factory/MonterreyDigitalHub/Modulo 17 - Alumnos por grado/clean_students_complete.csv')

df.drop(['Unnamed: 0'], axis = 1, inplace=True)

grado = df['grade'].value_counts().to_frame()
grado.columns = ['Cantidad']

grado






# COMMAND ----------

index=list(grado.index)

index

# COMMAND ----------

fig, ax=plt.subplots(figsize=(15,10))



ax.set_ylabel("Cantidad")

ax.set_xlabel("grade")

ax.set_title("étudiants per niveau")

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')

for s in ["top", "bottom", "left", "right"]:
    ax.spines[s].set_visible(False)
    
p1=ax.bar(index,grado["Cantidad"],color ="red", width = .70)
