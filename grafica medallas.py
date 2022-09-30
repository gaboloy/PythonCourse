# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path='/dbfs/mnt/dpo/AI_Factory/MonterreyDigitalHub/Modulo 12 - Actividad Competidores Olimpicos/athlete_events.csv'
df_Olimpicos = pd.read_csv(path)
df_Olimpicos.head()

# COMMAND ----------

df=df_Olimpicos.dropna(subset=["Medal"])

df=df.groupby(["NOC"])["Medal"].count().sort_values(ascending=False).head(10)

df=df.sort_values(ascending=True).reset_index()

df



# COMMAND ----------

xposition=list(np.arange(1,11))
labelsx=df["Medal"].to_list()



# COMMAND ----------

fig, ax=plt.subplots(figsize=(15,10))

for i in range(len(xposition)):
    
    plt.text(x = xposition[i]-1.14, y = labelsx[i]+99, s = labelsx[i], size = 8, fontstyle="oblique")
ax.grid(b = True, color ="green",
        linestyle ='-', linewidth = 0.2,
        alpha = .5)

ax.set_ylabel("#Medailles")

ax.set_xlabel("Pays")

ax.set_title("devoir de Medailles")

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')

for s in ["top", "bottom", "left", "right"]:
    ax.spines[s].set_visible(False)
    
p1=ax.bar(df["NOC"],df["Medal"],color ="red", width = .70)

# COMMAND ----------


fig, ax = plt.subplots(figsize =(16, 9))
#ax.bar_label(p1)
for i in range(len(xposition)):
    plt.text(x = xposition[i]-1.16, y = labelsx[i]+100, s = labelsx[i], size = 10, fontstyle='italic')
ax.grid(b = True, color ='grey',
        linestyle ='-', linewidth = 0.2,
        alpha = .5)
ax.set_ylabel('No. of Medals')
ax.set_xlabel('National Olympic Committee')
ax.set_title('Top 10 Countries by Medals Won')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
p1=ax.bar(df['NOC'],df['Medal'],color ='navy', width = .65)
