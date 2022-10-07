# Databricks notebook source
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('/dbfs/mnt/dpo/AI_Factory/MonterreyDigitalHub/Modulo 17 - Alumnos por grado/clean_students_complete.csv')



# COMMAND ----------

conditions1= [
    df["reading_score"]<70,
    (df["reading_score"]>=70)&(df["reading_score"]<df["reading_score"].median()),
    (df["reading_score"]>=df["reading_score"].median())&(df["reading_score"]<95),
    (df["reading_score"]>=95)&(df["reading_score"]<df["reading_score"].max()),
    df["reading_score"]==df["reading_score"].max()
]


# COMMAND ----------

df["math_score"].median()


# COMMAND ----------

conditions2= [
    df["math_score"]<67.89,
    (df["math_score"]>=67.89)&(df["math_score"]<df["math_score"].median()),
    (df["math_score"]>=df["math_score"].median())&(df["math_score"]<93.2),
    (df["math_score"]>=93.2)&(df["math_score"]<df["math_score"].max()),
    df["math_score"]==df["math_score"].max()
]


# COMMAND ----------

valeurs=[
    "pas suffisant",
    "moyennement suffisant",
    "suffisant",
    "plus que suffisant",
    "génie"
]

# COMMAND ----------

df["cat_read"]=np.select(conditions1,valeurs)
df["cat_math"]=np.select(conditions2,valeurs)

df.head()

# COMMAND ----------

calificaciones = df.groupby(["school_name","grade","gender","cat_read",]).agg({"Student ID":"count"}).reset_index()

calificaciones = calificaciones[calificaciones["cat_read"]=="génie"]

# COMMAND ----------

#df["last_name"] = df["student_name"].map(lambda x: x.split(" ")[1])
#af = df.groupby(["school_name","last_name","transform_reading_score"]).agg({"Student ID":"count"}).reset_index()
#af = af[af["transform_reading_score"]=="excelente"]



calificaciones["index"] = calificaciones["school_name"]+" "+calificaciones["grade"]



calificaciones = calificaciones.sort_values("Student ID").tail(50)

# COMMAND ----------


plt.figure(figsize = (33,8))


plt.xticks(rotation=45)


sns.barplot(data=calificaciones, x="index", y="Student ID")
