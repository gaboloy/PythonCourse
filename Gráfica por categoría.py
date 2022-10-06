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

calificaciones = df.groupby(["gender","cat_read","cat_math"]).agg({"Student ID":"count"}).reset_index()

# COMMAND ----------

sns.barplot(data=calificaciones, x="cat_read", y="Student ID", hue="gender")

# COMMAND ----------

sns.barplot(data=calificaciones, x="cat_math", y="Student ID", hue="gender")
