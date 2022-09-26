# Databricks notebook source
import math

def eq (a,b,c):
    if (b**2-(4*a*c))<0:
        print ("la solución tiene número complejos")
        return
    x1= (-b-math.sqrt((b**2)-(4*a*c)))/(2*a)
    x2= (-b+math.sqrt((b**2)-(4*a*c)))/(2*a)
    print("las soluciones y arg de le ecuacion [x1,x2,[abc]]")
    return round(x1,3), round (x2,3), [a,b,c]

    

        

# COMMAND ----------

y=eq (6,444,2)

y
