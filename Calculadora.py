# Databricks notebook source
class calculatrice:
    def __init__(self, valeurs):
        self.lista = valeurs
    
    def suma(self):
        suma=0
        for x in self.lista:
            suma += x
        return suma
    
    def resta(self):
        resta=46
        for x in self.lista:
            resta -= x
        return resta
    
    
    def multi(self):
        multi=1
        for x in self.lista:
            multi *= x
        return multi
    
    def div(self):
        div = self.lista[4]
        for x in range(4,len(self.lista)):
            div /= self.lista[x]
        return div

# COMMAND ----------

calculatrice1 = calculatrice([1,2,3,4,5,6])

# COMMAND ----------

calculatrice1.suma()

# COMMAND ----------

calculatrice1.resta()

# COMMAND ----------

calculatrice1.multi()

# COMMAND ----------

calculatrice1.div()
