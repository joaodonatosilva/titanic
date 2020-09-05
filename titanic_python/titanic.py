# Use machine learning to create a model that predicts which passengers survived the Titanic shipwreck
# ----------------------------- LIBRARIES  -----------------------------
# linear algebra
import numpy as np

# data processing
import pandas as pd

# data visualization
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
#import ggplot
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
import plotnine as p9

# ----------------------------- DATA READING -----------------------------
train_data = pd.read_csv("Dataset/train.csv")
train_data_shape = train_data.shape
#print("train.csv has "+str(train_data_shape)+" rows x columns")


# ----------------------------- DATA CLEANING -----------------------------
#train_data.info()
#print(train_data.describe())
#print(train_data.head())

# Age, Cabin, Embarked columns are not totally filled
print("-------------- BEGIN")
#dataset_age = pd.isnull(train_data["Age"])
#dataset_age.to_csv("../titanic/Dataset/temp.csv")
#print(dataset_age.describe())
#dataset_age.info()
print("-------------- END")

# ----------------------------- DATA ANALYSIS -----------------------------

print("------------------- Gender")
# %women survived
women = train_data.loc[train_data.Sex == 'female']["Survived"]
women_rate = (sum(women)/len(women))*100
print("% Survival of women: ",women_rate,"%")
print("Total of women: ",len(women), " / 891")

# %man survived
man = train_data.loc[train_data.Sex == 'male']["Survived"]
man_rate = (sum(man)/len(man))*100
print("% Survival of man: ",man_rate,"%")
print("Total of man: ",len(man), " / 891")

print("------------------- Ticket class")

pclass1 = train_data.loc[train_data.Pclass == 1]["Survived"]
pclass1_rate = (sum(pclass1)/len(pclass1))*100
print("Survival of Pclass1: ",pclass1_rate,"%")

pclass2 = train_data.loc[train_data.Pclass == 2]["Survived"]
pclass2_rate = (sum(pclass2)/len(pclass2))*100
print("Survival of Pclass2: ",pclass2_rate,"%")

pclass3 = train_data.loc[train_data.Pclass == 3]["Survived"]
pclass3_rate = (sum(pclass3)/len(pclass3))*100
print("Survival of Pclass3: ",pclass3_rate,"%")

print("------------------- Age")




# a = np.arange(0,80,10)
# print(a)

# a10 = train_data.loc[train_data.Age > 0 and train_data.Age < 10]["Survived"]
# print(a10)


# ----------------------------- DATA VISUALIZATION -----------------------------

plottest=(p9.ggplot(data=train_data,
                    mapping= p9.aes(x='Survived',fill='Sex'))
 + p9.geom_bar()
 + p9.theme_bw()
 + p9.ggtitle("Primeiro Plot")
 + p9.xlab('xxx')
 + p9.ylab('yyy')
 )

print(plottest)

plot2 = (p9.ggplot(data=train_data,
         mapping = p9.aes(x='Survived',fill='Sex'))
         + p9.geom_bar()
         + p9.facet_grid(' . ~ Pclass')
         )

print(plot2)


plot3 = (p9.ggplot()
         + p9.geom_point(data=train_data,
         mapping = p9.aes(x='Age',y='Fare',color='Sex')))
        # + p9.geom_line())

print(plot3)


test=px.scatter(train_data,x='Age',y='Fare',color="Sex")
plot(test)


plot4 = (p9.ggplot(data=train_data,
                   mapping = p9.aes(x='Survived',y='Pclass'))
         + p9.geom_dotplot())
print(plot4)
