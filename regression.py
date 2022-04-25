#just importing the modules here :)
import numpy as np #aka number python, used to work with high level numeric problems in python
import matplotlib.pyplot as plt #this  module is basically a library for graphs and plotting the data on graphs
import pandas as pd #used for data structural problems
from sklearn.model_selection import train_test_split #this module I guess, is specifically made for ML cuz all or most of the libraries are ML specific

dataset=pd.read_csv('D:\Linear\salary_data.csv') #please change the file location in ur accordance srry :(
x=dataset.iloc[:, :-1].values #.iloc is used to locate the values for us to plot a graph
y=dataset.iloc[:, 1].values 
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0) #defining the training and the testing set

from sklearn.linear_model import LinearRegression #importing the linear regression model from sklearn
rgrsr=LinearRegression()
rgrsr.fit(x_train,y_train)
#from here we basically plot the graph
viz_train=plt
viz_train.scatter(x_train, y_train, color="red")
viz_train.plot(x_train, rgrsr.predict(x_train), color="blue")
viz_train.title("Salary vs Experience (Training model)")
viz_train.xlabel('Years of experience')
viz_train.ylabel('Salary')
viz_train.show()

viz_test=plt
viz_test.scatter(x_test, y_test, color="red")
viz_test.plot(x_train, rgrsr.predict(x_train), color="blue")
viz_test.title("Salary vs Experience (Test model)")
viz_test.xlabel("years of experience")
viz_test.ylabel("salary")
viz_test.show()

y_pred=rgrsr.predict(x_test) #used to get the predicted values and .predict can only be used with 2D arrays but in this case x_test is already present in form of a 2D array so yeah we don't need to
print(y_pred)
