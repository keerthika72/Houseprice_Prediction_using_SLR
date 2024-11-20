# importing all the required libraries:
import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

#Loading the dataset:
dataset = pd.read_csv(r"C:\Users\monik\Downloads\SLR - House price prediction\House_data.csv")


space = dataset["sqft_living"]
price=dataset["price"]
x = np.array(space).reshape(-1,1) # x is the independent variable.
y = np.array(price) # y is the dependent varibale{price}


# Now splitting the dataset for the training and the testing:
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.25,random_state=0) #Random state = o , so that it won't take the vaue randomly from the dataset.

x_train = x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)

#Lets train the dataset using the simple linear regression algorithm.
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Now will predict the values:
y_pred = regressor.predict(x_test)
    
# Lets do the comparision between the actual and the predicted data values:
comparision = pd.DataFrame({"Actual Value" : y_test ,"Predicted" : y_pred})

# Visualizing the training datasets:
plt.scatter(x_train, y_train, color='green') 
plt.plot(x_train, regressor.predict(x_train), color='black')
plt.title('House Price vs House Area(Training set)')
plt.xlabel('Area per square feet')
plt.ylabel('Price')
plt.show()

# Visualizing the testing datasets
plt.scatter(x_test, y_test, color='green') 
plt.plot(x_train, regressor.predict(x_train), color='black')
plt.title('House Price vs House Area(Test set)')
plt.xlabel('Area per square feet')
plt.ylabel('Price')
plt.show()

# Lets predict the price for the some specified areas:
y_560 = regressor.predict([[560]]) 
y_200 = regressor.predict([[200]])
print(y_560)
print(y_890)


bias = regressor.score(x_train, y_train)
variance = regressor.score(x_test, y_test) 
train_mse = mean_squared_error(y_train, regressor.predict(x_train)) 
test_mse = mean_squared_error(y_test, y_pred) 

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# Save the trained model to disk
filename = 'house price_prdiction.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as house price_prdiction.pkl")

import os 
print(os.getcwd())
