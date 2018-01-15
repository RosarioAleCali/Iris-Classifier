import sys
import os
import time
import pandas as pd
import mglearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Grabbing the Dataset and splitting up the data for training and testing
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

def describeData():
    print("Keys of iris_dataset: {}".format(iris_dataset.keys()))
    print(iris_dataset['DESCR'][:193] + '\n...')
    print("Target names: {}".format(iris_dataset['target_names']))
    print("Feature names: {}".format(iris_dataset['feature_names']))
    print("Type of data: {}".format(type(iris_dataset['data'])))
    print("Shape of data: {}".format(iris_dataset['data'].shape))
    print("First five rows of data:\n{}".format(iris_dataset['data'][:5]))
    print("Type of target: {}".format(type(iris_dataset['target'])))
    print("Shape of target: {}".format(iris_dataset['target'].shape))
    print("Target:\n{}".format(iris_dataset['target']))
    print("X_train shape: {}".format(X_train.shape))
    print("y_train shape: {}".format(y_train.shape))
    print("X_test shape: {}".format(X_test.shape))
    print("y_test shape: {}".format(y_test.shape))

    raw_input("\nPress Enter to continue...")

def plotData():
    # create dataframe from data in X_train
    # label the columns using the strings in iris_dataset.feature_names
    iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
    # create a scatter matrix from the dataframe, color by y_train
    ts = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
    plt.show()

def getDouble(message):
    while True:
        try:
            return float(input(message))
            break
        except:
            print("That's not a valid input!")

def getValues():
    sepalLenght = getDouble("Sepal length (cm): ")
    sepalWidth = getDouble("Sepal width (cm): ")
    petalLenght = getDouble("Petal length (cm): ")
    petalWidth = getDouble("Petal width (cm): ")

    return np.array([[sepalLenght, sepalWidth, petalLenght, petalWidth]])

def menu():
    menuOption = 0

    print("Please choose one of the following options")
    print("==========================================")
    print("1. Train the Model")
    print("2. Make a Prediction")
    print("3. Evaluate the Model")
    print("4. Plot the Dataset")
    print("5. Describe the Dataset")
    print("6. Exit")
    
    while True:
        try:
            menuOption = int(input("> "))
            break
        except:
            print("That's not a valid option!")
    
    return menuOption

def main():
    menuOption = 0
    modelTrained = False
    global knn

    while menuOption != 6:
        os.system('clear')
        menuOption = menu()
        
        if menuOption == 1:
            os.system('clear')
            if modelTrained == False:
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(X_train, y_train)
                modelTrained = True
                print("Model Trained Successfully!")
        elif menuOption == 2:
            os.system('clear')
            if modelTrained == True:
                X_new = getValues()
                prediction = knn.predict(X_new)
                print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))
            else:
                print("You need to train the model first")
        elif menuOption == 3:
            os.system('clear')
            if modelTrained == True:
                print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
            else:
                print("You need to train the model first")
        elif menuOption == 4:
            os.system('clear')
            plotData()
        elif menuOption == 5:
            os.system('clear')
            describeData()
        elif menuOption == 6:
            os.system('clear')
            print("Goodbye!")
        else:
            print("\nInvalid menu selection!")

        time.sleep(1)
  
if __name__== "__main__":
    main()