#importing  the libraries
import pandas as pd
import numpy as np
#importing dataset

dataset =pd.read_csv("breast_cancer.csv")
X = dataset.iloc[:,1:-1].values
y= dataset.iloc[:,-1].values

#splitting Dataset into training and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.2, random_state= 0)

# training our logistic regression model

from sklearn.linear_model import LogisticRegression

l_Classifier = LogisticRegression(random_state = 0)
l_Classifier.fit(X_train,y_train)




#prediting the test set result

y_pred = l_Classifier.predict(X_test)

#making the the confusion matrix

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
print(cm)

# computing the Accuracy with the k fold validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = l_Classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


