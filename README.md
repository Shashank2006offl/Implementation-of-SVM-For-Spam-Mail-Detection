# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1. Start Program

STEP 2. Import the necessary packages.

STEP 3. Read the given csv file and display the few contents of the data.

STEP 4. Assign the features for x and y respectively.

STEP 5. Split the x and y sets into train and test sets.

STEP 6. Convert the Alphabetical data to numeric using CountVectorizer.

STEP 7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

STEP 8. Find the accuracy of the model.

STEP 9.Stop


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SHASHANK R
RegisterNumber:  212223230205
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data = pd.read_csv("spam.csv",encoding='Windows-1252')
data.head()
data.shape
x=data['v2'].values
y=data['v1'].values
x.shape
y.shape
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.35, random_state=42)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer()
x_train = cv.fit_transform(x_train) 
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
acc=accuracy_score(y_test,y_pred)
acc
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
y_pred

![image](https://github.com/user-attachments/assets/f56989ef-20ae-4fe9-ab58-c5fbed4d71bf)

Accuracy

![image](https://github.com/user-attachments/assets/9356f432-894f-4635-9fd6-c427492dab79)

Confusion matrix

![image](https://github.com/user-attachments/assets/b23efc68-39d3-4e41-a7d4-10d28f22dc87)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
