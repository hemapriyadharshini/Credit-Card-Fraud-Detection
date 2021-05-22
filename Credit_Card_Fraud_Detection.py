# IMPORTING PACKAGES
import pandas as pd # for data processing
import numpy as np # for arrays
import matplotlib.pyplot as plt # Data visualization
from termcolor import colored as cl # text customization
import itertools # for complex iterations

from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.model_selection import train_test_split # Split training data
from sklearn.tree import DecisionTreeClassifier # Decision tree model
from sklearn.neighbors import KNeighborsClassifier # KNN model
from sklearn.linear_model import LogisticRegression # Logistic regression model
from sklearn.svm import SVC # SVM model
from sklearn.ensemble import RandomForestClassifier # Random forest model
from xgboost import XGBClassifier # Gradient Boosting model

from sklearn.metrics import confusion_matrix # for Model evaluation
from sklearn.metrics import accuracy_score # for model validation
from sklearn.metrics import f1_score # for model validation

# Import data
df = pd.read_csv('creditcard.csv') #Read data
df.drop('Time', axis = 1, inplace = True) #Drop field - 'Time'

#Pre-processing and Exploratory Analysis
Txns = len(df)  #Get the no of rows
nonfraud_count = len(df[df.Class == 0]) #Count total no of non fraud txns
fraud_count = len(df[df.Class == 1])#Count total no of fraud txns
fraud_percentage = round(fraud_count/nonfraud_count*100, 2) # calc fraud %

#Fraud Txn stats
print("************* Project: Credit Card Fraud Detection *************")
print('Txn Count Summary:')
print('Total number of Txns:',Txns)
print('Number of Non-fraud Txns:',nonfraud_count)
print('Number of Confirmed fraud Txns:',fraud_count)
print('Percentage of Confirmed fraud Txns:',fraud_percentage,"%")

nonfraud_Txns = df[df.Class == 0] 
fraud_Txns = df[df.Class == 1]

print('\nTxn Amount Summary:') #Fraud Txn amount summary
print("\nNon-Fraud txn amount:\n",nonfraud_Txns.Amount.describe()) #Describe nonfraud txns amount field
print("\nFraud txn amount:\n",fraud_Txns.Amount.describe()) #Describe fraud txns amount field

#Normalize amount field
sc = StandardScaler() #Since the amount values are highly non-scalable i.e., highly vary in values, standardization is necessary to obtain a stable model with great accuracy.   
amount = df['Amount'].values 

df['Amount'] = sc.fit_transform(amount.reshape(-1, 1)) #All amount values are transformed and standardized using this function. Output values will be between 0 and 1

#Train Model - split train and test data

x = df.drop('Class', axis = 1).values #Remove class column
y = df['Class'].values #Keep class column

#x_train = Training data set; x_test= test data set; y_train= set of labels to all the data in x_train; y_test=set of labels to all the data in y_train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) #Assign 20% of dataset for testing; Random state - controls the shuffling before the data split. Here 0 means no shuffling

#Model Development

# 1. Decision Tree
tree_model = DecisionTreeClassifier(max_depth = 4, criterion = 'entropy') #max_depth = The maximum depth of the tree; criterion = function to measure the quality of a split;
tree_model.fit(x_train, y_train) #Fit model
tree_yhat = tree_model.predict(x_test) #Predict model

# 2. K-Nearest Neighbors
n = 5
knn = KNeighborsClassifier(n_neighbors = n) #Number of neighbors to use by default for kneighbors queries.
knn.fit(x_train, y_train) #Fit Model
knn_yhat = knn.predict(x_test) #Model prediction

# 3. Logistic Regression
lr = LogisticRegression() #Execute Logistic Regression model
lr.fit(x_train, y_train) #Fit Model
lr_yhat = lr.predict(x_test) #Model prediction

# 4. SVM 
svm = SVC() #Execute Support Vector Machine Model
svm.fit(x_train, y_train) #Fit Model
svm_yhat = svm.predict(x_test) #Model Prediction

# 5. Random Forest Tree
rf = RandomForestClassifier(max_depth = 4) #Execute Random forest algorithm with a maximum depth = 4
rf.fit(x_train, y_train) #Fit Model
rf_yhat = rf.predict(x_test) #Model Prediction

# 6. XGBoost
xgb = XGBClassifier(max_depth = 4)#Execute Gradient Boosting algorithm with a maximum depth = 4
xgb.fit(x_train, y_train) #Fit Model
xgb_yhat = xgb.predict(x_test)#Model Prediction

#Evaluation
#1. Accuracy
print("\n1. Model Prediction Completed")
print("\n1. Accuracy Results:")
print('Logistic Regression:',round(accuracy_score(y_test, lr_yhat)*100,2),"%")
print('KNN:',round(accuracy_score(y_test, knn_yhat)*100,2),"%")
print('SVM:',round(accuracy_score(y_test, svm_yhat)*100,2),"%")
print('Decision Tree:',round(accuracy_score(y_test, tree_yhat)*100,2),"%")
print('Random Forest:',round(accuracy_score(y_test, rf_yhat)*100,2),"%")
print('XGBoost:',round(accuracy_score(y_test, xgb_yhat)*100,2),"%")

#2. F1 Score
print("\n2. F1 Score Results:")
print('Logistic Regression:',round(f1_score(y_test, lr_yhat)*100,2),"%")
print('KNN:',round(f1_score(y_test, knn_yhat)*100,2),"%")
print('SVM:',round(f1_score(y_test, svm_yhat)*100,2),"%")
print('Decision Tree:',round(f1_score(y_test, tree_yhat)*100,2),"%")
print('Random Forest:',round(f1_score(y_test, rf_yhat)*100,2),"%")
print('XGBoost:',round(f1_score(y_test, xgb_yhat)*100,2),"%")

# 3. Confusion Matrix

# defining the plot function

def plot_confusion_matrix(cm, classes, title, normalize = False, cmap = plt.cm.Blues):
    title = 'Confusion Matrix of {}'.format(title)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    s = [['TN','FP'], ['FN', 'TP']]
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(s[i][j])+" = "+str(cm[i][j]), #Text to display in the confusion matrix plot+alignment
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')

   
    plt.ylabel('Actual Label') #Y Label
    plt.xlabel('Predicted Label') #X Label
    plt.tight_layout()

# Compute confusion matrix for the models
lr_matrix = confusion_matrix(y_test, lr_yhat, labels = [0, 1]) # Logistic Regression
knn_matrix = confusion_matrix(y_test, knn_yhat, labels = [0, 1]) # K-Nearest Neighbors
svm_matrix = confusion_matrix(y_test, svm_yhat, labels = [0, 1]) # Support Vector Machine
tree_matrix = confusion_matrix(y_test, tree_yhat, labels = [0, 1]) # Decision Tree
rf_matrix = confusion_matrix(y_test, rf_yhat, labels = [0, 1]) # Random Forest Tree
xgb_matrix = confusion_matrix(y_test, xgb_yhat, labels = [0, 1]) # XGBoost

# Plot the confusion matrix
plt.figure(figsize=(7,7))

# 1. Decision tree

tree_cm_plot = plot_confusion_matrix(tree_matrix, 
                                classes = ['Not Fraud','Fraud'], 
                                normalize = False, title = 'Decision Tree')
plt.savefig('tree_cm_plot.png')
plt.show()

# 2. K-Nearest Neighbors

knn_cm_plot = plot_confusion_matrix(knn_matrix, 
                                classes = ['Not Fraud','Fraud'], 
                                normalize = False, title = 'KNN')
plt.savefig('knn_cm_plot.png')
plt.show()

# 3. Logistic regression

lr_cm_plot = plot_confusion_matrix(lr_matrix, 
                                classes = ['Not Fraud','Fraud'], 
                                normalize = False, title = 'Logistic Regression')
plt.savefig('lr_cm_plot.png')
plt.show()

# 4. Support Vector Machine

svm_cm_plot = plot_confusion_matrix(svm_matrix, 
                                classes = ['Not Fraud','Fraud'], 
                                normalize = False, title = 'SVM')
plt.savefig('svm_cm_plot.png')
plt.show()

# 5. Random forest tree

rf_cm_plot = plot_confusion_matrix(rf_matrix, 
                               classes = ['Not Fraud','Fraud'], 
                                normalize = False, title = 'Random Forest Tree')
plt.savefig('rf_cm_plot.png')
plt.show()

# 6. XGBoost

xgb_cm_plot = plot_confusion_matrix(xgb_matrix, 
                                classes = ['Not Fraud','Fraud'], 
                                normalize = False, title = 'XGBoost')
plt.savefig('xgb_cm_plot.png')
plt.show()
