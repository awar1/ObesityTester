import numpy as np
import pandas as pd
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# scale 0 - 3 means 0 - never, 1 - sometimes, 2 - often, 3 - always, except when question is about number of something
# scale 0,1 means 0 - no, 1 - yes, except when description after "-" says otherwise

#gender - (0, 1 - woman, man), age - (in years), height - (in meters),
#family_history_with_overweight - (0, 1), 
#FAVC - frequent consumption of high caloric food (0, 1),
#FCVC - frequency of eating vegetables (0 - 3), NCP - number of main meals (0-3),
#CAEC- frequency of eating between meals (0 - 3), Smoke - (0, 1),
#CH2O - liters of water during the day (0 - 3), SCC - counting calories (0, 1),
#FAF - physical activity (0 - 3), TUE - time spent using electronic devices (0-3),
#CALC - consumption of alcohol (0 - 3),
#MTRANS - means of transportation (0 - car, 1 - motorcycle, 2 - bike, 3 - public tranportation,
#4 - pedestrian)

# reading the data from csv file
obesity = pd.read_csv("obesity3.csv", usecols=[i for i in range(0,17)], header=0)

#deleting outliers
numericl_col = ['Age','Height','Weight']
for i in range(3):
    for x in [numericl_col[i]]:
        q75,q25 = np.percentile(obesity.loc[:,x],[75,25])
        intr_qr = q75-q25
    
        max = q75+(1.5*intr_qr)
        min = q25-(1.5*intr_qr)
    
        obesity.loc[obesity[x] < min,x] = min
        obesity.loc[obesity[x] > max,x] = max

# dividing columns between characteristics and result
charasteristics_columns = ['Gender','Age','Height','Weight','family_history_with_overweight','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']
X = obesity[charasteristics_columns]
Y = obesity.Obesity

#standarization of data
ss = StandardScaler()
X = ss.fit_transform(X)


def decision_tree_prediction(X_train, X_test, y_train, y_test):
    """predicts body mass index according to decision tree algorithm"""
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def random_forest_prediction(X_train, X_test, y_train, y_test):
    """predicts body mass index according to random forest algorithm"""
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rf=rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred = np.around(y_pred,0)
    return y_pred

def KNN_prediction(X_train, X_test, y_train, y_test):
    """predicts body mass index according to K Nearest Neighbours algorithm"""
    clf = KNeighborsClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def svm_prediction(X_train, X_test, y_train, y_test):
    """predicts body mass index according to Support Vector Machine algorithm"""
    cls = svm.SVC(kernel = "linear")
    cls = cls.fit(X_train,y_train)
    y_pred = cls.predict(X_test)
    return y_pred
    

def hybrid_prediction(X_train, X_test, y_train, y_test):
    """predicts body mass index based on arithmetic average of 4 upper functions"""
    y_predDT = decision_tree_prediction(X_train, X_test, y_train, y_test)
    y_predKNN = KNN_prediction(X_train, X_test, y_train, y_test)
    y_predRF = random_forest_prediction(X_train, X_test, y_train, y_test)
    y_predSVM = svm_prediction(X_train, X_test, y_train, y_test)
    y_pred = (y_predDT+y_predKNN+y_predRF+y_predSVM)/4
    y_pred = np.around(y_pred,0)
    return y_pred

# dividing data into 5 sets for crossvalidation
kf = KFold(n_splits=5, random_state=None)
scores = [[] for _ in range(5)]
functions = [decision_tree_prediction, random_forest_prediction, KNN_prediction, svm_prediction, hybrid_prediction]

kfold = kf.split(X)
# Calculating precision of each function
for k, (train, test) in enumerate(kfold):
    X_train , X_test =X[train],X[test]
    y_train , y_test = Y[train] , Y[test]
    for i in range(0,5):
        y_pred = functions[i](X_train,X_test,y_train,y_test)
        score = metrics.accuracy_score(y_test, y_pred)
        scores[i].append(score)

algorytmy = ["Decision Tree", "Random Forest", "KNN", "SVM", "hybrid"]
for i in range(0,5):
    print("Algorithm {0} worked with precision {1:.2f}{2} and standard deviation equal {3:.2f}".format(algorytmy[i],np.mean(scores[i])*100,'%', np.std(scores[i])))

