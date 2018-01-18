import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.externals import joblib

def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

train_df = pd.read_csv('../data/feature/sw_training_v2.csv')
test_df = pd.read_csv('../data/feature/sw_testing_v2.csv')

print train_df.shape
print test_df.shape
# print train_df.describe()
print("------------------------\n")

X_train = train_df.iloc[:, 2:]
X_test = test_df.iloc[:, 2:]

Y_train = train_df['_label']
Y_test = test_df['_label']

# X = train_df.drop('_label', axis=1)
# print X.head()

# Split data into train set and test set
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123, stratify=Y)

# Fitting the Transform API
scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
print X_train_scaled.mean(axis=0)
print X_train_scaled.std(axis=0)

X_test_scaled = scaler.transform(X_test)
print X_test_scaled.mean(axis=0)
print X_test_scaled.std(axis=0)

print("\n------------RF Regressor-----------")

pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

# print pipeline.get_params()

# hyperparameters = { 
# 	'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
# 	'randomforestregressor__max_depth': [None, 5, 3, 1]
# 	}

hyperparameters = { 
	'randomforestregressor__max_features' : ['auto'],
	'randomforestregressor__max_depth': [3]
	}


# Sklearn cross-validation with pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=3)
# Fit and tune model
clf.fit(X_train, Y_train)
print clf.best_params_
print clf.refit

# Predict a new set of data
Y_pred = clf.predict(X_test)

print r2_score(Y_test, Y_pred)
print mean_squared_error(Y_test, Y_pred)

print("\n-----------RF Classifier-----------")

trained_model = random_forest_classifier(X_train, Y_train)
predictions = trained_model.predict(X_test)

# Train and Test Accuracy
print "Train Accuracy :: ", accuracy_score(Y_train, trained_model.predict(X_train))
print "Test Accuracy  :: ", accuracy_score(Y_test, predictions)
print " Confusion matrix ", confusion_matrix(Y_test, predictions)

fpr_rf, tpr_rf, _ = roc_curve(Y_test, predictions)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='SW_RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()



print("\n-----------Dump model-----------")
# joblib.dump(clf, '../Model/sw_rf_regressor_v1.pkl')

