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
import seaborn as sns

def prepare(df, threshold=1.0):	
	ndf = df[df._rate < threshold]	
	x = ndf.drop('_label', axis=1)
	y = ndf.loc[:,'_label']
	print "new set::", x.shape
	return x, y

def training(x, y):
	print("\n-----------Training RF Classifier-----------")
	# pipeline = make_pipeline(RandomForestClassifier(n_estimators=50, random_state=0))
	
	# hyperparameters = { 
		# 'randomforestclassifier__max_features' : ['auto', 'log2', 0.8],
		# 'randomforestclassifier__max_depth': [None, 5, 3, 1],
		# 'randomforestclassifier__n_estimators': [10, 50, 100, 500],
	# }
	clf = RandomForestClassifier(n_estimators=50, max_features='auto', random_state=0)
	# clf = GridSearchCV(pipeline, hyperparameters, cv=3)
	# print clf.best_params_
	# print clf.refit
	
	clf.fit(x, y)
	return clf

def validate(y_train_pred, y_train, y_test_pred, y_test):
	print "Accuracy :: ", accuracy_score(y_test, y_test_pred)
	print " Confusion matrix ::\n", confusion_matrix(y_test, y_test_pred)

	fpr_rf_train, tpr_rf_train, _ = roc_curve(y_train, y_train_pred)
	fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_pred)
	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr_rf_train, tpr_rf_train, label='PC_Train')
	plt.plot(fpr_rf, tpr_rf, label='PC_Test')
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.show()

def main():
	print "\n"
	trainDF = pd.read_csv('../data/feature/pc_training_v1.csv', index_col=0)
	testDF = pd.read_csv('../data/feature/pc_testing_v1.csv', index_col=0)
	allDF = pd.concat([trainDF, testDF])
	print "all::", allDF.shape
	X, Y = prepare(allDF, 0.9)

	# Split data into train set and test set
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=123, stratify=Y)
	print x_test.shape
	
	tmodel = training(x_train, y_train)
	y_train_pred = tmodel.predict(x_train)
	y_test_pred = tmodel.predict(x_test)
	validate(y_train_pred, y_train ,y_test_pred, y_test)


if __name__ == "__main__":
	main()