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
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib
from sklearn import svm
import seaborn as sns

def visualize(df):
	# Scatterplot
	# Recommended way
	sns.lmplot(x='_rate', y='_entropy_hour', data=df, fit_reg=False, hue='_label')

	# Tweak using Matplotlib
	# plt.ylim(0, None)
	# plt.xlim(0, None)
	# plt.show()

	# Boxplot
	# Pre-format DataFrame
	# stats_df = df.drop(['_label', '_total', '_rate', 'cnt_event_type'], axis=1)
	# sns.boxplot(data=stats_df)
	# plt.show()

	# # Set theme
	# sns.set_style('whitegrid')

	# # Violin plot
	# sns.violinplot(x='Type 1', y='_rate', data=df)
	# plt.show()

def prepare(df, threshold=1.0):
	# show the label info of dataframe
	print "label count(old):: ", df.groupby('_label').size()
	ndf = df[df._rate <= threshold]
	print "label count(new):: ", ndf.groupby('_label').size()
	x = ndf.drop('_label', axis=1)
	# Drop rate from features
	x = x.drop('_rate', axis=1)
	
	y = ndf.loc[:,'_label']
	
	print "data shape::", x.shape
	print x.columns
	return x, y

def showFeatureImportance(x, clf):
	importances = clf.feature_importances_
	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print("Feature ranking:")

	for f in range(x.shape[1]):
		print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

def fit(x, y):
	return fitSVM(x, y)
	# return fitRF(x, y)

def fitSVM(x, y):
	clf = svm.NuSVC()
	clf.fit(x, y)
	# print clf
	return clf

def fitRF(x, y):
	print("\n-----------Training RF Classifier-----------")
	# pipeline = make_pipeline(RandomForestClassifier(n_estimators=50, random_state=0))
	
	# hyperparameters = { 
		# 'randomforestclassifier__max_features' : ['auto', 'log2', 0.8],
		# 'randomforestclassifier__max_depth': [None, 5, 3, 1],
		# 'randomforestclassifier__n_estimators': [10, 50, 100, 500],
	# }
	# clf = GridSearchCV(pipeline, hyperparameters, cv=3)
	# print clf.best_params_
	# print clf.refit

	clf = RandomForestClassifier(n_estimators=50, max_features='auto', random_state=0)
	clf.fit(x, y)
	
	showFeatureImportance(x, clf)
	return clf


def validate(plt_label, y_train_pred, y_train, y_test_pred, y_test):
	print "Accuracy:: ", accuracy_score(y_test, y_test_pred)
	print " Confusion matrix:: \n", confusion_matrix(y_test, y_test_pred)

	# fpr_rf_train, tpr_rf_train, _ = roc_curve(y_train, y_train_pred, pos_label = 1)
	fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_pred, pos_label = 1)
	print plt_label, "AUC:: ", auc(fpr_rf, tpr_rf)
	# plt.plot(fpr_rf_train, tpr_rf_train, label='NTF_Train')
	plt.plot(fpr_rf, tpr_rf, label=plt_label)


def pipeline(plt_label, trainDF, testDF):
	isTrainAll = True
	if isTrainAll:
		allDF = pd.concat([trainDF, testDF])
		print "all::", allDF.shape
		X, Y = prepare(allDF, 0.95)
		# Split data into train set and test set
		x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123, stratify=Y)		
	else:
		x_train, y_train = prepare(trainDF, 0.95)
		x_test, y_test = prepare(testDF, 0.95)

	# print "Test data:: ", x_test.shape
	tmodel = fit(x_train, y_train)
	y_train_pred = tmodel.predict(x_train)
	y_test_pred = tmodel.predict(x_test)
	validate(plt_label, y_train_pred, y_train ,y_test_pred, y_test)

def main():
	print("\n")
	train_stat = pd.read_csv('../data/feature/stat_training_v2.csv', index_col=0)
	test_stat = pd.read_csv('../data/feature/stat_testing_v2.csv', index_col=0)

	train_ntf = pd.read_csv('../data/feature/ntf_training_v3.csv', index_col=0)
	test_ntf = pd.read_csv('../data/feature/ntf_testing_v3.csv', index_col=0)

	train_pc = pd.read_csv('../data/feature/pc_training_v2.csv', index_col=0)
	test_pc = pd.read_csv('../data/feature/pc_testing_v2.csv', index_col=0)

	train_sw = pd.read_csv('../data/feature/sw_training_v4.csv', index_col=0)
	test_sw = pd.read_csv('../data/feature/sw_testing_v4.csv', index_col=0)

	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')	

	pipeline("STAT_TEST", train_stat, test_stat)
	pipeline("NTF_TEST", train_ntf, test_ntf)
	pipeline("ParCon_TEST", train_pc, test_pc)
	pipeline("SecWar_TEST", train_sw, test_sw)

	plt.legend(loc='best')
	plt.show()
	# visualize(trainDF)
	# print trainDF.describe()


if __name__ == "__main__":
	main()