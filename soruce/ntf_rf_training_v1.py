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

def prepare(df):
	x = df.drop('_label', axis=1)
	y = df.loc[:,'_label']
	return x, y

def training(x, y):
	print("\n-----------RF Classifier-----------")
	model = RandomForestClassifier()
	model.fit(x, y)
	return model

def predict(trained_model, x):
	predictions = trained_model.predict(x)
	return predictions

def validate(y_pred, y_test):
	print "Accuracy :: ", accuracy_score(y_test, y_pred)
	print " Confusion matrix ", confusion_matrix(y_test, y_pred)

	fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred)
	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr_rf, tpr_rf, label='RF_Test')
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.show()

def main():
	trainDF = pd.read_csv('../data/feature/ntf_training_v1.csv', index_col=0)
	testDF = pd.read_csv('../data/feature/ntf_testing_v1.csv', index_col=0)
	# visualize(trainDF)
	# print trainDF.describe()

	x_train, y_train = prepare(trainDF)
	x_test, y_test = prepare(testDF)
	
	tmodel = training(x_train, y_train)
	y_pred = predict(tmodel, x_test)
	validate(y_pred, y_test)


if __name__ == "__main__":
	main()