import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

def main():
	train_df = pd.read_csv('../data/feature/sw_training_v2.csv', index_col=0)
	x_train = train_df.loc[train_df._label == 1,:]
	x_test = train_df.loc[train_df._label == 0,:]
	print train_df.size
	print x_train.size
	print x_test.size

	clf = svm.OneClassSVM(nu=0.1, kernel="sigmoid", gamma=0.1)
	clf.fit(x_test)
	y_train_pred = clf.predict(x_train)
	y_test_pred = clf.predict(x_test)
	n_error_train = y_train_pred[y_train_pred == -1].size
	n_error_test = y_test_pred[y_test_pred == -1].size

	print "training error::",n_error_train
	print "testing error::", n_error_test


if __name__ == '__main__':
	print("\n")
	main()

