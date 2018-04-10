import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
	print "\n"
	df = pd.read_csv('../data/others/offline_time_diff_datas.csv')
	print df.describe()
	offtime1 = df['period']
	offtime2 = df['period'].apply(lambda x: np.log10(x/60) if x > 0 else x )

	print offtime2.describe()

	# sns.set_style("whitegrid")
	# df = sns.load_dataset("../data/others/offline_time_diff_datas.csv")
	# ax = sns.boxplot(x=df['period'])
	sns.set(color_codes=True)
	# sns.distplot(offtime1);
	sns.boxplot(data=offtime1)

	# Scatterplot
	# Recommended way
	# sns.lmplot(x='_rate', y='_entropy_hour', data=offtime, fit_reg=False, hue='_label')

	# Tweak using Matplotlib
	# plt.ylim(0, None)
	# plt.xlim(0, None)
	# plt.show()

	# Boxplot
	# Pre-format DataFrame
	# stats_df = df.drop(['_label', '_total', '_rate', 'cnt_event_type'], axis=1)
	# sns.boxplot(data=offtime)
	# plt.show()

	# # Set theme
	# sns.set_style('whitegrid')

	# # Violin plot
	# sns.violinplot(x='Type 1', y='_rate', data=df)
	plt.show()

if __name__ == '__main__':
	main()