import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from sklearn.preprocessing import StandardScaler

def hashEventCase(df):
	 # Hash the eventcase as md5
	 idx = [hashlib.md5(val).hexdigest() for val in df['__EVENTCASE__']]
	 return idx

def ratio(df, category):
	df.loc[:,[category]] = df[category]/df['__PARENTALCONTROL__total_event']
	return df[category].values

def scale(data):
	data_scaled = StandardScaler().fit_transform(data.values.reshape(-1,1).astype('float64'))
	return data_scaled.reshape(-1)
	

def transform(df):
	# ntfid_arp = get_ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_ARP_SPOOFING_FOUND')

	data = {
	'_label' : df['__PARENTALCONTROL__station_final_status'].values,
	'_total': scale(df['__PARENTALCONTROL__total_event']),
	'_rate': df['__PARENTALCONTROL__rate'].values,
	'_entropy_hour': df['__PARENTALCONTROL__entropy_for_hours_of_day'].values,
	'cnt_app': df['__PARENTALCONTROL__distinct_payload_app_id'].values,
	'cnt_device': df['__PARENTALCONTROL__distinct_payload_device_id'].values,
	'scaled_kid_back_home': scale(df['__PARENTALCONTROL__event_type_kids_back_to_home']),
	'scaled_inapp_detected': scale(df['__PARENTALCONTROL__event_type_inappropriate_app_detected']),
	'scaled_net_limit' : scale(df['__PARENTALCONTROL__event_type_network_time_limited']),
	'scaled_web_filtered': scale(df['__PARENTALCONTROL__event_type_web_site_filtered']),
	# 'scaled_cat_0': scale(df['__PARENTALCONTROL__payload_cat_id_0']),
	# 'scaled_cat_1': scale(df['__PARENTALCONTROL__payload_cat_id_1']),
	'ratio_web_filtered': ratio(df, '__PARENTALCONTROL__event_type_web_site_filtered'),
	'ratio_net_limit': ratio(df, '__PARENTALCONTROL__event_type_network_time_limited'),
	'ratio_inapp_detected': ratio(df, '__PARENTALCONTROL__event_type_inappropriate_app_detected'),
	'ratio_kid_back_home': ratio(df, '__PARENTALCONTROL__event_type_kids_back_to_home'),
	# 'scaled_action_block': scale(df['__PARENTALCONTROL__payload_action_block']),
	# 'scaled_action_accept': scale(df['__PARENTALCONTROL__payload_action_accept']),
	}
	return data

def load(path):
	df = pd.read_csv(path)
	# filter out if the death time is too long
	df = df[df.__PARENTALCONTROL__death_period < 680400]
	return df

def extract(input, output):
	df = load(input)
	idDf = hashEventCase(df)
	featureDf = pd.DataFrame(transform(df), index = idDf)
	print "shape::", featureDf.shape
	print "the size of 1::", len(featureDf[featureDf._label == 1].index)
	print "the size of 0::", len(featureDf[featureDf._label == 0].index)
	featureDf.to_csv(output)


def main():
	print('\n')
	print("------------Extract Training Data------------\n")
	extract("../data/training/Train_extracted_parental_control_feature_2018-01-17_21-20-30.csv", '../data/feature/pc_training_v2.csv')

	print("------------Extract Testing Data------------\n")
	extract("../data/testing/Test_extracted_parental_control_feature_2018-01-17_21-20-30.csv", '../data/feature/pc_testing_v2.csv')


if __name__ == '__main__':
	main()