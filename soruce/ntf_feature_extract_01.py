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
	df.loc[:,[category]] = df[category]/df['__NOTIFICATION__total_event']
	return df[category].values

def scale(data):
	data_scaled = StandardScaler().fit_transform(data.values.reshape(-1,1).astype('float64'))
	return data_scaled.reshape(-1)


def transform(df):
	data = {
	'_label' : df['__NOTIFICATION__station_final_status'].values,
	'_total': scale(df['__NOTIFICATION__total_event']),
	'_rate': df['__NOTIFICATION__rate'].values,
	'_entropy_hour': df['__NOTIFICATION__entropy_for_hours_of_day'].values,
	'cnt_event_type': scale(df['__NOTIFICATION__event_type_sent']),
	'ntfid_arp': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_ARP_SPOOFING_FOUND'),
	'ntfid_box_conn': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_BOX_CONNECTED'),
	'ntfid_box_discon': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_BOX_DISCONNECTED'),
	'ntfid_conn_at_home': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_CONNECTED_AT_HOME'),
	'ntfid_defaut_pass': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_DEFAULT_PASSWORD'),
	# 'ntfid_identify': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_IDENTIFY'),
	# 'ntfid_inapp': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_INAPPROPRIATE_APP'),
	# 'ntfid_net_attack': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_NETWORK_ATTACK'),
	# 'ntfid_net_control': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_NETWORK_CONTROL'),
	'ntfid_net_controlblk': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_NETWORK_CONTROL_BLOCK'),
	'ntfid_new_device': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_NEW_DEVICE_JOINED'),
	'ntfid_device_router': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_NEW_DEVICE_JOINED_ROUTER'),
	# 'ntfid_new_client_ver': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_NEW_IN_CLIENT_VERSION'),
	# 'ntfid_ransomware': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_RANSOMWARE'),
	# 'ntfid_web_filter': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_WEBSITES_FILTERING'),
	# 'ntfid_web_filter_blk': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_WEBSITES_FILTERING_BLOCK'),
	# 'ntfid_web_filter_deny': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_WEBSITES_FILTERING_DENIED'),
	'ntfid_web_threat': ratio(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_WEB_THREATS')
	}
	return data

def main():
	print('\n')
	print("------------Load Data------------\n")
	trainDF = pd.read_csv("../data/training/Train_extracted_notification_feature_2018-01-17_21-20-30.csv")
	testDF = pd.read_csv("../data/testing/Test_extracted_notification_feature_2018-01-17_21-20-30.csv")

	trainID = hashEventCase(trainDF)
	testID = hashEventCase(testDF)
	
	trainData = pd.DataFrame(transform(trainDF), index = trainID)
	testData = pd.DataFrame(transform(testDF), index = testID)
	# print trainData.shape
	print trainData.describe()

	trainData.to_csv('../data/feature/ntf_training_v2.csv')
	testData.to_csv('../data/feature/ntf_testing_v2.csv')

if __name__ == "__main__":
    main()