import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib

def hashEventCase(df):
	 # Hash the eventcase as md5
	 idx = [hashlib.md5(val).hexdigest() for val in df['__EVENTCASE__']]
	 return idx

def get_ratio_notification_id(df, category):
	df.loc[:,[category]] = df[category]/df['__NOTIFICATION__total_event']
	return df[category]


def transform(df):
	ntfid_arp = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_ARP_SPOOFING_FOUND')
	ntfid_box_conn = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_BOX_CONNECTED')
	ntfid_box_discon = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_BOX_DISCONNECTED')
	ntfid_conn_at_home = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_CONNECTED_AT_HOME')
	ntfid_def_pass = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_DEFAULT_PASSWORD')
	ntfid_identify = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_IDENTIFY')
	ntfid_inapp = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_INAPPROPRIATE_APP')
	ntfid_net_attack = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_NETWORK_ATTACK')
	ntfid_net_control = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_NETWORK_CONTROL')
	ntfid_net_controlblk = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_NETWORK_CONTROL_BLOCK')
	ntfid_new_device = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_NEW_DEVICE_JOINED')
	ntfid_device_router = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_NEW_DEVICE_JOINED_ROUTER')
	ntfid_new_client_ver = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_NEW_IN_CLIENT_VERSION')
	ntfid_ransomware = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_RANSOMWARE')
	ntfid_web_filter = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_WEBSITES_FILTERING')
	ntfid_web_filter_blk = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_WEBSITES_FILTERING_BLOCK')
	ntfid_web_filter_deny = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_WEBSITES_FILTERING_DENIED')
	ntfid_web_threat = get_ratio_notification_id(df, '__NOTIFICATION__payload_notification_id_PUSH_NOTIFICATION_WEB_THREATS')

	data = {
	'_label' : df['__NOTIFICATION__station_final_status'].values,
	'_total': df['__NOTIFICATION__total_event'].values,
	'_rate': df['__NOTIFICATION__rate'].values,
	'_entropy_hour': df['__NOTIFICATION__entropy_for_hours_of_day'].values,
	'cnt_event_type': df['__NOTIFICATION__event_type_sent'].values,
	'ntfid_arp': ntfid_arp.values,
	'ntfid_box_conn': ntfid_box_conn.values,
	'ntfid_box_discon': ntfid_box_discon.values,
	'ntfid_conn_at_home': ntfid_conn_at_home.values,
	'ntfid_def_pass': ntfid_def_pass.values,
	'ntfid_identify': ntfid_identify.values,
	'ntfid_inapp': ntfid_inapp.values,
	'ntfid_net_attack': ntfid_net_attack.values,
	'ntfid_net_control': ntfid_net_control.values,
	'ntfid_net_controlblk': ntfid_net_controlblk.values,
	'ntfid_new_device': ntfid_new_device.values,
	'ntfid_device_router': ntfid_device_router.values,
	'ntfid_new_client_ver': ntfid_new_client_ver.values,
	'ntfid_ransomware': ntfid_ransomware.values,
	'ntfid_web_filter': ntfid_web_filter.values,
	'ntfid_web_filter_blk': ntfid_web_filter_blk.values,
	'ntfid_web_filter_deny': ntfid_web_filter_deny.values,
	'ntfid_web_threat': ntfid_web_threat.values
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

	trainData.to_csv('../data/feature/ntf_training_v1.csv')
	testData.to_csv('../data/feature/ntf_testing_v1.csv')

if __name__ == "__main__":
    main()