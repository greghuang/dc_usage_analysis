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
	df.loc[:,[category]] = df[category]/df['__SECURITYWARNING__total_event']
	return df[category].values

def get_ratio_network_attack(df, category):
	df.loc[df.__SECURITYWARNING__event_type_network_attack_prevented > 0,[category]] = df[category]/df['__SECURITYWARNING__event_type_network_attack_prevented']
	return df[category]

def scale(data):
	data_scaled = StandardScaler().fit_transform(data.values.reshape(-1,1).astype('float64'))
	return data_scaled.reshape(-1)

def transform(df):
	net_web_attack = get_ratio_network_attack(df, '__SECURITYWARNING__payload_rule_category_Web Attack')
	net_virus = get_ratio_network_attack(df, '__SECURITYWARNING__payload_rule_category_Virus/Worm')
	net_scan = get_ratio_network_attack(df, '__SECURITYWARNING__payload_rule_category_Scan')
	net_misc = get_ratio_network_attack(df,'__SECURITYWARNING__payload_rule_category_Misc')
	net_ddos = get_ratio_network_attack(df, '__SECURITYWARNING__payload_rule_category_DoS/DDoS')
	net_buf_overflow = get_ratio_network_attack(df, '__SECURITYWARNING__payload_rule_category_Buffer Overflow')
	net_access = get_ratio_network_attack(df, '__SECURITYWARNING__payload_rule_category_Access Control')
	net_trojan = get_ratio_network_attack(df, '__SECURITYWARNING__payload_rule_category_Backdoor/Trojan')

	data = {
	'_label' : df['__SECURITYWARNING__station_final_status'].values,
	'_total': scale(df['__SECURITYWARNING__total_event']),
	'_rate': df['__SECURITYWARNING__rate'].values,
	'_entropy_hour': df['__SECURITYWARNING__entropy_for_hours_of_day'].values,
	'ra_network_attack': ratio(df, '__SECURITYWARNING__event_type_network_attack_prevented'),
	'ra_weak_password': ratio(df,'__SECURITYWARNING__event_type_weak_password_detected'),
	'ra_web_threat': ratio(df, '__SECURITYWARNING__event_type_web_threat_blocked'),
	'net_web_attack': net_web_attack.values,
	'net_virus': net_virus.values,
	'net_scan': net_scan.values,
	'net_misc': net_misc.values,
	'net_ddos': net_ddos.values,
	'net_buf_overflow': net_buf_overflow.values,
	'net_access': net_access.values,
	'net_trojan': net_trojan.values,
	'dcnt_catid': df['__SECURITYWARNING__distinct_payload_cat_id'].values,
	'dcnt_device': df['__SECURITYWARNING__distinct_payload_device_id'].values,
	'dcnt_profile': df['__SECURITYWARNING__distinct_payload_profile_id'].values,
	'dcnt_rule': df['__SECURITYWARNING__distinct_payload_rule_category'].values,
	'network_attack': df['__SECURITYWARNING__event_type_network_attack_prevented'].values,
	}
	return data;

def load(path):
	df = pd.read_csv(path)
	# filter out if the death time is too long
	df = df[df.__SECURITYWARNING__death_period < 680400]
	return df

def main():
	print('\n')
	print("------------Load Data------------\n")
	trainDF = load("../data/training/Train_extracted_security_warning_feature_2018-01-17_21-20-30.csv")
	testDF = load("../data/testing/Test_extracted_security_warning_feature_2018-01-17_21-20-30.csv")

	trainID = hashEventCase(trainDF)
	testID = hashEventCase(testDF)
	
	trainData = pd.DataFrame(transform(trainDF), index = trainID)
	testData = pd.DataFrame(transform(testDF), index = testID)
	# print trainData.shape
	print trainData.describe()

	trainData.to_csv('../data/feature/sw_training_v4.csv')
	testData.to_csv('../data/feature/sw_testing_v4.csv')

if __name__ == "__main__":
    main()