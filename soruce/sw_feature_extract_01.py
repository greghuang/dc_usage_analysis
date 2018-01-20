import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib

print('\n')

def get_ratio_network_attack(dataframe, category):
	dataframe.loc[df.__SECURITYWARNING__event_type_network_attack_prevented > 0,[category]] = df[category]/df['__SECURITYWARNING__event_type_network_attack_prevented']
	return dataframe[category]


def main():
	# df = pd.read_csv("../data/training/Train_extracted_security_warning_feature_2018-01-17_14-13-13.csv")
	df_test = pd.read_csv("../data/testing/Test_extracted_security_warning_feature_2018-01-17_14-13-13.csv")
	df = df_test
	# df['hash'] = [hashlib.md5(val).hexdigest() for val in df['__EVENTCASE__']]
	# print(df.loc[:5, ['hash', '__SECURITYWARNING__total_event']])
	# print(df.loc[:,['__SECURITYWARNING__total_event']])

	# print(df.loc[10])

	 # Hash the eventcase as md5
	idx = [hashlib.md5(val).hexdigest() for val in df['__EVENTCASE__']]
	# cols = ['label', 'total_evt']
	# ndf = df.loc[:,['__SECURITYWARNING__station_final_status']]
	# ndf.index = idx
	# ndf.columns = cols
	# print(df['__SECURITYWARNING__station_final_status'].values)
	# ndf['label'] = df[['__SECURITYWARNING__station_final_status']].values

	network_att_ratio = df['__SECURITYWARNING__event_type_network_attack_prevented']/df['__SECURITYWARNING__total_event']
	weak_pass_ratio = df['__SECURITYWARNING__event_type_weak_password_detected']/df['__SECURITYWARNING__total_event']
	web_threat_ratio = df['__SECURITYWARNING__event_type_web_threat_blocked']/df['__SECURITYWARNING__total_event']

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
		'sw_total': df['__SECURITYWARNING__total_event'].values,
		'net_web_attack': net_web_attack.values,#df['__SECURITYWARNING__payload_rule_category_Web Attack'].values,
		'net_virus': net_virus.values, #df['__SECURITYWARNING__payload_rule_category_Virus/Worm'].values,
		'net_scan': net_scan.values, #df['__SECURITYWARNING__payload_rule_category_Scan'].values,
		'net_misc': net_misc.values, #df['__SECURITYWARNING__payload_rule_category_Misc'].values,
		'net_ddos': net_ddos.values, #df['__SECURITYWARNING__payload_rule_category_DoS/DDoS'].values,
		'net_buf_overflow': net_buf_overflow.values, #df['__SECURITYWARNING__payload_rule_category_Buffer Overflow'].values,
		'net_access': net_access.values, #df['__SECURITYWARNING__payload_rule_category_Access Control'].values,
		'net_trojan': net_trojan.values, #df['__SECURITYWARNING__payload_rule_category_Backdoor/Trojan'].values,
		'dcnt_catid': df['__SECURITYWARNING__distinct_payload_cat_id'].values,
		'dcnt_device': df['__SECURITYWARNING__distinct_payload_device_id'].values,
		'dcnt_profile': df['__SECURITYWARNING__distinct_payload_profile_id'].values,
		'dcnt_rule': df['__SECURITYWARNING__distinct_payload_rule_category'].values,
		'network_attack': df['__SECURITYWARNING__event_type_network_attack_prevented'].values,
		'sw_rate': df['__SECURITYWARNING__rate'].values,
		'ra_network_attack': network_att_ratio.values,
		'ra_weak_password': weak_pass_ratio.values,
		'ra_web_threat': web_threat_ratio.values
	}

	ndf = pd.DataFrame(data, index = idx)

	print ndf.shape
	print ndf.describe()

	# save the features to a file
	# ndf.to_csv('../data/feature/sw_testing_v2.csv')

if __name__ == "__main__"
	main()
