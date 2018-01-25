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
	df.loc[:,[category]] = df[category]/df['__STATION__total_event']
	return df[category].values

def scale(data):
	data_scaled = StandardScaler().fit_transform(data.values.reshape(-1,1).astype('float64'))
	return data_scaled.reshape(-1)

def transform(df):
	data = {
	'_label' : df['__STATION__station_final_status'].values,
	'_total': scale(df['__STATION__total_event']),
	'_rate': df['__STATION__rate'].values,
	'_entropy_hour': df['__STATION__entropy_for_hours_of_day'].values,
	'payload_malf_false': scale(df['__STATION__payload_malfunctioning_false']),
	'payload_malf_true': scale(df['__STATION__payload_malfunctioning_true']),
	# 'payload_au_graceful': scale(df['__STATION__payload_offline_trigger_AU_GRACEFUL_SHUTDOWN']),
	# 'payload_au_graceful': df['__STATION__payload_offline_trigger_AU_GRACEFUL_SHUTDOWN'].values,
	'payload_lastwill': df['__STATION__payload_offline_trigger_MQTT_LASTWILL'].values,
	'ratio_offline': ratio(df, '__STATION__payload_status_OFFLINE'),
	'ratio_online': ratio(df, '__STATION__payload_status_ONLINE'),
	'condition_updated': df['__STATION__event_type_condition_updated'].values,
	'cnt_firmware_ver': df['__STATION__distinct_payload_firmware_version'].values,
	'cnt_payload_ip': df['__STATION__distinct_payload_ip'].values,
	'cnt_profile_id': df['__STATION__distinct_payload_profile_id'].values
	}
	return data;

def load(path):
	df = pd.read_csv(path)
	# filter out if the death time is too long
	df = df[df.__STATION__death_period < 680400]
	return df

def main():
	print('\n')
	print("------------Load Data------------\n")
	trainDF = load("../data/training/Train_extracted_station_feature_2018-01-17_21-20-30.csv")
	testDF = load("../data/testing/Test_extracted_station_feature_2018-01-17_21-20-30.csv")

	trainID = hashEventCase(trainDF)
	testID = hashEventCase(testDF)
	
	trainData = pd.DataFrame(transform(trainDF), index = trainID)
	testData = pd.DataFrame(transform(testDF), index = testID)
	print "Train::", trainData.shape
	print "Test::", testData.shape
	
	trainData.to_csv('../data/feature/stat_training_v2.csv')
	testData.to_csv('../data/feature/stat_testing_v2.csv')

if __name__ == "__main__":
    main()