import json
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as md

from pylab import rcParams
from matplotlib import rc
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import preprocessing

# plot option
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8


def anomaly_label(path, filename):
    df = pd.read_csv(path)
    with open('/daintlab/data/anomaly_detection/NAB-master/labels/combined_labels.json') as f:
        label = json.load(f)
    df['anomaly_label'] = df.timestamp.isin(label[filename]).astype('int')
    # df['value'] = df[df['anomaly_label'] == 0]['value'].fillna(method='ffill')
    
    return df


def convert_timestamp_to_epoch(data_frame):
    data_frame['timestamp'] = pd.to_datetime(data_frame['timestamp'])
    data_frame['time_epoch'] = data_frame['timestamp'].astype(np.int64)
    return data_frame


def split_dataset(path, filename, test=False):
    dataset = anomaly_label(path, filename)
    dataset = convert_timestamp_to_epoch(dataset)
    train_df = dataset[:int(len(dataset) * 0.8)]
    # valid_df = dataset[int(len(dataset) * 0.8): int(len(dataset) * 0.8 + len(dataset) * 0.1)]
    valid_df = dataset[int(len(dataset) * 0.8): ]
    
    test_df = dataset[int(len(dataset) * 0.8 + len(dataset) * 0.1): ]
    

    dataset.to_csv(path[:-4] + '_preprocessed.csv', index=False)
    if not test:
        dataset.to_csv(path[:-4] + '_preprocessed.csv', index=False)
        train_df.to_csv(path[:-4] + '_train.csv', index=False)
        valid_df.to_csv(path[:-4] + '_valid.csv', index=False)
    # test_df.to_csv(path[:-4]+'_test.csv', index=False)


if __name__ == "__main__":
    # tr_filename = 'realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv'
    # tr_path = "/daintlab/data/anomaly_detection/NAB-master/data/realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv"
    # tst_filename = 'realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv'
    # test_path = "/daintlab/data/anomaly_detection/NAB-master/data/realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv"
    tr_path = "/daintlab/data/anomaly_detection/NAB-master/data/realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv"
    tr_filename = "realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv"

    tst_path = "/daintlab/data/anomaly_detection/NAB-master/data/realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv"
    tst_filename = "realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv"
    
    # train and valid
    split_dataset(tr_path, tr_filename)
    
    # test
    split_dataset(tst_path, tst_filename, test=True)
    tr_ds = pd.read_csv("/daintlab/data/anomaly_detection/NAB-master/data/realAWSCloudwatch/rds_cpu_utilization_e47b3b_preprocessed.csv")
    val_ds = pd.read_csv("/daintlab/data/anomaly_detection/NAB-master/data/realAWSCloudwatch/rds_cpu_utilization_e47b3b_valid.csv")

    print(tr_ds)
