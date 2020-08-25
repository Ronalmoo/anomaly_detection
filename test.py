import os
import json
import wandb
import torch
import torch.nn.functional as F
from torch import nn, optim
import pandas as pd
import numpy as np
import argparse
import matplotlib
import random
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataloader import AnomalyDataset
from metrics import *
from model import *
from plot import prepare_plot
from sklearn.metrics import f1_score

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def predict_anomalies(model, dl, criterion, test=False):
    model.load_state_dict(torch.load(f'weights/ec2_transformer.pt'))
    model.eval()
    with torch.no_grad():
        plot_inputs, plot_outputs = [], []
        reconstruction_errors = []
        for step, mb in enumerate(tqdm(dl)):
            inputs = mb.cuda()
            outputs = model(inputs)
            reconstruction_error = criterion(outputs, inputs.squeeze(1))
            # predictions.append(loss.cpu().numpy().flatten)
            reconstruction_errors.append(reconstruction_error.item())
            plot_inputs.append(inputs.squeeze(1))
            plot_outputs.append(outputs)
        # predictions
        plt.plot(plot_inputs[:100])
        plt.plot(plot_outputs[:100], linestyle='--')
        plt.plot(reconstruction_errors[:100], linestyle='--')

        # if test:
        #     plt.plot(plot_inputs[:100])
        #     plt.plot(plot_outputs[:100], linestyle='--')
        plt.savefig(f'transformer_:100.png', dpi=300)
        # plt.close()

    return reconstruction_errors


def save_result(df, pred):
        df['anomaly_predicted'] = pred
        ax = prepare_plot(df)
        predicted_anomalies = df.loc[df['anomaly_predicted'] == 1, ['time_epoch', 'value']]
        ax.scatter(predicted_anomalies['time_epoch'], predicted_anomalies['value'], s=50, color='red')
        plt.savefig('result_transformer.png', dpi=300)
        plt.close()

        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataloader config
    parser.add_argument("--batch_size", default=1, type=int, help="Size of mini batch")
    
    # metric config

    parser.add_argument("--num_epochs", default=20, type=int, help="Number of total training epochs")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="learning rate")
    
    # model config
    parser.add_argument("--num_features", default=1, type=int, help="Number of features")
    parser.add_argument("--num_layers", default=1, type=int, help="Number of layers")
    parser.add_argument("--num_classes", default=1, type=int, help="Number of classes")
    parser.add_argument("--hidden_size", default=10, type=int, help="Number of hidden_size")
    parser.add_argument("--device", default=5, type=int, help="device number of gpu")

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    # model = LSTM(1, 1, 1, 1, 1, bidirectional=True)
    # model = LstmAutoEncoder(1, 1, 32, 1)
    # model = SAE(1)
    # model = StackedGRU(1, 1, 1)

    model = TransAm(feature_size=10)
    model.cuda()


    criterion = nn.MSELoss()
    # criterion = nn.L1Loss(reduction='sum').to(device)
    
    
    tr_path = "/daintlab/data/anomaly_detection/NAB-master/data/realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv"
    tr_filename = "realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv"

    tst_path = "/daintlab/data/anomaly_detection/NAB-master/data/realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv"
    tst_filename = "realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv"

    tr_ds = AnomalyDataset(tr_path, tr_path, tr_filename, tr_filename)
    tr_dl = DataLoader(tr_ds)
    tst_ds = AnomalyDataset(tst_path, tst_path, tst_filename, tst_filename)
    tst_dl = DataLoader(tst_ds, shuffle=False, batch_size=args.batch_size)

    with open('/daintlab/data/anomaly_detection/NAB-master/labels/combined_labels.json') as f:
        labels = json.load(f)
    tst_df = pd.read_csv(tst_path)
    tst_df['anomaly_label'] = tst_df['timestamp'].isin(labels[tst_filename]).astype(int)
    tst_df['timestamp'] = pd.to_datetime(tst_df['timestamp'])
    tst_df['time_epoch'] = tst_df['timestamp'].astype(np.int64)
 
    tr_reconstruction_errors = predict_anomalies(model, tr_dl, criterion)
    print(np.mean(tr_reconstruction_errors))
    threshold = np.mean(tr_reconstruction_errors) + 1 * np.std(tr_reconstruction_errors)
    tst_reconstruction_errors = predict_anomalies(model, tst_dl, criterion)

    print(threshold)

    predicted_test_anomalies = list(map(lambda v: 1 if v > threshold else 0, tst_reconstruction_errors))

    save_result(tst_df, predicted_test_anomalies)    
    anomaly_label = tst_df['anomaly_label'].astype(np.float32)
    
    precision, recall, f1 = CalculatePrecisionRecallF1Metrics(anomaly_label, predicted_test_anomalies)
    fpr, tpr, average_roc_auc = CalculateROCAUCMetrics(anomaly_label, predicted_test_anomalies)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1}")

    print(f"fpr: {fpr}")
    print(f"tpr: {tpr}")
    print(f"average_roc_auc: {average_roc_auc}")
