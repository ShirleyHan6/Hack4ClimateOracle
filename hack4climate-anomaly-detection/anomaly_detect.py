"""
Reference: https://github.com/chen0040/keras-anomaly-detection
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras_anomaly_detection.library.plot_utils import visualize_reconstruction_error
from keras_anomaly_detection.library.recurrent import LstmAutoEncoder


def main():
    data_dir_path = './data'
    model_dir_path = './models'
    ecg_data = pd.read_csv(data_dir_path + '/ground_anomaly.csv')
    ecg_data = ecg_data[1:]
    # print([name for name in ecg_data.columns])
    ecg_data=ecg_data.drop(['TIMESTAMP', 'RECORD', 'AmbTemp_C_Avg', 'InvPAC_kW_Avg', 'PwrMtrP_kW_Avg'], axis=1)
    ecg_np_data = ecg_data.as_matrix()
    scaler = MinMaxScaler()
    ecg_np_data = scaler.fit_transform(ecg_np_data)

    ae = LstmAutoEncoder()
    print(ecg_data.shape)
    column = ecg_data.shape[0]
    print(column)

    # fit the data and save model into model_dir_path
    ae.fit(ecg_np_data[:10000, :], model_dir_path=model_dir_path, estimated_negative_sample_ratio=0.95)

    # load back the model saved in model_dir_path detect anomaly
    ae.load_model(model_dir_path)
    anomaly_information = ae.anomaly(ecg_np_data[:10000, :])
    reconstruction_error = []
    abnormal_number = 0
    idx_list = []
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        if is_anomaly:
            abnormal_number = abnormal_number + 1
            print(idx)
            idx_list.append(idx)
            print('# ' + str(idx) + ' is abnormal.')
        reconstruction_error.append(dist)
    print(abnormal_number)
    print(idx_list)
    visualize_reconstruction_error(reconstruction_error, ae.threshold)


if __name__ == '__main__':
    main()
