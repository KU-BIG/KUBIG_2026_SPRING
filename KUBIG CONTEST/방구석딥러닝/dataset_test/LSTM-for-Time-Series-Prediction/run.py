from dataset import FNSPIDDataset  # dataset.py에 클래스 정의돼있어야 함
import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model as k_load
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import MeanSquaredError
import tensorflow as tf

class Timer():

	def __init__(self):
		self.start_dt = None

	def start(self):
		self.start_dt = dt.datetime.now()

	def stop(self):
		end_dt = dt.datetime.now()
		print('Time taken: %s' % (end_dt - self.start_dt))

class Model():
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
      print('[Model] Loading model from file %s' % filepath)
      self.model = k_load(filepath, compile=False)
      self.model.compile(loss='mse', optimizer='adam')


    def build_model(self, configs):
        timer = Timer()
        timer.start()

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir, sentiment_type, model_name, num_csvs):
      all_x, all_y = [], []
      for _ in range(int(steps_per_epoch)):
        x_batch, y_batch = next(data_gen)
        all_x.append(x_batch)
        all_y.append(y_batch)
        
      x_train = np.concatenate(all_x, axis=0)
      y_train = np.concatenate(all_y, axis=0)
      
      self.model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[ModelCheckpoint(filepath=os.path.join(save_dir, f"{model_name}_{sentiment_type}_{num_csvs}.h5"), monitor='loss', save_best_only=True)],
        verbose=1,
        shuffle=False
    )
      

    def predict_point_by_point(self, data):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        # Default: window_size， 50, prediction_len，50
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs


    def predict_sequences_multiple_modified(self, data, window_size, prediction_len):
        # window_size = 50, prediction_len = 3
        prediction_seqs = []
        for i in range(0, len(data), prediction_len):
            curr_frame = data[i]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :], verbose=0)[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
                # curr_frame = np.append(curr_frame, predicted[-1])
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
      #Shift the window by 1 new prediction each time, re-run predictions on new window
      print('[Model] Predicting Sequences Full...')
      curr_frame = data[0]
      predicted = []
      for i in range(len(data)):
        predicted.append(self.model.predict(curr_frame[newaxis,:,:], verbose=0)[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
      return predicted

import os
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

import tensorflow as tf
import numpy as np


# 텐서플로우가 즉시 실행 모드에서 함수를 실행하도록 강제합니다.
tf.config.run_functions_eagerly(True)
# 넘파이 변환 에러를 방지하기 위해 디버그 모드를 활성화합니다.
tf.data.experimental.enable_debug_mode()


current_time = datetime.now().strftime("%Y%m%d%H")


def output_results_and_errors_multiple(predicted_data, true_data, true_data_base, prediction_len, file_name,
                                       sentiment_type, num_csvs, folder_name):

    save_df = pd.DataFrame()
										   
    # 데이터 준비
    save_df['True_Data'] = true_data.reshape(-1)
    save_df['Base'] = true_data_base.reshape(-1)

	# 역정규화 : 진짜 주가로 복구
    save_df['True_Data_origin'] = (save_df['True_Data'] + 1) * save_df['Base']

    # 예측 데이터 준비
    if predicted_data:
        all_predicted_data = np.concatenate([p for p in predicted_data])
    else:
        all_predicted_data = predicted_data

    file_name = file_name.split(".")[0]
    sentiment_type = str(sentiment_type)

    # 역정규화 : 진짜 주가로 복구
    save_df['Predicted_Data'] = pd.Series(all_predicted_data)
    save_df['Predicted_Data_origin'] = (save_df['Predicted_Data'] + 1) * save_df['Base']
    save_df = save_df.fillna(np.nan)


    ## 결과 폴더 생성
    # 결과 상위 폴더 이름 결정 :"test_result_5_wN10_k1"
    suffix = folder_name.replace("data_", "") # "wN10_k1"만 추출
    result_folder = f"test_result_{num_csvs}_{suffix}"
    
    # 결과 하위 폴더 이름 (종목_sentiment_시간)
    sub_folder_name = f"{file_name}_{sentiment_type}_{current_time}"

	## 저장될 위치 지정
	# 폴더의 전체 주소 작성하기 (방 생성은 아님)
    full_path = os.path.join(result_folder, sub_folder_name)
	# 폴더 방 실제로 만들기
    os.makedirs(full_path, exist_ok=True)
	# 폴더의 이름 짓기
    save_file_path = os.path.join(full_path, f"{sub_folder_name}_predicted_data.csv")
    # save_df을 폴더에 넣기
    save_df.to_csv(save_file_path, index=False)
    print(f"Data saved to {save_file_path}")


    ## 성능평가 폴더 생성
	# 길이가 다를 경우 짧은 쪽에 맞추기 
    min_length = min(len(save_df['Predicted_Data']), len(save_df['True_Data']))
    predicted_data = save_df['Predicted_Data'][:min_length]
    true_data = save_df['True_Data'][:min_length]

    # Mae, Mse, R2 수행 
    mae = mean_absolute_error(true_data, predicted_data)
    mse = mean_squared_error(true_data, predicted_data)
    r2 = r2_score(true_data, predicted_data)

    # print("MAE:", mae)
    # print("MSE:", mse)
    # print("R²:", r2)
    results_df = pd.DataFrame({
        'MAE': [mae],
        'MSE': [mse],
        'R2': [r2]
    })

    ## 저장될 위치 지정
    eval_file_path = os.path.join(full_path, f"{sub_folder_name}_eval.csv")

    results_df.to_csv(eval_file_path, index=False)
    print(f"\nResults saved to {eval_file_path}")



# Main Function
def main(configs, data_filename, sentiment_type, flag_pred, model_name, num_csvs, folder_name):
    symbol_name = name.split('.')[0]
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    
    base_path = "dataset_test/LSTM-for-Time-Series-Prediction"
    data = FNSPIDDataset(
        os.path.join(base_path, folder_name, data_filename),
        configs['data']['train_test_split'],
        configs['data']['columns'],
        configs['data']['columns_to_normalise'],
        configs['data']['prediction_length']
    )

    model = Model()
    model_path = f"saved_models/{model_name}_{sentiment_type}_{num_csvs}.h5"
    if os.path.exists(model_path):
        model.load_model(model_path)
    else:
        model.build_model(configs)

    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    print("X:", x.shape)
    # print(x[0])
    print("Y:", y.shape)
    # print(y)

    # out-of memory generative training
    steps_per_epoch = math.ceil(
        (data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir'],
        sentiment_type=sentiment_type,
        model_name=model_name,
        num_csvs=num_csvs
    )

    if flag_pred:
        if symbol_name in pred_names:
            print("-----Predicting-----")
            x_test, y_test, y_base = data.get_test_data(
                seq_len=configs['data']['sequence_length'],
                normalise=configs['data']['normalise'],
                cols_to_norm=configs['data']['columns_to_normalise']
            )
            print("test data:")
            print("X:", x_test.shape)
            print("Y:", y_test.shape)
            predictions = model.predict_sequences_multiple_modified(x_test, configs['data']['sequence_length'],
                                                                    configs['data']['prediction_length'])

            output_results_and_errors_multiple(predictions, y_test, y_base, configs['data']['prediction_length'],
                                               symbol_name, sentiment_type, num_csvs, folder_name)


if __name__ == '__main__':
    model_name = "LSTM"
    sentiment_types = ["sentiment"]
    data_folders = [
        "data_wN10_k1", "data_wN10_k3", "data_wN10_k6", 
        "data_wN20_k1", "data_wN20_k3", "data_wN20_k6"
    ]
    # Test csvs = 25
    # names = ['AAPL.csv', 'ABBV.csv', 'ACGLO.csv', 'AFGD.csv', 'AGM-A.csv', 'AKO-A.csv', 'AMD.csv', 'AMZN.csv', 'ARTLW.csv', 'BABA.csv', 'BCDAW.csv', 'BH-A.csv', 'BHFAL.csv', 'BRK-B.csv', 'BROGW.csv', 'C.csv', 'CIG-C.csv', 'CLSN.csv', 'COST.csv', 'CRD-A.csv', 'CVX.csv', 'DIS.csv', 'FDEV.csv', 'FITBO.csv', 'GAINL.csv', 'GE.csv', 'GECCM.csv', 'GOOG.csv', 'GRP-UN.csv', 'GTN-A.csv', 'HCXY.csv', 'HVT-A.csv', 'INBKZ.csv', 'INTC.csv', 'KO.csv', 'MSFT.csv', 'NVDA.csv', 'OCFCP.csv', 'PBR-A.csv', 'PYPL.csv', 'QQQ.csv', 'QVCD.csv', 'SBUX.csv', 'T.csv', 'TSLA.csv', 'TSM.csv', 'UCBIO.csv', 'WFC.csv', 'WMT.csv', 'WSO-B.csv']

    # Test csvs = 50
    names_50 = ['aal.csv', 'AAPL.csv', 'ABBV.csv', 'AMD.csv', 'amgn.csv', 'AMZN.csv', 'BABA.csv',
                'bhp.csv', 'bidu.csv', 'biib.csv', 'BRK-B.csv', 'C.csv', 'cat.csv', 'cmcsa.csv', 'cmg.csv',
                'cop.csv', 'COST.csv', 'crm.csv', 'CVX.csv', 'dal.csv', 'DIS.csv', 'ebay.csv', 'GE.csv',
                'gild.csv', 'gld.csv', 'GOOG.csv', 'gsk.csv', 'INTC.csv', 'KO.csv', 'mrk.csv', 'MSFT.csv',
                'mu.csv', 'nke.csv', 'nvda.csv', 'orcl.csv', 'pep.csv', 'pypl.csv', 'qcom.csv', 'QQQ.csv',
                'SBUX.csv', 'T.csv', 'tgt.csv', 'tm.csv', 'TSLA.csv', 'TSM.csv', 'uso.csv', 'v.csv', 'WFC.csv',
                'WMT.csv', 'xlf.csv']

    # Test csvs = 25
    names_25 = ['AAPL.csv', 'ABBV.csv', 'AMZN.csv', 'BABA.csv', 'BRK-B.csv', 'C.csv', 'COST.csv', 'CVX.csv', 'DIS.csv',
                'GE.csv',
                'INTC.csv', 'MSFT.csv', 'nvda.csv', 'pypl.csv', 'QQQ.csv', 'SBUX.csv', 'T.csv', 'TSLA.csv', 'WFC.csv',
                'KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv']

    # Test csvs = 5
    names_5 = ['KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv']

    all_names = [names_5]
    pred_names = ['KO', 'AMD', "TSM", "GOOG", 'WMT']
    for folder in data_folders:
      for names in all_names:
        num_stocks = len(names)
          # num_stocks = 5
            # num_stocks = 25
            # num_stocks = 50
        # For the first and second runs, only model training was performed
        # In the third run, it will train and make predictions
        for i in range(3):
          if_pred = (i==2)
          for sentiment_type in sentiment_types:
            for name in names:
              print(f"Dataset: {folder} | Stock: {name} | Iteration: {i+1}")
              config_path = f"/content/FNSPID/dataset_test/LSTM-for-Time-Series-Prediction/{sentiment_type}_config.json"
              configs = json.load(open(config_path, 'r'))
              main(configs, name, sentiment_type, if_pred, model_name, num_stocks, folder)
