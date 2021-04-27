import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import Dense, Input, Bidirectional, LSTM, Reshape, Concatenate, Conv1D, TimeDistributed
from tensorflow.keras.models import Model
import sys
import os
import argparse
import yaml
import tensorflow.keras.backend as K
from utils.ssa import SSA
from utils.reprocess_daily import extract_data, ed_extract_data, roll_data
from utils.data_loader import get_input_data
from utils.epoch_size_tuning import get_epoch_size_list


def getMonth(_str):
    return _str.split('/')[1]


def getYear(_str):
    return _str.split('/')[2]


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calcError(row):
    item_df = {}
    item_df['var_score_q'] = r2_score(row['real_q'], row['ensemble_q'])
    item_df['mse_q'] = mean_squared_error(row['real_q'], row['ensemble_q'])
    item_df['mae_q'] = mean_absolute_error(row['real_q'], row['ensemble_q'])
    item_df['mape_q'] = mean_absolute_percentage_error(row['real_q'], row['ensemble_q'])

    item_df['var_score_h'] = r2_score(row['real_h'], row['ensemble_h'])
    item_df['mse_h'] = mean_squared_error(row['real_h'], row['ensemble_h'])
    item_df['mae_h'] = mean_absolute_error(row['real_h'], row['ensemble_h'])
    item_df['mape_h'] = mean_absolute_percentage_error(row['real_h'], row['ensemble_h'])

    return pd.Series(item_df,
                     index=['var_score_q', 'mse_q', 'mae_q', 'mape_q', 'var_score_h', 'mse_h', 'mae_h', 'mape_h'])


class Ensemble:
    def __init__(self, mode, model_kind, sigma_lst=[1, 2, 3], default_n=20, epoch_num=2, epoch_min=100, epoch_step=50, **kwargs):
        self.mode = mode
        self.model_kind = model_kind

        self.log_dir = kwargs.get('log_dir')
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._ssa_kwargs = kwargs.get('ssa')

        self.data_file = self._data_kwargs.get('data_file')
        self.dt_split_point_outer = self._data_kwargs.get('split_point_outer')
        self.dt_split_point_inner = self._data_kwargs.get('split_point_inner')
        self.cols_x = self._data_kwargs.get('cols_x')
        self.cols_y = self._data_kwargs.get('cols_y')
        self.cols_gt = self._data_kwargs.get('cols_gt')
        self.target_timestep = self._data_kwargs.get('target_timestep')
        self.window_size = self._data_kwargs.get('window_size')
        self.norm_method = self._data_kwargs.get('norm_method')

        self.batch_size = self._model_kwargs.get('batch_size')
        self.epoch_min = epoch_min
        self.epoch_num = epoch_num
        self.epoch_step = epoch_step
        self.epoch_max = self.epoch_min + (epoch_num - 1) * epoch_step
        self.epochs_out = self._model_kwargs.get('epochs_out')
        self.input_dim = self._model_kwargs.get('in_dim')
        self.output_dim = self._model_kwargs.get('out_dim')
        self.patience = self._model_kwargs.get('patience')
        self.dropout = self._model_kwargs.get('dropout')

        self.sigma_lst = sigma_lst
        self.default_n = default_n

        self.data = self.generate_data()
        self.inner_model = self.build_model_inner()
        self.outer_model = self.build_model_outer()

    def generate_data(self, true_t_timestep=1):
        dat = get_input_data(self.data_file, self.default_n, self.sigma_lst)
        # dat = getSSAData(self.data_file, 20, [1,2,3]) # thay doi trong config
        #dat_q = pd.read_csv('./RawData/Kontum-daily.csv', header=0, index_col=0)
        #gen_dat = gen_dat.to_numpy()
        dat = dat.to_numpy()

        data = {}
        data['shape'] = dat.shape

        test_outer = int(dat.shape[0] * self.dt_split_point_outer)
        train_inner = int((dat.shape[0] - test_outer) * (1 - self.dt_split_point_inner))
        #train_outer = dat.shape[0] - train_inner - test_outer

        if self.model_kind == 'rnn_cnn':
            x, y, scaler, y_gt = extract_data(dataframe=dat,
                                              window_size=self.window_size,
                                              target_timstep=self.target_timestep,
                                              cols_x=self.cols_x,
                                              cols_y=self.cols_y,
                                              cols_gt=self.cols_gt,
                                              mode=self.norm_method)
            if true_t_timestep != 1:
                _, y_true, _, y_gt = extract_data(dataframe=dat,
                                                  window_size=self.window_size,
                                                  target_timstep=true_t_timestep,
                                                  cols_x=self.cols_x,
                                                  cols_y=self.cols_y,
                                                  cols_gt=self.cols_gt,
                                                  mode=self.norm_method)
                y_test_out_true = y[-test_outer:]
                return y_test_out_true

            x_train_in, y_train_in, y_gt_train_in = x[:train_inner, :], y[:train_inner, :], y_gt[:train_inner, :]
            x_test_in, y_test_in, y_gt_test_in = x[train_inner:-test_outer, :], y[train_inner:-test_outer, :], y_gt[
                train_inner:-test_outer, :]
            x_test_out, y_test_out, y_gt_test_out = x[-test_outer:, :], y[-test_outer:, :], y_gt[-test_outer:, :]

            for cat in ["train_in", "test_in", "test_out"]:
                x, y, y_gt = locals()["x_" + cat], locals()["y_" + cat], locals()["y_gt_" + cat]
                print(cat, "x: ", x.shape, "y: ", y.shape)
                data["x_" + cat] = x
                data["y_" + cat] = y
                data["y_gt_" + cat] = y_gt

        elif self.model_kind == 'en_de':
            en_x, de_x, de_y, scaler = ed_extract_data(dataframe=dat,
                                                       window_size=self.window_size,
                                                       target_timstep=self.target_timestep,
                                                       cols_x=self.cols_x,
                                                       cols_y=self.cols_y,
                                                       mode=self.norm_method)

            en_x_train_in, de_x_train_in, y_train_in = en_x[:train_inner, :], de_x[:
                                                                                   train_inner, :], de_y[:
                                                                                                         train_inner, :]
            en_x_test_in, de_x_test_in, y_test_in = en_x[train_inner:-test_outer, :], de_x[
                train_inner:-test_outer, :], de_y[train_inner:-test_outer, :]
            en_x_test_out, de_x_test_out, y_test_out = en_x[-test_outer:, :], de_x[-test_outer:, :], de_y[
                -test_outer:, :]

            for cat in ["train_in", "test_in", "test_out"]:
                en_x, de_x, de_y = locals()["en_x_" + cat], locals()["de_x_" + cat], locals()["y_" + cat]
                print(cat, "en_x: ", en_x.shape, "de_x: ", de_x.shape, "de_y: ", de_y.shape)
                data["en_x_" + cat] = en_x
                data["de_x_" + cat] = de_x
                data["y_" + cat] = de_y
        #data['y_train_out'] = data['y_test_in']

        data['scaler'] = scaler
        return data

    def build_model_inner(self):
        if self.model_kind == 'rnn_cnn':
            from model.models.multi_rnn_cnn import model_builder
            model = model_builder(self.input_dim, self.output_dim, self.window_size, self.target_timestep, self.dropout)
            model.save_weights(self.log_dir + 'ModelPool/init_model.hdf5')
        elif self.model_kind == 'en_de':
            from model.models.en_de import model_builder
            model = model_builder(self.input_dim, self.output_dim, self.target_timestep, self.dropout)
            model.save_weights(self.log_dir + 'ModelPool/init_model.hdf5')
        return model

    def train_model_inner(self):
        train_shape = self.data['y_test_in'].shape
        test_shape = self.data['y_test_out'].shape
        # print(train_shape)
        # print(test_shape)
        step = int((self.epoch_max - self.epoch_min) / self.epoch_step) + 1
        self.data['sub_model'] = step

        x_train_out = np.zeros(shape=(train_shape[0], self.target_timestep, step, train_shape[1]))
        x_test_out = np.zeros(shape=(test_shape[0], self.target_timestep, step, test_shape[1]))
        j = 0  # submodel index

        lst_epoch_size = get_epoch_size_list(self.epoch_num, self.epoch_min, self.epoch_step)

        if (self.mode == 'train' or self.mode == 'train-inner'):
            from model.models.en_de import train_model as ed_train
            for epoch in lst_epoch_size:
                # for epoch in range(self.epoch_min, self.epoch_max + 1, self.epoch_step):
                self.inner_model.load_weights(self.log_dir + 'ModelPool/init_model.hdf5')

                if self.model_kind == 'rnn_cnn':
                    from model.models.multi_rnn_cnn import train_model
                    self.inner_model, _ = train_model(self.inner_model,
                                                      self.data['x_train_in'],
                                                      self.data['y_train_in'],
                                                      self.batch_size,
                                                      epoch,
                                                      save_dir=self.log_dir + 'ModelPool/')
                elif self.model_kind == 'en_de':
                    from model.models.en_de import train_model
                    self.inner_model, _ = train_model(self.inner_model,
                                                      self.data['en_x_train_in'],
                                                      self.data['de_x_train_in'],
                                                      self.data['y_train_in'],
                                                      self.batch_size,
                                                      epoch,
                                                      save_dir=self.log_dir + 'ModelPool/')
                train, test = self.predict_in()
                # x_train_out = pd.concat([x_train_out,train],axis=1)
                # x_test_out = pd.concat([x_test_out,test],axis=1)
                print(train.shape)
                for i in range(self.target_timestep):
                    x_train_out[:, i, j, :] = train[:, :]
                    x_test_out[:, i, j, :] = test[:, :]
                j += 1
        else:
            for epoch in lst_epoch_size:
                # for epoch in range(self.epoch_min, self.epoch_max + 1, self.epoch_step):
                if self.model_kind == 'rnn_cnn':
                    self.inner_model.load_weights(self.log_dir + f'ModelPool/best_model_{epoch}.hdf5')
                    train, test = self.predict_in()
                elif self.model_kind == 'en_de':
                    self.inner_model.load_weights(self.log_dir + f'ModelPool/ed_best_model_{epoch}.hdf5')
                    train, test = self.predict_in()
                # x_train_out = pd.concat([x_train_out,train],axis=1)
                # x_test_out = pd.concat([x_test_out,test],axis=1)

                for i in range(self.target_timestep):
                    x_train_out[:, i, j, :] = train[:, :]
                    x_test_out[:, i, j, :] = test[:, :]
                j += 1

        self.data_out_generate(x_train_out, x_test_out)

    def predict_in(self, data=[]):
        if data == []:
            if self.model_kind == 'rnn_cnn':
                x_train_out = self.inner_model.predict(self.data['x_test_in'])
                x_test_out = self.inner_model.predict(self.data['x_test_out'])
            elif self.model_kind == 'en_de':
                x_train_out = self.inner_model.predict([self.data['en_x_test_in'], self.data['de_x_test_in']])
                x_test_out = self.inner_model.predict([self.data['en_x_test_out'], self.data['de_x_test_out']])
            return x_train_out, x_test_out
        else:
            num_sub = (self.epoch_max - self.epoch_min) / self.epoch_step + 1
            x_test_out = np.zeros((int(num_sub), self.output_dim))
            for ind, epoch in enumerate(range(self.epoch_min, self.epoch_max + 1, self.epoch_step)):
                if self.model_kind == 'rnn_cnn':
                    self.inner_model.load_weights(self.log_dir + f'ModelPool/best_model_{epoch}.hdf5')
                    x_test_out[ind, :] = self.inner_model.predict(data, batch_size=1)
                elif self.model_kind == 'en_de':
                    self.inner_model.load_weights(self.log_dir + f'ModelPool/ed_best_model_{epoch}.hdf5')
                    x_test_out[ind, :] = self.inner_model.predict(data, batch_size=1)
            x_test_out = x_test_out.reshape(1, -1)
            # print(f'X TEST OUT SHAPE: {x_test_out.shape}')
            return x_test_out

    def data_out_generate(self, x_train_out, x_test_out):
        shape = x_train_out.shape
        self.data['x_train_out'] = x_train_out.reshape(shape[0], shape[1], -1)
        print(self.data['x_train_out'].shape)
        self.data['y_train_out'] = self.data['y_test_in']  # .reshape(-3,self.target_timestep * self.output_dim)
        shape = x_test_out.shape
        self.data['x_test_out_submodel'] = x_test_out.reshape(shape[0], shape[1], -1)
        self.data['y_test_out'] = self.data['y_test_out']  # .reshape(-3,self.target_timestep * self.output_dim)

    # TODO: change to multiple timestep
    def build_model_outer(self):
        self.train_model_inner()
        in_shape = self.data['x_train_out'].shape
        print(f'Input shape: {in_shape}')

        input_submodel = Input(shape=(self.target_timestep, self.output_dim * self.data['sub_model']))
        input_val_x = Input(shape=(self.window_size, self.input_dim))

        rnn_1 = Bidirectional(
            LSTM(units=64,
                 return_sequences=True,
                 return_state=True,
                 dropout=self.dropout,
                 recurrent_dropout=self.dropout))
        rnn_1_out, forward_h, forward_c, backward_h, backward_c = rnn_1(input_val_x)
        state_h = Concatenate(axis=-1)([forward_h, backward_h])
        state_c = Concatenate(axis=-1)([forward_c, backward_c])

        # rnn_2 = LSTM(units=256,return_sequences=True)
        # rnn_2_out = rnn_2(input_submodel,initial_state=[state_h,state_c])

        rnn_2 = LSTM(units=128, return_sequences=False, dropout=self.dropout, recurrent_dropout=self.dropout)
        rnn_2_out = rnn_2(input_submodel, initial_state=[state_h, state_c])

        dense_4 = Dense(units=self.output_dim)
        output = dense_4(rnn_2_out)

        model = Model(inputs=[input_submodel, input_val_x], outputs=output)
        model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])

        return model

    def train_model_outer(self):
        if (self.mode == 'train' or self.mode == 'train-outer'):
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

            callbacks = []
            #lr_schedule = LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
            early_stop = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
            checkpoint = ModelCheckpoint(self.log_dir + 'best_model.hdf5',
                                         monitor='val_loss',
                                         verbose=1,
                                         save_best_only=True)

            # callbacks.append(lr_schedule)
            callbacks.append(early_stop)
            callbacks.append(checkpoint)

            if self.model_kind == 'rnn_cnn':
                history = self.outer_model.fit(x=[self.data['x_train_out'], self.data['x_test_in']],
                                               y=self.data['y_train_out'],
                                               batch_size=self.batch_size,
                                               epochs=self.epochs_out,
                                               callbacks=callbacks,
                                               validation_split=0.1)
            elif self.model_kind == 'en_de':
                history = self.outer_model.fit(x=[self.data['x_train_out'], self.data['en_x_test_in']],
                                               y=self.data['y_train_out'],
                                               batch_size=self.batch_size,
                                               epochs=self.epochs_out,
                                               callbacks=callbacks,
                                               validation_split=0.1)

            if history is not None:
                self.plot_training_history(history)

        elif self.mode == 'test':
            self.outer_model.load_weights(self.log_dir + 'best_model.hdf5')
            print('Load weight from ' + self.log_dir)

        # from keras.utils.vis_utils import plot_model
        # import os
        # os.environ["PATH"] += os.pathsep + 'D:/Graphviz2.38/bin/'
        # plot_model(model=self.outer_model, to_file=self.log_dir + 'model.png', show_shapes=True)

    def plot_training_history(self, history):
        fig = plt.figure(figsize=(10, 6))
        # fig.add_subplot(121)
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend()

        plt.savefig(self.log_dir + 'training_phase.png')

    def predict_and_plot(self):
        # x_extract_by_day = self.data['x_test_out_submodel'][:, 0, :, :]
        if self.model_kind == 'rnn_cnn':
            results = self.outer_model.predict(x=[self.data['x_test_out_submodel'], self.data['x_test_out']])
        elif self.model_kind == 'en_de':
            results = self.outer_model.predict(x=[self.data['x_test_out_submodel'], self.data['en_x_test_out']])
        print(f'The output shape: {results.shape}')

        fig = plt.figure(figsize=(10, 6))
        fig.add_subplot(121)
        plt.plot(self.data['y_test_out'][:, 0, 0], label='ground_truth_Q')
        plt.plot(results[:, 0, 0], label='predict_Q')
        plt.legend()

        fig.add_subplot(122)
        plt.plot(self.data['y_test_out'][:, 0, 1], label='ground_truth_H')
        plt.plot(results[:, 0, 1], label='predict_H')
        plt.legend()

        plt.savefig(self.log_dir + 'predict.png')
        return results

    def roll_prediction(self):
        result = []
        gtruth = []
        for ind in range(len(self.data['x_test_out']) - 7):
            x = self.data['x_test_out'][ind]
            gt = []
            res0_sub = self.predict_in(data=x[np.newaxis, :])
            res0 = self.outer_model.predict(x=[res0_sub[np.newaxis, :], x[np.newaxis, :]], batch_size=1)
            x = x.tolist()
            x.append(res0.reshape(self.output_dim).tolist())
            gt.append(self.data['y_gt_test_out'][ind])
            for i in range(1, 7):
                res_sub = self.predict_in(np.array(x[-self.window_size:])[np.newaxis, :])
                res = self.outer_model.predict(
                    x=[res_sub[np.newaxis, :], np.array(x[-self.window_size:])[np.newaxis, :]], batch_size=1)
                gt.append(self.data['y_gt_test_out'][ind + i])
                x.append(res.reshape(self.output_dim).tolist())

            result.append(x[-7:])
            gtruth.append(gt)

        result = np.array(result)
        gtruth = np.array(gtruth)
        print(f'RESULT SHAPE: {result.shape}')
        print(f'GTRUTH SHAPE: {gtruth.shape}')
        return result, gtruth

    def retransform_prediction(self, mode=''):
        if mode == '':
            result = self.predict_and_plot()
        else:
            result, y_test = self.roll_prediction()
        mask = np.zeros(self.data['shape'])
        test_shape = self.data['y_gt_test_out'].shape[0] - 7

        lst_full_date = pd.read_csv(self.data_file)['date'].tolist()
        # lst_test_date = lst_full_date[int(len(lst_full_date) * (1- self.dt_split_point_outer) - 1) : ]

        for i in range(self.target_timestep if mode == '' else 7):
            total_frame = pd.DataFrame()

            if mode == '':
                mask[-test_shape:, self.cols_gt] = self.data['y_gt_test_out'][:, i, :]
                actual_data = self.data['scaler'].inverse_transform(mask)[-test_shape:, self.cols_gt]
            else:
                mask[-test_shape:, self.cols_gt] = y_test[:, i, :]
                actual_data = self.data['scaler'].inverse_transform(mask)[-test_shape:, self.cols_gt]

            mask[-test_shape:, self.cols_y] = result[:, i, :]
            actual_predict = self.data['scaler'].inverse_transform(mask)[-test_shape:, self.cols_y]

            predict_frame = pd.DataFrame(actual_data[:, 0], columns=['real_q'])
            predict_frame['real_h'] = actual_data[:, 1]

            predict_frame['ensemble_q'] = actual_predict[:, 0]
            predict_frame['ensemble_h'] = actual_predict[:, 1]

            len_df = int(len(lst_full_date) * (1 - self.dt_split_point_outer) - 1)
            predict_frame['date'] = lst_full_date[len_df:len_df + len(actual_data)]
            total_frame = total_frame.append(predict_frame)

            print('SAVING CSV...')
            total_frame.to_csv('./log/data_analysis/predict_val_{}.csv'.format(i), index=None)

    def evaluate_model(self, mode=''):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # actual_dat = self.data['y_test_out']
        # actual_pre = self.predict_and_plot()
        lst_data = []
        for i in range(self.target_timestep if mode == '' else 7):
            df = pd.read_csv('./log/data_analysis/predict_val_{}.csv'.format(i))
            actual_dat = df[['real_q', 'real_h']]
            actual_pre = df[['ensemble_q', 'ensemble_h']]

            item_df = {}
            item_df['var_score_q'] = r2_score(actual_dat.iloc[:, 0], actual_pre.iloc[:, 0])
            item_df['mse_q'] = mean_squared_error(actual_dat.iloc[:, 0], actual_pre.iloc[:, 0])
            item_df['mae_q'] = mean_absolute_error(actual_dat.iloc[:, 0], actual_pre.iloc[:, 0])
            item_df['mape_q'] = mean_absolute_percentage_error(actual_dat.iloc[:, 0], actual_pre.iloc[:, 0])

            item_df['var_score_h'] = r2_score(actual_dat.iloc[:, 1], actual_pre.iloc[:, 1])
            item_df['mse_h'] = mean_squared_error(actual_dat.iloc[:, 1], actual_pre.iloc[:, 1])
            item_df['mae_h'] = mean_absolute_error(actual_dat.iloc[:, 1], actual_pre.iloc[:, 1])
            item_df['mape_h'] = mean_absolute_percentage_error(actual_dat.iloc[:, 1], actual_pre.iloc[:, 1])
            lst_data.append(item_df)
        eval_df = pd.DataFrame(
            data=lst_data,
            columns=['var_score_q', 'mse_q', 'mae_q', 'mape_q', 'var_score_h', 'mse_h', 'mae_h', 'mape_h'])
        eval_df.to_csv('./log/data_analysis/total_error.csv')
        # visualize
        df_viz = pd.read_csv('./log/data_analysis/predict_val_0.csv')
        actual_dat = df_viz[['real_q', 'real_h']]
        actual_pre = df_viz[['ensemble_q', 'ensemble_h']]

        fig = plt.figure(figsize=(10, 6))
        fig.add_subplot(121)
        plt.plot(actual_dat.iloc[:, 0], label='actual_ground_truth_Q')
        plt.plot(actual_pre.iloc[:, 0], label='actual_predict_Q')
        plt.legend()

        fig.add_subplot(122)
        plt.plot(actual_dat.iloc[:, 1], label='ground_truth_H')
        plt.plot(actual_pre.iloc[:, 1], label='predict_H')
        plt.legend()

        plt.savefig(self.log_dir + 'predict_actual.png')

        # write score
        _str = f'Model: H: R2: {np.mean(eval_df["var_score_h"].tolist())} MSE: {np.mean(eval_df["mse_h"].tolist())} MAE: {np.mean(eval_df["mae_h"].tolist())} MAPE: {np.mean(eval_df["mape_h"].tolist())} \
                            \nQ: R2: {np.mean(eval_df["var_score_q"].tolist())} MSE: {np.mean(eval_df["mse_q"].tolist())} MAE: {np.mean(eval_df["mae_q"].tolist())} MAPE: {np.mean(eval_df["mape_q"].tolist()) }\n'

        with open(self.log_dir + 'evaluate_score_total.txt', 'a') as f:
            f.write(_str)

        return (np.mean(eval_df["mse_q"].tolist()), np.mean(eval_df["mse_h"].tolist()))

    def evaluate_model_by_month(self):
        df = pd.read_csv('./log/data_analysis/predict_val_{}.csv'.format(0))
        df['month'] = df['date'].apply(getMonth)
        df['year'] = df['date'].apply(getYear)

        # groupby month year then  calc error
        item_df = df.groupby(['month', 'year'], as_index=False).apply(calcError)

        item_df.to_csv('./log/data_analysis/total_error_monthly.csv')


if __name__ == '__main__':
    K.clear_session()

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str, help='Run mode.')
    parser.add_argument('--model', default='rnn_cnn', type=str, help='Model used.')
    args = parser.parse_args()

    np.random.seed(69)

    with open('./settings/model/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.mode == 'train' or args.mode == 'train-inner' or args.mode == 'train-outer':
        model = Ensemble(args.mode, args.model, sigma_lst=[
                         1, 2, 3], default_n=20, epoch_num=4, epoch_min=100, epoch_step=50, **config)
        model.train_model_outer()
        # model.roll_prediction()
        # model.retransform_prediction()
        # model.evaluate_model()
    elif args.mode == "test":
        model = Ensemble(args.mode, args.model, sigma_lst=[
                         1, 2, 3], default_n=20, epoch_num=4, epoch_min=100, epoch_step=50, **config)
        model.train_model_outer()
        # model.roll_prediction()
        model.retransform_prediction(mode='roll')
        model.evaluate_model(mode='roll')
        # model.evaluate_model_by_month()
    else:
        raise RuntimeError('Mode must be train or test!')
