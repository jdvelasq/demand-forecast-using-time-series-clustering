from training.forecast.forecast_models import train_arima_model
from common.load_input_data import get_dir_main
import pandas as pd
import datetime
import time
import glob
import os

__author__ = "Jose Fernando Montoya Cardona"
__credits__ = ["Jose Fernando Montoya Cardona"]
__email__ = "jomontoyac@unal.edu.co"


def train_each_cluster(train, date_init, date_fin, op_red, type_day, transform='decompose-Fourier'
                       , type_decompose='additive', n_decompose=1, n_coeff_fourier=4):
    print('\n\t\t cluster number: ', train.name)

    train_arima_model(train, date_init, date_fin, op_red, type_day, transform=transform, type_decompose=type_decompose
                      , n_decompose=n_decompose, n_coeff_fourier=n_coeff_fourier)


def dynamic_train(df):
    df_train = df.train
    df_train.sort_values(by='fechahora', ascending=True, inplace=True, ignore_index=True)
    transform, n_decompose, n_f = df.transform_model, df.num_decompose, df.num_coeff_fourier
    date_init, date_fin = df.date_init.strftime('%Y-%m-%d'), df.date_fin.strftime('%Y-%m-%d')
    op_red, type_day, type_decompose = df.cod_op_red, df.type_day, df.type_decompose
    print('Executing op_red: ', op_red, '\n\t type_day: ', type_day)
    cols = list(df_train.columns)
    cols.remove('fechahora')
    start_time = time.time()
    df_train[cols].apply(
        lambda x: train_each_cluster(x, date_init=date_init, date_fin=date_fin, op_red=op_red, type_day=type_day
                                     , transform=transform, n_decompose=n_decompose, n_coeff_fourier=n_f
                                     , type_decompose=type_decompose), axis=0)
    end_time = abs(time.time() - start_time)
    print('duration_pred: ', end_time)


def forecast_train_process(date_init, date_fin, transform='decompose-Fourier', list_num_decompose=(1, 4)
                           , list_num_coeff_fourier=(12, 5), type_decompose='additive'):
    if isinstance(list_num_coeff_fourier, int):
        list_num_coeff_fourier = [list_num_coeff_fourier]
    if isinstance(list_num_decompose, int):
        list_num_decompose = [list_num_decompose]
    start_time = time.time()
    dir_train_name = os.sep.join([get_dir_main(), 'training', 'cluster', 'results', date_init + '_' + date_fin])
    list_files_train = glob.glob(dir_train_name + os.sep + '*.csv')

    date_f_fin = datetime.datetime.strptime(date_fin + ' 23:00:00', '%Y-%m-%d %H:%M:%S')
    date_init = datetime.datetime.strptime(date_init, '%Y-%m-%d')
    date_fin = datetime.datetime.strptime(date_fin, '%Y-%m-%d')
    df_dir_train = pd.DataFrame(list_files_train, columns=['dir_name_train'])
    df_dir_train['date_init'] = date_init
    df_dir_train['date_fin'] = date_f_fin
    df_dir_train['cod_op_red'] = df_dir_train.dir_name_train.apply(lambda x: x.split(os.sep)[-1].split('_')[-2])
    df_dir_train['type_day'] = df_dir_train.dir_name_train.apply(lambda x:
                                                                 x.split(os.sep)[-1].split('_')[-1].split('.')[0])
    df_dir_train['train'] = df_dir_train.apply(
        lambda x: pd.read_csv(x.dir_name_train, sep=',', header=0, encoding='ansi', parse_dates=False)
        , axis=1)

    if transform == 'decompose-Fourier':
        df_param_transform = pd.DataFrame(
            {'num_decompose': list_num_decompose, 'num_coeff_fourier': list_num_coeff_fourier})
        df_param_transform['key'] = 1
        df_dir_train['key'] = 1
        df_dir_train = df_dir_train.merge(df_param_transform, how='left', left_on='key', right_on='key')
        df_dir_train.drop(columns=['key'], inplace=True)
        df_dir_train['transform_model'] = transform
        df_dir_train['type_decompose'] = type_decompose
        df_dir_train.apply(dynamic_train, axis=1)
    print('total_time_execution_forecast_train_process(sec): ', abs(time.time() - start_time))


if __name__ == '__main__':
    d_i = '2017-01-01'
    d_f = '2020-01-03'
    type_transform = 'decompose-Fourier'
    t_decompose = 'additive'
    n_decompose = (1, 2)
    n_coeff_fourier = (75, 65)
    start_process = time.time()
    forecast_train_process(date_init=d_i, date_fin=d_f, transform=type_transform, type_decompose=t_decompose
                           , list_num_decompose=n_decompose, list_num_coeff_fourier=n_coeff_fourier)
    end_process = abs(time.time() - start_process)
    print('total_time_execution (sec): ', end_process)
