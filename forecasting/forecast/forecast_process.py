from forecasting.forecast.update_forecast_models import forecast_arima_model
from common.load_input_data import get_type_day_date, get_dir_main
import pandas as pd
import numpy as np
import datetime
import glob
import time
import os


def get_data_train(date_train_init, date_update_fin):
    dir_train = os.sep.join([get_dir_main(), 'forecasting', 'cluster', 'results', date_train_init + '_' + date_update_fin])
    files_train = glob.glob(dir_train + os.sep + '*.csv')
    df_dir_train = pd.DataFrame(files_train, columns=['dir_name_train'])
    df_dir_train['date_train_init'] = date_train_init
    df_dir_train['date_update_fin'] = date_update_fin
    df_dir_train['cod_op_red'] = df_dir_train.dir_name_train.apply(lambda x: x.split(os.sep)[-1].split('_')[-2])
    df_dir_train['type_day'] = df_dir_train.dir_name_train.apply(lambda x:
                                                                 x.split(os.sep)[-1].split('_')[-1].split('.')[0])
    df_dir_train['train'] = df_dir_train.apply(
        lambda x: pd.read_csv(x.dir_name_train, sep=',', header=0, encoding='ansi', parse_dates=False), axis=1)

    return df_dir_train


def get_train_models(date_train_init, date_train_fin):
    dir_train = os.sep.join([get_dir_main(), 'training', 'forecast', 'models', date_train_init + '_' + date_train_fin])
    files_train = glob.glob(dir_train + os.sep + '*.pkl')
    df_dir_train = pd.DataFrame(files_train, columns=['dir_model_train'])
    df_dir_train['cod_op_red'] = df_dir_train.dir_model_train.apply(lambda x: x.split(os.sep)[-1].split('_')[0])
    df_dir_train['type_day'] = df_dir_train.dir_model_train.apply(lambda x: x.split(os.sep)[-1].split('_')[1])
    df_dir_train['t_transform'] = df_dir_train.dir_model_train.apply(lambda x:
                                                                     x.split(os.sep)[-1].split('_')[-1].split('.')[0])
    n_transform = df_dir_train.t_transform.drop_duplicates().shape[0]
    if n_transform == 1:
        transform = df_dir_train.t_transform.drop_duplicates().values[0]
        if transform == 'decompose-Fourier' or transform == 'decompose':
            df_dir_train['type_decompose'] = df_dir_train.dir_model_train.apply(lambda x:
                                                                                x.split(os.sep)[-1].split('_')[3])
            df_dir_train['num_decompose'] = df_dir_train.dir_model_train.apply(lambda x:
                                                                               x.split(os.sep)[-1].split('_')[-3])
            cols = ['dir_model_train', 'cod_op_red', 'type_day', 't_transform', 'type_decompose', 'num_decompose']
            df_dir_train = df_dir_train[cols]
            df_train = df_dir_train.groupby(by=list(df_dir_train.columns)[1:]).agg(lambda x: ",".join(x)).reset_index()
            return df_train
        else:
            raise ValueError('invalid variable transform {}.'.format(transform))

    else:
        raise ValueError(
            'exist more than one or anything type transform in training models. Number types transform get {}.'.format(
                n_transform))


def get_num_forecast_update(row_train, date_fin, date_train_fin):
    cod_or = [row_train.cod_op_red]
    t_day = [row_train.type_day]
    date_train_fin_t_day = datetime.datetime.strptime(max(row_train.train.fechahora).split()[0], "%Y-%m-%d")
    df_dates = pd.date_range(start=date_train_fin, end=date_fin, freq='D')
    df_dates = df_dates.to_frame(index=False, name='date')
    df_dates['type_day'] = df_dates.date.apply(get_type_day_date)

    d_train_f = date_train_fin_t_day.strftime('%Y-%m-%d')
    d_update_f = row_train.date_update_fin
    d_f = date_fin.strftime('%Y-%m-%d')
    df_dates_update = df_dates.query("type_day in @t_day and date > @date_train_fin and date <= @d_update_f")
    df_dates_forecast = df_dates.query("type_day in @t_day and date > @d_update_f and date <=@d_f")
    num_update = df_dates_update.shape[0]*24
    num_forecast = df_dates_forecast.shape[0]*24
    return pd.Series((num_update, num_forecast))


def get_dates_type_day(t_day, date_init, date_fin):
    df_dates = pd.date_range(start=date_init, end=date_fin, freq='H').to_frame(index=False, name='date')
    df_dates['type_day'] = df_dates.date.apply(get_type_day_date)
    list_t_day = [t_day]
    df_dates_type_day = df_dates.query("type_day in @list_t_day")
    list_dates = list(df_dates_type_day.date.dt.strftime("%Y-%m-%d %H:%M:%S"))
    return list_dates


def forecast_each_cluster(train, num_update, num_forecast, dir_model_train, transform='decompose-Fourier'
                          , type_decompose='additive'):
    pred, pipeline = forecast_arima_model(train, num_update, num_forecast, dir_model_train=dir_model_train
                                          , transform=transform, type_decompose=type_decompose)
    return pred


def dynamic_forecast(df, date_update_fin, date_fin):
    print('\n Executing type day: ', df.type_day)
    df_train, dir_model_train = df.train, df.dir_model_train
    df_train.sort_values(by='fechahora', ascending=True, inplace=True, ignore_index=True)
    transform, type_decompose, n_update, n_pred = df.t_transform, df.type_decompose, df.num_update, df.num_predict
    cols = list(df_train.columns)
    cols.remove('fechahora')
    start_time = time.time()
    if transform == 'decompose-Fourier' or transform == 'decompose':
        pred_clusters = df_train[cols].apply(lambda x: forecast_each_cluster(x, transform=transform
                                                                             , type_decompose=type_decompose
                                                                             , num_update=n_update
                                                                             , num_forecast=n_pred
                                                                             , dir_model_train=dir_model_train), axis=0)
    else:
        raise ValueError('invalid variable transform {}.'.format(transform))
    end_time = abs(time.time()-start_time)
    print('duration_forecast_each_cluster: ', end_time)
    pred = pred_clusters.sum(axis=1)
    date_init_pred = (datetime.datetime.strptime(date_update_fin, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    df_pred_type_day = pd.DataFrame({'dates': get_dates_type_day(df.type_day, date_init_pred, date_fin)
                                    , 'predict': np.array(pred)})
    df_pred_type_day.set_index('dates', inplace=True)
    return df_pred_type_day


def export_results(df, date_update_fin, date_fin):
    ext = '.csv'
    date_init_pred = (datetime.datetime.strptime(date_update_fin, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    dir_save = os.sep.join([get_dir_main(), 'forecasting', 'forecast', 'results'
                            , date_init_pred+'_'+date_fin.strftime('%Y-%m-%d')+ext])
    df.to_csv(dir_save, encoding='ansi', index=False)


def forecast_process(date_train_init, date_train_fin, date_update_fin, date_fin, mod_agg='mean'):
    df_train = get_data_train(date_train_init, date_update_fin)
    df_models = get_train_models(date_train_init, date_train_fin)
    df_forecast = df_train.merge(df_models, how='left', left_on=['cod_op_red', 'type_day']
                                 , right_on=['cod_op_red', 'type_day'])
    date_f_fin = datetime.datetime.strptime(date_fin + ' 23:00:00', '%Y-%m-%d %H:%M:%S')
    date_fin = datetime.datetime.strptime(date_fin, '%Y-%m-%d')
    df_forecast[['num_update', 'num_predict']] = df_forecast.apply(
        lambda x: get_num_forecast_update(x, date_fin, date_train_fin), axis=1)
    df_forecast = df_forecast.query("num_predict not in [0]").reset_index(drop=True)
    df_forecast['pred'] = df_forecast.apply(lambda x: dynamic_forecast(x, date_update_fin, date_f_fin), axis=1)
    if mod_agg == 'mean':
        d_result = df_forecast.groupby(by=['cod_op_red', 'type_day'])['pred'].apply(lambda x:
                                                                                    np.mean(np.array(x))).reset_index()
        d_result.sort_values(by=['cod_op_red', 'dates'], ascending=[1, 1], inplace=True, ignore_index=True)
        export_results(d_result, date_update_fin, date_fin)
    else:
        raise ValueError('invalid variable mod_agg {}.'.format(mod_agg))

    return d_result


if __name__ == '__main__':
    start_process = time.time()
    d_forecast = forecast_process(date_train_init='2017-01-01', date_train_fin='2020-01-03'
                                  , date_update_fin='2020-02-07', date_fin='2020-02-23')
    end_process = abs(time.time() - start_process)
    print('total_time_execution (sec): ', end_process)
