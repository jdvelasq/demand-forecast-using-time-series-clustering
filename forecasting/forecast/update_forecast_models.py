from common.transform_data import decompose_series_with_periods
import pandas as pd
import numpy as np
import pickle
import os

__author__ = "Jose Fernando Montoya Cardona"
__credits__ = ["Jose Fernando Montoya Cardona"]
__email__ = "jomontoyac@unal.edu.co"


def load_model_cluster(name_cluster, dir_model_train, transform='decompose-Fourier'):
    n_cluster = [name_cluster]
    files_models = dir_model_train.split(',')
    df_models = pd.DataFrame(files_models, columns=['name_file'])
    df_models['num_cluster'] = df_models.name_file.apply(lambda x: x.split(os.sep)[-1].split('_')[-2].split('-')[1])
    df_cluster_filter = df_models.query('num_cluster in @n_cluster')
    if df_cluster_filter.shape[0] == 0:
        raise ValueError(
            'the {} number cluster does not exist in models training. Must be retrain model.'.format(name_cluster))
    else:
        if transform == 'decompose-Fourier' or transform == 'decompose':
            df_cluster_filter['list_decompose'] = df_cluster_filter.name_file.apply(
                lambda x: x.split(os.sep)[-1].split('_')[-4].split('-')[1:])
            df_cluster_filter['pipeline'] = df_cluster_filter.name_file.apply(lambda x: pickle.load(open(x, 'rb')))
            return df_cluster_filter.pipeline.values[0], df_cluster_filter.list_decompose.values[0]
        else:
            raise ValueError('invalid variable transform {}.'.format(transform))


def forecast_arima_model(data_train, num_update, num_forecast, dir_model_train, transform='decompose-Fourier'
                         , type_decompose='additive', filter_decompose=None):
    print('number_cluster: ', data_train.name)
    pipeline, list_decompose = load_model_cluster(data_train.name, dir_model_train, transform=transform)
    data_train = np.array(data_train)[~np.isnan(np.array(data_train))]
    # print(pipeline.summary())
    if transform == 'decompose-Fourier' or transform == 'decompose':

        if num_update == 0:
            forecast_seasonal, trend_residual, gap = decompose_series_with_periods(data=data_train
                                                                                   , list_periods=list_decompose
                                                                                   , type_decompose=type_decompose
                                                                                   , num_forecast=num_forecast)
            forecast_trend_residual = np.array(pipeline.predict(num_forecast + gap))
            forecast = forecast_seasonal + forecast_trend_residual[gap:]
            return forecast, pipeline
        else:
            forecast_seasonal, trend_residual_new, gap = decompose_series_with_periods(data=data_train[num_update:]
                                                                                       , list_periods=list_decompose
                                                                                       , type_decompose=type_decompose
                                                                                       , num_forecast=num_forecast)
            data_update = trend_residual_new[-num_update:]
            print('\t\t Executing update pipeline...')
            pipeline.update(data_update, maxiter=50)
            print('\t\t Finish update pipeline.')
            # print(pipeline.summary())
            forecast_trend_residual = np.array(pipeline.predict(num_forecast + gap))
            forecast = forecast_seasonal + forecast_trend_residual[gap:]
            return forecast, pipeline

    else:
        raise ValueError('invalid variable transform {}.'.format(transform))