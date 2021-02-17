from common.transform_data import decompose_series_search_periods, get_period_signal_num_k
from common.load_input_data import get_dir_main
from pmdarima.pipeline import Pipeline
from pmdarima import preprocessing as ppc
import pmdarima as pm
import numpy as np
import pickle
import os


def save_model_dir(pipeline, transform, num_cluster, op_red, type_day, type_model, date_init, date_fin
                   , periods_decompose=(), n_decompose='', type_decompose='additive'):
    ext = '.pkl'

    dir_model_save = os.sep.join([get_dir_main(), 'training', 'forecast', 'models', date_init+'_'+date_fin])
    if not os.path.exists(dir_model_save):
        os.makedirs(dir_model_save)
    if 'decompose' in transform:
        filename = '_'.join([op_red, type_day, type_model, type_decompose, 'pd-'+'-'.join(periods_decompose)
                            , 'nd-'+str(n_decompose), 'cluster-'+str(num_cluster), transform])
    elif 'normal' in transform or 'fourier' in transform:
        filename = '_'.join([op_red, type_day, type_model, 'cluster-'+str(num_cluster), transform])
    else:
        raise ValueError('invalid variable transform {}.'.format(transform))

    pickle.dump(pipeline, open(dir_model_save+os.sep+filename+ext, 'wb'))


def get_transform_model(data_train, transform='decompose-Fourier', type_decompose='additive'
                        , filter_decompose=None, n_decompose=1, n_coeff_fourier=13, n_periods_forecast=24):

    if transform == 'decompose':

        forecast_seasonal, trend_residual, periods_decompose = decompose_series_search_periods(data=data_train,
                                                                                    num_decompose=n_decompose,
                                                                                    num_forecast=n_periods_forecast)
        m_f, _, _, _ = get_period_signal_num_k(trend_residual)
        n_diffs_trend_residual = pm.arima.ndiffs(trend_residual, max_d=5)
        ns_diffs_trend_residual = pm.arima.nsdiffs(trend_residual, m=m_f, max_D=5)

        return forecast_seasonal, trend_residual, n_diffs_trend_residual, ns_diffs_trend_residual, periods_decompose, m_f

    elif transform == 'Fourier':

        n_diffs = pm.arima.ndiffs(data_train, max_d=5)
        m_f, k_f, _, _ = get_period_signal_num_k(data_train, n_coeff_fourier)

        return n_diffs, m_f, k_f

    elif transform == 'decompose-Fourier':

        forecast_seasonal, trend_residual, periods_decompose = decompose_series_search_periods(data=data_train,
                                                                                    num_decompose=n_decompose,
                                                                                    num_forecast=n_periods_forecast)
        n_diffs_trend_residual = pm.arima.ndiffs(trend_residual, max_d=5)
        m_f, k_f, _, _ = get_period_signal_num_k(trend_residual, n_coeff_fourier)

        return forecast_seasonal, trend_residual, n_diffs_trend_residual, periods_decompose, m_f, k_f

    elif transform == 'decompose-Fourier-log':

        forecast_seasonal, trend_residual, periods_decompose = decompose_series_search_periods(data=data_train,
                                                                                    num_decompose=n_decompose,
                                                                                    num_forecast=n_periods_forecast)
        trend_residual = np.log(trend_residual)
        n_diffs_trend_residual = pm.arima.ndiffs(trend_residual, max_d=5)
        m_f, k_f, _, _ = get_period_signal_num_k(trend_residual, n_coeff_fourier)

        return forecast_seasonal, trend_residual, n_diffs_trend_residual, periods_decompose, m_f, k_f

    elif transform == 'normal':
        n_diffs = pm.arima.ndiffs(data_train, max_d=5)
        m_f, _, _, _ = get_period_signal_num_k(data_train)
        ns_diffs = pm.arima.nsdiffs(data_train, m=m_f, max_D=5)

        return n_diffs, ns_diffs, m_f
    else:
        raise ValueError('invalid variable transform {}.'.format(transform))


def train_arima_model(data_train, date_init, date_fin, op_red, type_day, transform='decompose-Fourier'
                      , type_decompose='additive', n_decompose=1, n_coeff_fourier=4, filter_decompose=None):
    num_cluster = data_train.name
    data_train = np.array(data_train)[~np.isnan(np.array(data_train))]
    type_model = 'arima'

    if transform == 'decompose-Fourier' or transform == 'decompose-Fourier-log':
        print('n_decompose: ', n_decompose, 'n_coeff_fourier: ', n_coeff_fourier)
        forecast_seasonal, trend_residual, n_diffs, periods_decompose, m_f, k_f = get_transform_model(data_train, transform=transform
                                                                                        , type_decompose=type_decompose
                                                                                        , n_decompose=n_decompose
                                                                                        , n_coeff_fourier=n_coeff_fourier)
        pipeline_trend_residual = Pipeline([
            ('fourier', ppc.FourierFeaturizer(m=m_f, k=k_f))
            , ("model", pm.AutoARIMA(d=n_diffs, seasonal=False, trace=True, error_action='ignore'
                                     , maxiter=30, max_p=4, max_q=4, suppress_warnings=True, with_intercept=True))])
        print('\t\t\t training model...')
        pipeline_trend_residual.fit(trend_residual)
        print(pipeline_trend_residual.summary())
        # aic_model = pipeline_trend_residual.steps[-1][1].model_.aic()
        print('\t\t\t saving model...')
        save_model_dir(pipeline_trend_residual, transform, num_cluster, op_red, type_day, type_model, date_init
                       , date_fin, periods_decompose, str(n_decompose), type_decompose)
        print('\t\t\t finish save model...')
    elif transform == 'Fourier':

        n_diffs, m_f, k_f = get_transform_model(data_train, transform=transform, n_coeff_fourier=n_coeff_fourier)
        pipeline = Pipeline([
            ('fourier', ppc.FourierFeaturizer(m=m_f, k=k_f))
            , ("model", pm.AutoARIMA(d=n_diffs, seasonal=False, trace=True, error_action='ignore'
                                     , maxiter=30, max_p=4, max_q=4, suppress_warnings=True, with_intercept=True))])
        pipeline.fit(data_train)
        save_model_dir(pipeline, transform, num_cluster, op_red, type_day, type_model, date_init, date_fin)

    elif transform == 'decompose':
        forecast_seasonal, trend_residual, n_diffs, ns_diffs, periods_decompose, m_f = get_transform_model(data_train
                                                                                             , transform=transform
                                                                                             , type_decompose=type_decompose
                                                                                             , n_decompose=n_decompose)
        pipeline_trend_residual = Pipeline(
            [("model", pm.AutoARIMA(d=n_diffs, D=ns_diffs, seasonal=True, m=m_f, trace=True, error_action='ignore'
                                    , maxiter=30, max_p=4, max_q=4, suppress_warnings=True, with_intercept=True))])
        pipeline_trend_residual.fit(trend_residual)
        save_model_dir(pipeline_trend_residual, transform, num_cluster, op_red, type_day, type_model, date_init
                       , date_fin, periods_decompose, str(n_decompose), type_decompose)
    elif transform == 'normal':
        n_diffs, ns_diffs, m_f = get_transform_model(data_train, transform=transform)
        pipeline = Pipeline(
            [("model", pm.AutoARIMA(d=n_diffs, D=ns_diffs, seasonal=True, m=m_f, trace=True, error_action='ignore'
                                    , maxiter=30, max_p=4, max_q=4, suppress_warnings=True, with_intercept=True))])
        pipeline.fit(data_train)
        save_model_dir(pipeline, transform, num_cluster, op_red, type_day, type_model, date_init, date_fin)
    else:
        raise ValueError('invalid variable transform {}.'.format(transform))
