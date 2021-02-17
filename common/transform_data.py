from common.func_plots import get_plot_pca
from common.func_plots import get_plot_line
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from datetime import datetime as ddtime
from scipy import signal
import datetime as dtime
import pmdarima as pm
import pandas as pd
import numpy as np
import math


__author__ = "Jose Fernando Montoya Cardona"
__credits__ = ["Jose Fernando Montoya Cardona"]
__email__ = "jomontoyac@unal.edu.co"


def round_down_to_even(f):
    return math.floor(f / 2.) * 2


def transform_data(df_data, date_i, date_f, get_only_dem=False):
    cols_demh = ['d' + str(i).zfill(2) for i in range(1, 25)]
    cols_df_t = ['fecha', 'codsbm'] + cols_demh
    dft = df_data[cols_df_t]
    cols_dft = [str(i).zfill(2) + ':00:00' for i in range(0, 24)]
    dft.columns = ['fecha', 'codsbm'] + cols_dft
    date_f = (ddtime.strptime(date_f, '%Y-%m-%d')+dtime.timedelta(days=1)).strftime('%Y-%m-%d')

    dft = dft[(dft.fecha >= ddtime.strptime(date_i, '%Y-%m-%d')) & (dft.fecha < ddtime.strptime(date_f, '%Y-%m-%d'))]
    dft = pd.melt(dft, id_vars=['fecha', 'codsbm'], var_name='hora', value_vars=cols_dft, value_name='demanda')
    dft['fechahora'] = dft['fecha'].dt.strftime('%Y-%m-%d') + ' ' + dft['hora']
    dft.fechahora = pd.to_datetime(dft.fechahora)
    dft_sbm = pd.pivot_table(data=dft, values='demanda', columns='codsbm', index=dft.fechahora)

    if get_only_dem:
        dft = dft.groupby(by='fechahora')['demanda'].sum()
        dft = dft.reset_index()

    return dft, dft_sbm


def normalize_min_max(arr):
    '''
    Función que realiza la estandarización Z de las variables

    Entrada: DataFrame --filas: muestras, columnas: características
    Salida: DataFrame estandarizado por columnas, es decir por características

    '''
    arr = arr[~np.isnan(arr)].reshape(-1, 1)
    minmax_scaler = MinMaxScaler()
    df_norm = minmax_scaler.fit_transform(arr)
    #     df_norm = pd.DataFrame(df_norm, columns=df.columns)
    return df_norm.reshape(-1)


def standar_z(df):
    '''
    Función que realiza la estandarización Z de las variables

    Entrada: DataFrame --filas: muestras, columnas: características
    Salida: DataFrame estandarizado por columnas, es decir por características

    '''
    standar_scaler = StandardScaler()
    df_stand = standar_scaler.fit_transform(df)
    df_stand = pd.DataFrame(df_stand, columns=df.columns, index=df.index)
    return df_stand


def get_matrix_pca(matrix_features, exp_variance=0.9, show_plot=False, dynamic_component=True, n_comp=40):
    pca = PCA(n_components=matrix_features.shape[1], svd_solver='full')
    pca.fit(matrix_features)
    ev = pd.DataFrame({'Explained_variance': pca.explained_variance_ratio_,
                       'Cum_explained_variance': np.cumsum(pca.explained_variance_ratio_),
                       'n_components': list(range(1, matrix_features.shape[1] + 1))
                       })
    if dynamic_component:
        n_components = ev[ev['Cum_explained_variance'] <= exp_variance]['n_components'].values[-1]
        print('Getting PCA')
        print('Número de componentes que explican el ', '{:.1f}'.format(exp_variance * 100), '% de la varianza: ',
              n_components)
    else:
        n_components = n_comp
        exp_var = ev[ev['n_components'] == n_components]['Cum_explained_variance'].values[0]
        print('Getting PCA')
        print('Con ', n_components, ' componentes se explica el ', '{:.1f}'.format(exp_var * 100), '% de la varianza')
    pca_int = PCA(n_components=n_components)
    m_pca = pca_int.fit_transform(matrix_features)
    m_pca = pd.DataFrame(m_pca, columns=['PC_' + str(pca).zfill(2) for pca in range(1, n_components + 1)],
                         index=matrix_features.index)

    if show_plot:
        get_plot_pca(ev)

    return m_pca


def group_dem_users_cluster(dem_data, m_features_labels):
    df_labels_sbm = m_features_labels.reset_index()[['codsbm', 'labels']]
    df_group = pd.merge(dem_data, df_labels_sbm, how='left', left_on='codsbm', right_on='codsbm')
    df_group_label = df_group.groupby(by=['fechahora', 'labels'])['demanda'].sum().reset_index()
    df_train_labels = pd.pivot_table(data=df_group_label, values='demanda', columns='labels',
                                     index=df_group_label.fechahora)
    return df_train_labels


def log_transform(s_dem_data):
    s_log_data = pd.Series(np.log(s_dem_data))
    return s_log_data


def get_period_signal_num_k(data, n_coeff_fourier=4):
    f, pxx_den = signal.periodogram(data)
    m_f = round_down_to_even(round(1 / f[list(pxx_den).index(max(pxx_den))], 0))
    if m_f < n_coeff_fourier * 2:
        k_f = round_down_to_even(m_f / 2)
    else:
        k_f = n_coeff_fourier

    return m_f, k_f, f, pxx_den


def conditional_seasonal(seasonal, num_forecast, m, gap_pred):
    if gap_pred + num_forecast > m:
        gap_seasonal = list(seasonal)[gap_pred:m]
        new_n_forecast = gap_pred + num_forecast - m
        ratio = new_n_forecast / m
        ent, res = int(str(ratio).split('.')[0]), int(round((ratio - int(str(ratio).split('.')[0])) * m, 0))
        pred_seasonal = np.array(gap_seasonal + list(seasonal)[0:m] * ent + list(seasonal)[0:res])
        return pred_seasonal
    elif gap_pred + num_forecast <= m:
        pred_seasonal = np.array(list(seasonal)[gap_pred:num_forecast+gap_pred])
        return pred_seasonal


def second_conditional_seasonal(seasonal, num_forecast, m):
    if num_forecast > m:
        ratio = num_forecast / m
        ent, res = int(str(ratio).split('.')[0]), int(round((ratio - int(str(ratio).split('.')[0])) * m, 0))
        pred_seasonal = np.array(list(seasonal)[0:m] * ent + list(seasonal)[0:res])
        return pred_seasonal
    elif num_forecast <= m:
        # pred_seasonal = np.array(list(seasonal)[int(m/2):num_forecast+int(m/2)])
        pred_seasonal = np.array(list(seasonal)[0:num_forecast])
        return pred_seasonal


def get_seasonal(seasonal, num_forecast, m, gap_pred):
    print('seasonal_shape: ', seasonal.shape, 'period: ', m)

    if gap_pred/m <= 1:
        print('condition_gap/m < 1: ', gap_pred/m)
        # pred_seasonal = conditional_seasonal(seasonal, num_forecast, m, gap_pred)
        pred_seasonal = second_conditional_seasonal(seasonal, num_forecast, m)
    else:
        ratio_gap = gap_pred/m
        new_gap_pred = int(round((ratio_gap - int(str(ratio_gap).split('.')[0])) * m, 0))
        # pred_seasonal = conditional_seasonal(seasonal, num_forecast, m, new_gap_pred)
        pred_seasonal = second_conditional_seasonal(seasonal, num_forecast, m)

    return pred_seasonal


def decompose_series_forecast_seasonal(series, m, forecast_seasonal, gap_pred=0, num_forecast=24, type_decompose='additive', filter_decompose=None):
    s_decompose = pm.arima.decompose(series, type_=type_decompose, m=m, filter_=filter_decompose)
    get_plot_line(pd.DataFrame(s_decompose.seasonal))
    seasonal = s_decompose.seasonal

    pred_seasonal = get_seasonal(seasonal, num_forecast, m, gap_pred)

    if type_decompose == 'additive':
        forecast_seasonal += pred_seasonal
    elif type_decompose == 'multiplicative':
        forecast_seasonal = forecast_seasonal * pred_seasonal
    trend = np.array(s_decompose.trend)[~np.isnan(np.array(s_decompose.trend))]
    residual = s_decompose.random[~np.isnan(s_decompose.random)]
    trend_residual = trend + residual
    return trend_residual, forecast_seasonal


def decompose_series_search_periods(data, type_decompose='additive', num_decompose=1, filter_decompose=None, num_forecast=24):
    threshold_power = 6
    gap = 0
    periods_decompose = []
    if type_decompose == 'additive':
        forecast_seasonal = np.zeros(num_forecast)
    elif type_decompose == 'multiplicative':
        forecast_seasonal = np.ones(num_forecast)
    else:
        raise ValueError('invalid variable type decompose {}.'.format(type_decompose))
    for i in range(1, num_decompose+1):
        len_data = len(data)
        val_period, _, f, pxx_den = get_period_signal_num_k(data)
        if val_period < len_data:
            m = val_period
        else:
            periods = 1 / f[np.where(pxx_den >= max(pxx_den) / threshold_power)]
            powers = pxx_den[np.where(pxx_den >= max(pxx_den) / threshold_power)]
            if len(periods) > 1:
                new_periods = periods[1:]
                new_powers = powers[1:]
                m = round_down_to_even(round(new_periods[list(new_powers).index(max(new_powers))], 0))
            else:
                m = val_period

        if m < len_data:
            periods_decompose.append(str(m))
            data, forecast_seasonal = decompose_series_forecast_seasonal(series=data, m=m
                                                                         , forecast_seasonal=forecast_seasonal
                                                                         , num_forecast=num_forecast
                                                                         , type_decompose=type_decompose
                                                                         , filter_decompose=filter_decompose
                                                                         , gap_pred=gap)
            gap += int(m / 2)
        else:
            print('max_num_decompose_possible: ', i-1)

            return forecast_seasonal, data, periods_decompose
    return forecast_seasonal, data, periods_decompose


def decompose_series_with_periods(data, list_periods, type_decompose='additive', filter_decompose=None, num_forecast=24):
    gap = 0
    if type_decompose == 'additive':
        forecast_seasonal = np.zeros(num_forecast)
    elif type_decompose == 'multiplicative':
        forecast_seasonal = np.ones(num_forecast)
    else:
        raise ValueError('invalid variable type decompose {}.'.format(type_decompose))
    for m in list_periods:
        m = int(m)
        len_data = len(data)
        if m < len_data:
            data, forecast_seasonal = decompose_series_forecast_seasonal(data, m, forecast_seasonal
                                                                         , num_forecast=num_forecast
                                                                         , type_decompose=type_decompose
                                                                         , gap_pred=gap)
            gap += int(m / 2)
        else:
            raise ValueError('invalid period {} to decompose because length of signal is {}.'.format(m, len_data))
    return forecast_seasonal, data, gap
