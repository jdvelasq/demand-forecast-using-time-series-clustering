from common.transform_data import normalize_min_max, standar_z
import pandas as pd
import numpy as np
import warnings


def matrix_features(pv_data, num_arm=60, var_standar_z=True, features=('fourier', 'time')):
    df_time_features = get_time_features(pv_data)
    df_fourier_features = get_fourier_features(pv_data, num_arm)
    if 'fourier' and 'time' in features:
        df_features = pd.merge(df_time_features, df_fourier_features, how='left', left_index=True, right_index=True)
    elif 'fourier' in features:
        df_features = df_fourier_features.copy()
    elif 'time' in features:
        df_features = df_time_features.copy()
    else:
        raise ValueError('invalid feature option {}. Only allow fourier and time options'.format(features))

    if var_standar_z:
        df_features_z = standar_z(df_features)
        return df_features_z
    else:
        return df_features


def get_time_features(pv_data):
    df_feat_time = pv_data.apply(metrics_time, axis=0)
    df_feat_time = df_feat_time.transpose()
    return df_feat_time


def get_fourier_features(pv_data, num_arm=60):
    warnings.filterwarnings("ignore")
    df_fourier = pv_data.apply(lambda x: coeff_fourier(x, num_arm), axis=0)
    columns_coef = ['coef_' + str(i).zfill(3) for i in range(1, num_arm + 1)]
    df_features_fourier = df_fourier.apply(lambda x: x[0].explode()).transpose()
    df_features_fourier.columns = columns_coef
    return df_features_fourier


def metrics_time(serie_data):
    # media
    media = serie_data.mean()
    # desviación estándar
    std = serie_data.std()
    # rms
    rms = ((serie_data ** 2).sum() / serie_data.shape[0]) ** 0.5
    # curtosis
    kurt = serie_data.kurtosis()
    # asimetría
    skew = serie_data.skew()
    # Rango Intercuartil
    irq = serie_data.quantile(0.75) - serie_data.quantile(0.25)
    # Factor de cresta
    if rms != 0:
        fc = serie_data.max() / rms
    else:
        fc = 0
    # Factor de forma
    if media != 0:
        ff = rms / media
    else:
        ff = 0
    # Factor de desviación
    if media != 0:
        fstd = std / media
    else:
        fstd = 0
    # Factor irq
    if rms != 0:
        firq = irq / rms
    else:
        firq = 0

    # Curtosis, asimetría, factor de cresta, factor de forma, factor de desviación, factor de irq
    features = [kurt, skew, fc, ff, fstd]

    name_features = ['kurt', 'skew', 'fc', 'ff', 'fstd']

    df_features = pd.Series(features, index=name_features)
    return df_features


def coeff_fourier(df, num_arm=10, fs=24, with_normalize=True):
    """
    Extracción características: Se obtienen los N primeros coeficientes de la tansformada discreta de fourier

    """
    T_signal = (1 / fs)

    signal = np.array(df)
    signal = signal[~np.isnan(signal)]

    if with_normalize:
        signal = normalize_min_max(signal)

    #     t_samp = signal.shape[0]/fs
    # t_samp = signal.shape[0] / fs
    tam_signal = signal.shape[0]
    t_samp = tam_signal / fs

    t = np.linspace(0, t_samp, tam_signal)  # Vector de tiempo de la señal

    at = t[2] - t[1]  # intervalos de tiempo
    #     AT = signal * at
    an = np.sum(signal * at)  # Coeficiente de serie de fourier

    a1 = 2 * an / t_samp  # Coeficiente de la serie de fourier
    a2 = an / (at * (len(t)))  # Coeficiente de la serie de fourier
    w = (2 * np.pi) / t_samp  # Frecuencia angular
    N = num_arm  # Cantidad de coeficientes de fourier a sacar

    def fourier_iter():
        serie_rec = np.ones(t.shape) * a1 / 2
        for k in range(1, N + 1):
            real = np.cos(w * k * t).reshape(1, tam_signal)
            img = np.sin(w * k * t).reshape(1, tam_signal)
            p_real = (real @ signal) * at  # Parte real
            p_imag = (img @ signal) * at  # Parte imaginaria
            A0 = (2 * p_real) / t_samp
            A1 = (2 * p_real) / t_samp
            B0 = (2 * p_imag) / t_samp
            B1 = (2 * p_imag) / t_samp
            #                 print(A0.shape)
            #                 serie = (a1/2) + A0 @ np.cos(w * k * t) + B0 @ np.sin(w * k * t)
            #                 print(A0.shape, B0.shape, np.cos(w * k * t).shape)
            serie = A0 * np.cos(w * k * t) + B0 * np.sin(w * k * t)
            serie_rec = serie_rec + serie
            # print('ITER: ', k, 'A0: ', A0, 'B0: ', B0, 'p_real: ', p_real, 'p_imag: ', p_imag, 'at: ', at)
            # get_plot_line(pd.DataFrame(serie_rec))
            #                 print(serie.shape)
            C = np.sqrt((A1 ** 2) + (B1 ** 2))  # Coeficiente de fourier
            yield C, serie_rec

    #     fourier_coefs = np.array(list(fourier_iter())).reshape(N,signal.shape[1])
    #     fourier_coefs = np.array(list(fourier_iter())).reshape(-1)
    fourier_coefs = np.array(list(fourier_iter()))
    df_fourier_coefs = pd.DataFrame(data=fourier_coefs)
    feat_fourier_coeff = df_fourier_coefs[0].explode()
    serie_trig_fourier = df_fourier_coefs[1][num_arm - 1]
    #     feat_fourier_coeff.columns = [df.name]
    return feat_fourier_coeff, serie_trig_fourier, t


# def series_fourier(num_arm):
#
#     def cn(num_arm):
#         c = y * np.exp(-1j * 2 * num_arm * np.pi * time / period)
#         return c.sum() / c.size
#
#     def f(x, num_arm):
#         f = np.array([2 * cn(i) * np.exp(1j * 2 * i * np.pi * x / period) for i in range(1, num_arm + 1)])
#         return f.sum()
#
#     y2 = np.array([f(t, 50).real for t in time])
