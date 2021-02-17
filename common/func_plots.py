from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

__author__ = "Jose Fernando Montoya Cardona"
__credits__ = ["Jose Fernando Montoya Cardona"]
__email__ = "jomontoyac@unal.edu.co"


def get_plot_pca(ev):
    (figure, ax) = plt.subplots(figsize=(10, 6))
    ax.plot(ev['n_components'], ev['Cum_explained_variance'], '--')
    ax.set_title("Varianza explicada por componente", fontsize=15)
    ax.set_xlabel('Número de componentes', fontsize=15)
    ax.set_ylabel('Varianza acumulada', fontsize=15)
    plt.show()


def get_plot_n_dim_to_two_dim(m_features, labels):
    x_embedded = TSNE(n_components=2).fit_transform(m_features)
    df_x_embedded = pd.DataFrame([x_embedded[:, 0], x_embedded[:, 1], labels]).transpose()
    df_x_embedded.columns = ['X1', 'X2', 'y']
    df_x_embedded.plot.scatter(x='X1', y='X2', c='y', colormap='tab10', figsize=(20, 10), legend=True)
    plt.show()


def get_plot_line(df_dem, idx_date=False):
    (figure, ax) = plt.subplots(figsize=(25, 5))
    if idx_date:
        df_dem.plot(ax=ax)
    else:
        df_dem.reset_index(drop=True).plot(ax=ax)
    plt.show()


def get_plot_acf_pacf(data, cod_or=None, type_day=None, n_lags=24*20, v_alpha=0.05, w_diff=False, num_diff=1):
    if w_diff:
        for i in range(1, num_diff+1):
            data = data.diff(periods=1)
            data = data[~np.isnan(data)]
    data = data[~np.isnan(data)]
    (figure1, ax1) = plt.subplots(figsize=(22, 11))
    plot_acf(data, lags=n_lags, alpha=v_alpha, ax=ax1)
    ax1.set_title('ACF-'+cod_or+'-'+type_day+'_cluster_number-'+data.name)
    plt.show()
    # (figure2, ax2) = plt.subplots(figsize=(22, 11))
    # plot_pacf(data, lags=n_lags, alpha=v_alpha, ax=ax2)
    # ax2.set_title('PACF-'+cod_or + '-' + type_day + '_cluster_number-' + data.name)
    # plt.show()
    # df_train_labels[1].reset_index(drop=True).plot(ax=ax)

