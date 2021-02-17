from training.forecast.forecast_train_process import forecast_train_process
from training.cluster.cluster_train_process import cluster_train_process
from forecasting.cluster.cluster_process import cluster_process
from common.load_input_data import get_dir_main
import time
import os

__author__ = "Jose Fernando Montoya Cardona"
__credits__ = ["Jose Fernando Montoya Cardona"]
__email__ = "jomontoyac@unal.edu.co"


def train_process(ops_red, types_days, directory_input_data, date_init, date_fin, transform, num_decompose=(1, 4)
                  , num_coeff_fourier=(13, 5), type_decompose='additive'):
    cluster_train_process(ops_red, types_days, directory_input_data, date_init, date_fin)
    cluster_process(directory_input_data, ops_red, types_days, date_init, date_fin, date_init, date_fin, is_train=True)
    forecast_train_process(date_init=date_init, date_fin=date_fin, transform=transform
                           , list_num_decompose=num_decompose, list_num_coeff_fourier=num_coeff_fourier
                           , type_decompose=type_decompose)


if __name__ == '__main__':
    start_time = time.time()
    dir_main = get_dir_main()
    dir_input_data = dir_main + os.sep + 'data' + os.sep + 'input'
    t_days = ['ORD', 'SAB', 'FESTIVO', 'DOM']
    operadores_red = ['ORD1']
    type_transform = 'decompose-Fourier'
    t_decompose = 'additive'
    n_decompose = (1, 2)
    n_coeff_fourier = (35, 75)
    d_i = '2017-01-01'
    d_f = '2020-01-03'
    train_process(operadores_red, t_days, dir_input_data, d_i, d_f, type_transform, n_decompose, n_coeff_fourier
                  , type_decompose=t_decompose)
    print('total_time_execution_training_process(sec): ', abs(time.time() - start_time))
