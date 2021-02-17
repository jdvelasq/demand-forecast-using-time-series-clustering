from common.transform_data import transform_data, get_matrix_pca
from training.cluster.cluster_models import cluster_kmeans
from common.load_input_data import get_data, get_dir_main
from common.feature_extraction import matrix_features
from training.cluster.opt_k_search import k_optimal
from common.statistic_test import stat_test
import pandas as pd
import pickle
import time
import os


def save_cluster_model_comp_pca(model_cluster, op_red, type_day, date_init, date_fin):

    dir_model_save = os.sep.join([get_dir_main(), 'training', 'cluster', 'models', date_init + '_' + date_fin])

    if not os.path.exists(dir_model_save):
        os.makedirs(dir_model_save)
    filename_model = op_red + '_' + type_day + '_cluster-model.pkl'
    pickle.dump(model_cluster, open(dir_model_save + os.sep + filename_model, 'wb'))


def cluster_train_process(ops_red, types_days, directory_input_data, date_init, date_fin):
    start_time = time.time()
    filename_components = 'n_components_features.csv'
    df_comp = pd.DataFrame()
    for op_red in ops_red:
        print('\n\n Executing OR: ', op_red)
        data_op_red = get_data(directory_input_data, op_red, date_init, date_fin)
        for var_type_day in types_days:
            print('\t type day: ', var_type_day)
            data_op_red_t_day = data_op_red.query('tipodia == @var_type_day')
            dem_data, pv_dem_data = transform_data(data_op_red_t_day, date_init, date_fin)
            m_features = matrix_features(pv_dem_data, features='fourier')
            stat_test(m_features)
            m_pca_features = get_matrix_pca(m_features, show_plot=False, dynamic_component=True)
            df_k_opt, labels, k_means_model = cluster_kmeans(x_train=m_pca_features, k_min=2, k_max=10)
            k = k_optimal(df_k_opt)
            _, _, model_cluster = cluster_kmeans(x_train=m_pca_features, k_min=k, k_max=k + 1)
            save_cluster_model_comp_pca(model_cluster, op_red, var_type_day, date_init, date_fin)
            dict_comp = {'cod_or': [op_red], 'type_day': [var_type_day],
                         'n_components': [m_pca_features.shape[1]]}
            comp = pd.DataFrame(dict_comp)
            df_comp = df_comp.append(comp)
    df_comp.to_csv(
        os.sep.join([get_dir_main(), 'training', 'cluster', 'models', date_init + '_' + date_fin, filename_components])
        , sep=',', encoding='ansi', index=False)
    print('total_time_execution_cluster_train_process(sec): ', abs(time.time() - start_time))


if __name__ == '__main__':
    dir_main = get_dir_main()
    dir_input_data = dir_main + os.sep + 'data' + os.sep + 'input'
    t_days = ['ORD', 'SAB', 'FESTIVO', 'DOM']
    operadores_red = ['ORD1']
    d_i = '2017-01-01'
    d_f = '2020-01-03'
    print(dir_input_data)
    cluster_train_process(operadores_red, t_days, dir_input_data, d_i, d_f)
