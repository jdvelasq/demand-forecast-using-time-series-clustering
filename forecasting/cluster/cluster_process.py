from common.transform_data import transform_data, get_matrix_pca, group_dem_users_cluster
from common.load_input_data import get_dir_main, get_data
from common.feature_extraction import matrix_features
from common.statistic_test import stat_test
from forecasting.cluster.validation_clusters import reduce_k_cluster
import pandas as pd
import pickle
import glob
import os

__author__ = "Jose Fernando Montoya Cardona"
__credits__ = ["Jose Fernando Montoya Cardona"]
__email__ = "jomontoyac@unal.edu.co"


def save_result_cluster(m_dem_reduce_cluster, op_red, type_day, date_init, date_fin, is_train):
    if is_train:
        dir_save_results = os.sep.join([get_dir_main(), 'training', 'cluster', 'results', date_init + '_' + date_fin])
    else:
        dir_save_results = os.sep.join([get_dir_main(), 'forecasting', 'cluster', 'results', date_init + '_' + date_fin])

    if not os.path.exists(dir_save_results):
        os.makedirs(dir_save_results)

    filename_result = 'cluster-data_' + op_red + '_' + type_day + '.csv'
    m_dem_reduce_cluster.to_csv(dir_save_results + os.sep + filename_result, index=True)


def get_clusters(directory_model, data_train, t_day, op_red):
    files = glob.glob(directory_model+os.sep+'*.pkl')
    df_files = pd.DataFrame(files, columns=['files'])
    df_files['ope_red'] = df_files.files.apply(lambda x: x.split(os.sep)[-1].split('_')[0])
    df_files['type_day'] = df_files.files.apply(lambda x: x.split(os.sep)[-1].split('_')[1])
    directory_filename_model = df_files.query('ope_red == @op_red and type_day == @t_day').files.values[0]
    loaded_model = pickle.load(open(directory_filename_model, 'rb'))
    predict_data = loaded_model.predict(data_train)
    return predict_data


def cluster_process(directory_input_data, ops_red, types_days, date_init_train, date_fin_train, date_init, date_fin
                    , is_train=False):
    dir_load_model_cluster = os.sep.join([get_dir_main(), 'training', 'cluster', 'models'
                                         , date_init_train+'_'+date_fin_train])
    filename_components = 'n_components_features.csv'
    df_comp = pd.read_csv(dir_load_model_cluster + os.sep + filename_components, sep=',', header=0, encoding='ansi')
    for op_red in ops_red:
        print('\n\n Executing OR: ', op_red)
        data_op_red = get_data(directory_input_data, op_red, date_init, date_fin)
        for var_type_day in types_days:
            data_op_red_t_day = data_op_red.query('tipodia == @var_type_day')
            print('\t type day: ', var_type_day)
            dem_data, pv_dem_data = transform_data(data_op_red_t_day, date_init, date_fin)
            m_features = matrix_features(pv_dem_data, features='fourier')
            stat_test(m_features)
            n_comp = df_comp.query('cod_or == @op_red and type_day == @var_type_day').n_components.values[0]
            m_pca_features = get_matrix_pca(m_features, show_plot=False, dynamic_component=False,
                                            n_comp=n_comp)
            labels = get_clusters(dir_load_model_cluster, m_pca_features, var_type_day, op_red)
            m_pca_features['labels'] = labels
            m_dem_cluster = group_dem_users_cluster(dem_data=dem_data, m_features_labels=m_pca_features)
            m_dem_reduce_cluster = reduce_k_cluster(m_dem_cluster, threshold_dem=0.02)
            save_result_cluster(m_dem_reduce_cluster, op_red, var_type_day, date_init, date_fin, is_train)


if __name__ == '__main__':
    dir_main = get_dir_main()
    dir_input_data = dir_main + os.sep + 'data' + os.sep + 'input'
    t_days = ['ORD', 'SAB', 'FESTIVO', 'DOM']
    operadores_red = ['ORD1']
    d_i_train = '2017-01-01'
    d_f_train = '2020-01-03'
    d_i = '2017-01-01'
    d_f = '2020-02-07'
    cluster_process(dir_input_data, operadores_red, t_days, d_i_train, d_f_train, d_i, d_f)
