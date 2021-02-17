from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import numpy as np

__author__ = "Jose Fernando Montoya Cardona"
__credits__ = ["Jose Fernando Montoya Cardona"]
__email__ = "jomontoyac@unal.edu.co"


def stat_test(mat_features):
    m_corr = np.array(mat_features.corr())
    print('determinant feature matrix is: ', np.linalg.det(m_corr))

    chi_square_value, p_value = calculate_bartlett_sphericity(m_corr)
    print('Bartlett test')
    print('value of chi square: ', chi_square_value, 'p-value: ', p_value)

    kmo_all, kmo_model = calculate_kmo(m_corr)
    print('Kaiser-Meyer-Olkin test')
    print('value of kmo: ', kmo_model)
