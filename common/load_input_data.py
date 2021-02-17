from holidays_co import is_holiday_date
import pandas as pd
import warnings
import glob
import os

__author__ = "Jose Fernando Montoya Cardona"
__credits__ = ["Jose Fernando Montoya Cardona"]
__email__ = "jomontoyac@unal.edu.co"


def get_dir_main():
    abspath = os.path.abspath(__file__)
    dir_name = os.path.dirname(abspath)
    dir_main = os.sep.join(dir_name.split(os.sep)[:-1])
    return dir_main


def get_type_day_date(s_date):
    if is_holiday_date(s_date):
        t_day = 'FESTIVO'
    elif s_date.day_name() == 'Sunday':
        t_day = 'DOM'
    elif s_date.day_name() == 'Saturday':
        t_day = 'SAB'
    else:
        t_day = 'ORD'
    return t_day


def imp_data(directory_input_data, op_red, date_init, date_fin):
    OpRed = [op_red]
    # ltipodia = [type_day]
    files = glob.glob(directory_input_data+os.sep+'*.csv')
    df_files = pd.DataFrame(files, columns=['files'])
    df_files['date_month'] = df_files.files.apply(lambda x: x.split(os.sep)[-1].split('_')[0])
    df_dates = pd.date_range(start=date_init[:-2]+'01'
                             , end=date_fin
                             , freq='MS').strftime('%Y-%m-%d').to_frame(name='u_date_month').reset_index(drop=True)
    df_s_dates = pd.merge(df_files, df_dates, how='inner', left_on='date_month', right_on='u_date_month')
    for filename in list(df_s_dates.files):
        df = pd.read_csv(filename, sep=',', header=0, encoding='ansi')
        # df = df.query("codOR in @OpRed and tipodia in @ltipodia")
        df = df.query("codOR in @OpRed")
        df.reset_index(drop=True, inplace=True)
        # print(filename, len(df.columns))
        yield df


def get_data(directory_input_data, op_red, date_init, date_fin):
    warnings.filterwarnings("ignore")
    df_t = pd.DataFrame([])
    for data in imp_data(directory_input_data, op_red, date_init, date_fin):
        df_t = df_t.append(data, ignore_index=True)

    cols = list(df_t.columns)
    cols.remove('tipodia')
    df_t = df_t[cols]
    df_t.fecha = pd.to_datetime(df_t.fecha)
    print('get type days')
    df_t['tipodia'] = df_t.apply(lambda x: get_type_day_date(x.fecha), axis=1)
    return df_t


if __name__ == '__main__':
    dir_input_data = os.sep.join([get_dir_main(), 'data', 'input'])
    data_OR = get_data(dir_input_data, 'ORD1', '2017-01-01', '2020-07-31')
    data_OR.to_csv('data_op_red.csv', encoding='ansi', index=False)
