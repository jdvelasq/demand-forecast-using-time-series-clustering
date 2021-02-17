import pandas as pd


def conditions_to_reduce(s_data, dem_total_mean, threshold_dem):
    n_null = s_data.tail(48).isna().sum()
    if n_null == 0 and s_data.mean() >= threshold_dem*dem_total_mean:
        return True
    else:
        return False


def reduce_k_cluster(data, threshold_dem):
    dem_total_mean = data.sum(axis=1).mean()
    df = data.apply(lambda x: conditions_to_reduce(x, dem_total_mean, threshold_dem))
    df.name = 'conditions'
    if df.sum() != df.shape[0]:
        dem_mean_c = data.mean()
        dem_mean_c.name = 'mean_dem'
        df_cond = pd.concat([df, dem_mean_c], axis=1)
        idx_max_dem = list(df_cond[df_cond.mean_dem == max(df_cond.mean_dem)].index)
        idx_n_cond = list(df_cond[df_cond.conditions == False].index)
        idx_y_cond = list(df_cond[df_cond.conditions == True].index)
        s_max_dem = data[idx_max_dem+idx_n_cond].sum(axis=1)
        s_max_dem.name = idx_max_dem[0]
        if data[idx_y_cond].shape[1] == 1:
            new_data = s_max_dem
        else:
            idx_y_cond.remove(idx_max_dem[0])
            new_data = pd.concat([data[idx_y_cond], s_max_dem], axis=1)
        print('Reduce cluster do it')
        return new_data
    else:
        print('Not found cluster to reduce')
        return data
