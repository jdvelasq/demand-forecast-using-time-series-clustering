def k_optimal(df_k):
    if df_k['k_opt'].unique().shape[0] != 1:
        if df_k['k_opt'].mode().values[0] == 2:
            k_pre_opt = df_k[df_k['k_opt'] != df_k['k_opt'].mode().values[0]]['k_opt'].values[0]
            return k_pre_opt
        elif df_k['k_opt'].mode().values[0] != 2:
            k_pre_opt = df_k['k_opt'].mode().values[0]
            return k_pre_opt
    else:
        return 2
