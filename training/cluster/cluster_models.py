from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd


def cluster_kmeans(x_train, k_min=3, k_max=10):

    silhouette = []
    db = []
    calin = []
    rows = pd.Series(['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score'])

    for i in range(k_min, k_max):
        kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10).fit(x_train)
        s = silhouette_score(x_train, kmeans.labels_, metric='euclidean', random_state=10)
        d = davies_bouldin_score(x_train, kmeans.labels_)
        ca = calinski_harabasz_score(x_train, kmeans.labels_)
        silhouette.append(s)
        db.append(d)
        calin.append(ca)

    sil_opt = silhouette.index(max(silhouette)) + k_min
    dav_opt = db.index(min(db)) + k_min
    cali_opt = calin.index(max(calin)) + k_min

    max_sil = max(silhouette)
    min_dav = min(db)
    max_cali = max(calin)

    max_measure = pd.Series([max_sil, min_dav, max_cali])

    k_opt = pd.Series([sil_opt, dav_opt, cali_opt])

    df_k = pd.DataFrame([silhouette, db, calin], columns=range(k_min, k_max))
    df_k['measure'] = rows
    df_k['k_opt'] = k_opt
    df_k['opt_measure'] = max_measure

    return df_k, kmeans.labels_, kmeans
