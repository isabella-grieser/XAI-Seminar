from explanationneighbors import get_n_neighbors_information
from explanationplots import *


def create_introduction_page_fig(model, x_pred, feature_names, show_feature_amount=3):
    list_val = x_pred.to_numpy().reshape(1, -1)[0][1:].tolist()
    table1 = create_table(list_val, feature_names)
    neighbors, _ = get_n_neighbors_information(model, x_pred, n_neighbors=1)
    table2 = create_table(neighbors, feature_names)
    feat_fig = create_feature_importance_plot(model, x_pred, feature_names, show_feature_amount=show_feature_amount)
    class_fig = create_class_cluster(model, x_pred)

    return table1, table2, feat_fig, class_fig


def create_deep_dive_page_fig(model, x_pred, feature_names):
    neighbors, _ = get_n_neighbors_information(model, x_pred, n_neighbors=4)
    table_list = []

    list_val = x_pred.to_numpy().reshape(1, -1)[0][1:].tolist()
    table = create_table(list_val, feature_names)
    table_list.append(table)
    for index, row in neighbors.iterrows():
        r = row[1:]
        table = create_table(r, feature_names)
        table_list.append(table)

    return table_list
