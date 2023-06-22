from explanationneighbors import get_n_neighbors_information
from explanationplots import *
from utils import *

def create_introduction_page_fig(model, x_pred, feature_names, show_feature_amount=3):
    list_val = x_pred.to_numpy().reshape(1, -1)[0].tolist()
    list_val = replace_dummy(list_val)
    feature_names_table = feature_names[:-5].tolist()
    feature_names_table += ["Zahlungsart"]
    table1 = create_table(list_val, feature_names_table)

    neighbors, distances = get_n_neighbors_information(model, x_pred, n_neighbors=1)
    y_hat = model.predict(neighbors)
    prob = model.predict_proba(neighbors)
    n = neighbors.to_numpy().reshape(1, -1)[0].tolist()
    n = replace_dummy(n)
    pred = "Betrug" if y_hat[0] == 1 else "Kein Betrug"
    n = [pred, prob[0][y_hat[0]]] + n
    feature_names_table = ["Vorhersage", "Wahrscheinlichkeit"] + feature_names_table
    table2 = create_table(n, feature_names_table)
    table2.update_layout(title_text=f"Ähnlichster Fall:")
    table2.update_layout({"margin": {"t": 50}})

    feat_fig = create_feature_importance_plot(model, x_pred, feature_names, show_feature_amount=show_feature_amount)
    class_fig = create_class_cluster(model, x_pred)

    return table1, table2, feat_fig, class_fig


def create_neighbors_page_fig(model, x, feature_names):
    x_pred = x.iloc[0]
    neighbors, distances = get_n_neighbors_information(model, x_pred, n_neighbors=3)
    table_list = []
    y_hat = model.predict(neighbors)
    prob = model.predict_proba(neighbors)
    list_val = x_pred.to_numpy().reshape(1, -1)[0].tolist()

    list_val = replace_dummy(list_val)
    feature_names = feature_names.tolist()[:-5] + ["Zahlungsart"]
    table = create_table(list_val, feature_names)
    table_list.append(table)

    feature_names = ["Vorhersage", "Wahrscheinlichkeit"] + feature_names
    i = 0
    for index, row in neighbors.iterrows():
        r = row.to_numpy().reshape(1, -1)[0].tolist()
        r = replace_dummy(r)
        pred = "Betrug" if y_hat[i] == 1 else "Kein Betrug"
        r = [pred, prob[i][y_hat[i]]] + r
        table = create_table(r, feature_names)
        table.update_layout(title_text=f"Nachbar {i+1}")
        table.update_layout({"margin": {"t": 50}})
        table_list.append(table)
        i += 1

    return table_list
