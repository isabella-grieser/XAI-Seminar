import pandas as pd



def get_n_neighbors(model, x_pred, n_neighbors=1):
    return model.knn.kneighbors(x_pred, n_neighbors=n_neighbors)

def get_n_neighbors_information(model, x_pred, n_neighbors=1):

    neighbors = get_n_neighbors(model, [x_pred] , n_neighbors=n_neighbors)

    distances = neighbors[0][0]
    indexes = neighbors[1][0]
    values = model.x.iloc[indexes]

    return values, distances

