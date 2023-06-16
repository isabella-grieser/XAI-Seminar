import numpy as np
import pandas as pd



def create_explanation_texts(model, x_pred, y_pred, feature_names, feature_description, show_feature_amount=5,
                             threshold=.05):
    explain_val = x_pred.iloc[0].to_numpy().reshape(1, -1)[0]
    model.load_explainer()
    shap = model.explainer.shap_values(x_pred)[0]
    median = pd.read_csv(model.helper_data_path)

    base = model.explainer.expected_value

    val = sum(shap)
    if val < base:
        shap = -shap

    # sort by feature importance
    feature_importance = pd.DataFrame(list(zip(feature_names, abs(shap), shap, explain_val)),
                                      columns=['col_name', 'shap', 'shap2', 'feature_val'])
    feature_importance.sort_values(by=['shap'], ascending=False, inplace=True)

    # get the most important feature names
    most_important_feats = feature_importance['col_name'].to_numpy()[:show_feature_amount]

    # get the feature importance percentage
    importance = abs(feature_importance['shap'].to_numpy())
    feature_importance['shap'] = feature_importance['shap'].apply(
        lambda x: x / np.sum(importance) * 100)
    importance = feature_importance['shap'].to_numpy()[:show_feature_amount]

    full_text = f"Die {len(most_important_feats)} wichtigsten features:"
    comparison_class = " Betrugstransaktion" if y_pred != 1 else "normalen Transaktion"
    rel_median = median[median['isFraud'] != y_pred]

    index = 1 if y_pred != 1 else 0
    all_lines = [full_text]
    for feat_name, feat_val, percent, s in zip(feature_importance['col_name'], feature_importance['feature_val'], importance, feature_importance["shap2"]):
        pro_con = "dafür" if s > 0 else "dagegen"
        full_text = f"Relevanz: {round(percent, 2)}%; Feature spricht {pro_con}: "
        other_median = rel_median[feat_name].to_numpy()[index]
        feature_comparison = feat_val - other_median
        if feature_comparison < other_median * (1 - threshold):
            comp_string = " ist geringer als bei einer "
        elif feature_comparison > other_median * (1 + threshold):
            comp_string = " ist höher als bei einer "
        else:
            comp_string = " ist gleich einer "
        full_text += feature_description[feat_name] + comp_string + comparison_class
        all_lines.append(full_text)

    return all_lines
