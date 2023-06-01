
def create_explanation_texts(plot, most_important_feats, importance_percentages, feature_vec, label, median, feature_description, threshold=.05):

    full_text = f"Top {len(most_important_feats)} features:\n"
    comparison_class = " a normal heartbeat signal" if label != "N" else " an atrial fibrillation signal"
    rel_median = median[median['label'] != label]

    for feat_name, percent in zip(most_important_feats, importance_percentages):
        full_text += f"Rel: {round(percent, 2)}%    "
        other_median = rel_median[feat_name].to_numpy()[0]
        feature_comparison = feature_vec[feat_name].to_numpy()[0] - other_median
        if feature_comparison < other_median *( 1 -threshold):
            comp_string = " is lower than for"
        elif feature_comparison > other_median *( 1 +threshold):
            comp_string = " is greater than for"
        else:
            comp_string = " is the same as for"
        full_text += feature_description[feat_name] + comp_string + comparison_class + "\n"

    plot.text(0.005, 0.85, full_text, horizontalalignment='left', verticalalignment='center', transform=plot.transAxes, fontsize=15, va='top', wrap=True)
