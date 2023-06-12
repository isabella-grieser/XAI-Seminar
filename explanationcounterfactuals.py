

def get_n_counterfactuals(model, x_pred, n_factuals=3):
    dice_exp = model.dice.generate_counterfactuals(x_pred, total_CFs=n_factuals, desired_class="opposite")

    print(dice_exp)

    return dice_exp