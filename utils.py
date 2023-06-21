def dummy_to_label(x_pred):
    x = x_pred.to_dict()
    if x["Zahlung"] == 1:
        return "Zahlung"
    if x["Transfer"] == 1:
        return "Transfer"
    if x["Auszahlung"] == 1:
        return "Auszahlung"
    if x["Debit"] == 1:
        return "Debit"
    if x["Einzahlung"] == 1:
        return "Einzahlung"


def replace_dummy(x):
    x_new = x[:-5]
    if x[-1] == 1:
        x_new.append("Transfer")
    if x[-2] == 1:
        x_new.append("Zahlung")
    if x[-3] == 1:
        x_new.append("Debit")
    if x[-4] == 1:
        x_new.append("Auszahlung")
    if x[-5] == 1:
        x_new.append("Einzahlung")
    return x_new


def label_to_dummy(x, label):
    if label == "Zahlung":
        x.extend([0, 0, 0, 1, 0])
    if label == "Transfer":
        x.extend([0, 0, 0, 0, 1])
    if label == "Auszahlung":
        x.extend([0, 1, 0, 0, 0])
    if label == "Debit":
        x.extend([0, 0, 1, 0, 0])
    if label == "Einzahlung":
        x.extend([1, 0, 0, 0, 0])
    return x
