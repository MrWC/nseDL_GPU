

def create_class_weight(labels_dict):
    factor = max(labels_dict.values())

    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = float(factor)/float(labels_dict[key])
        class_weight[key] = score

    return class_weight
