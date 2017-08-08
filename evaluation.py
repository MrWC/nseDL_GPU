import numpy as np
from keras.utils.visualize_util import plot
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import os


def evaluate_single_model(model, folder_name, model_name, features_test, labels_test, save_model=True):
    cat_labels_test = np_utils.to_categorical(labels_test)
    loss, acc = model.evaluate(features_test, cat_labels_test, verbose=2)
    write = "**********Evaluating " + str(model_name) + "************\n"
    write += 'Testing data size: ' + str(len(labels_test)) + '\n'
    write += str(Counter(labels_test)) + '\n'
    write += 'loss: ' + str(loss) + ' acc' + str("%.2f" % round(acc, 4)) + '\n'

    result = model.predict(features_test)
    result_label = np.argmax(result, 1)
    gt_label = labels_test

    con_matrix = confusion_matrix(gt_label, result_label)
    write += str(con_matrix) + '\n'
    write += "Classification report:\n"
    write += str(classification_report(gt_label, result_label)) + '\n'

    #  create folder if not exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if save_model:
        model_json = model.to_json()
        with open(folder_name + model_name + "_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(folder_name + model_name + "_model.h5")
        print("Saved model to disk")
        plot(model, to_file=folder_name + model_name + "_model.png", show_shapes=True)

    return write, acc


def evaluate_overall_app(triplet_model, walk_stop_model, car_bus_model, features_test, labels_test):
    write = "**********Evaluate Overall Result**********\n"
    write += "Using App labelled data, with 5 labels\n"
    triplet_result = triplet_model.predict(features_test)
    triplet_result = np.argmax(triplet_result, 1)

    car_bus_result = car_bus_model.predict(features_test)
    car_bus_result = np.argmax(car_bus_result, 1)

    walk_stop_result = walk_stop_model.predict(features_test)
    walk_stop_result = np.argmax(walk_stop_result, 1)
    result_label = []

    for t in list(enumerate(triplet_result)):
        if t[1] == 0:  # stationary or stop
            result_label.append(walk_stop_result[t[0]])
        elif t[1] == 1:  # mrt
            result_label.append(2)
        else:  # t[1] == 2
            if car_bus_result[t[0]] == 0:
                result_label.append(3)
            else:
                result_label.append(4)

    con_matrix = confusion_matrix(labels_test, result_label)
    write += str(con_matrix) + '\n'
    write += "Classification report:\n"
    write += str(classification_report(labels_test, result_label)) + '\n'

    return write


def evaluate_overall_manual(triplet_model, walk_stop_model, car_bus_model, features_test, labels_test):
    write = "**********Evaluate Overall Result**********\n"
    write += "Using manual labelled data, with 4 labels\n"
    triplet_result = triplet_model.predict(features_test)
    triplet_result = np.argmax(triplet_result, 1)

    car_bus_result = car_bus_model.predict(features_test)
    car_bus_result = np.argmax(car_bus_result, 1)

    result_label = []

    for t in list(enumerate(triplet_result)):
        if t[1] == 0:  # stationary or stop
            result_label.append(0)
        elif t[1] == 1:  # mrt
            result_label.append(2)
        else:  # t[1] == 2
            if car_bus_result[t[0]] == 0:
                result_label.append(3)
            else:
                result_label.append(4)

    con_matrix = confusion_matrix(labels_test, result_label)
    write += str(con_matrix) + '\n'
    write += "Classification report:\n"
    write += str(classification_report(labels_test, result_label)) + '\n'

    return write
