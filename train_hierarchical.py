import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau
from evaluation import evaluate_single_model, evaluate_overall_app, evaluate_overall_manual
from feature_calc import DL_FEATURES
import random
import datetime
from class_weight import create_class_weight
from collections import Counter
import app_dataset
import manual_dataset


begin_time = datetime.datetime.now()
train_opt = {'batch_size': 32, 'nb_epoch': 200, 'window_size': 6, 'DLNetwork': 'FULL', 'num_test': 0.333,
             'test_opt': 'random', 'l2': 0.00001,
             'remark': 'testing only'}
folder_name = './hierarchical/evaluation_report/' + str(datetime.datetime.now().strftime("%y-%m-%d %H:%M")) + '/'


def main():
    # get app_win_df
    app_win_df = app_dataset.get_app_win_df(train_opt['window_size'])
    # get manual_win_df
    manual_win_df = manual_dataset.get_manual_win_df(train_opt['window_size'])
    # set the win_label of manual_win_df as app_win_df
    # hence how manual_win_df does noe have win_label 1
    manual_win_df.win_label[(manual_win_df.win_label == 3)] = 4  # car
    manual_win_df.win_label[(manual_win_df.win_label == 2)] = 3  # bus
    manual_win_df.win_label[(manual_win_df.win_label == 1)] = 2  # mrt

    # divide app_win_df into app_win_df_test and app_win_df_train
    test_idx = random.sample(range(len(app_win_df)), int(len(app_win_df) * train_opt['num_test']))
    train_idx = [x for x in range(len(app_win_df)) if x not in test_idx]
    app_win_df_test = app_win_df.iloc[test_idx, :]
    app_win_df_train = app_win_df.iloc[train_idx, :]
    # divide manual_win_df into manual_win_df_test and manual_win_df_train
    test_idx = random.sample(range(len(manual_win_df)), int(len(manual_win_df) * train_opt['num_test']))
    train_idx = [x for x in range(len(manual_win_df)) if x not in test_idx]
    manual_win_df_test = manual_win_df.iloc[test_idx, :]
    manual_win_df_train = manual_win_df.iloc[train_idx, :]

    # train triplet model
    triplet_train_win_df = app_win_df_train.append(manual_win_df_train)

    # 0: walk or stationary
    # 2: mrt
    # 3: bus or car
    triplet_train_win_df.win_label[(triplet_train_win_df.win_label == 1)] = 0  # walk or stationary
    triplet_train_win_df.win_label[(triplet_train_win_df.win_label == 2)] = 1  # mrt
    triplet_train_win_df.win_label[(triplet_train_win_df.win_label == 3)] = 2  # bus or car
    triplet_train_win_df.win_label[(triplet_train_win_df.win_label == 4)] = 2  # bus or car

    triplet_model = train_model(triplet_train_win_df)

    # train walk VS stop
    walk_stop_train_win_df = app_win_df_train.copy()

    # 0: stationary
    # 1: walk
    walk_stop_train_win_df = walk_stop_train_win_df[(walk_stop_train_win_df.win_label == 0)
                                                    | (walk_stop_train_win_df.win_label == 1)]

    walk_stop_model = train_model(walk_stop_train_win_df)

    # train car VS bus
    car_bus_train_win_df = app_win_df_train.append(manual_win_df_train)

    # 3: bus
    # 4: car
    car_bus_train_win_df = car_bus_train_win_df[(car_bus_train_win_df.win_label == 4)
                                                | (car_bus_train_win_df.win_label == 3)]

    car_bus_train_win_df.win_label[(car_bus_train_win_df.win_label == 3)] = 0  # bus
    car_bus_train_win_df.win_label[(car_bus_train_win_df.win_label == 4)] = 1  # car

    car_bus_model = train_model(car_bus_train_win_df)

    write = "Evaluation Report of Train_Hierarchical \n"
    write += "Train option: " + str(train_opt) + "\n"
    write += "Features: " + str(DL_FEATURES) + "\n"
    write += "triplet_train_win_df: \n "
    write += str(Counter(triplet_train_win_df['win_label'])) + '\n'
    write += "walk_stop_train_win_df: \n "
    write += str(Counter(walk_stop_train_win_df['win_label'])) + '\n'
    write += "car_bus_train_win_df: \n "
    write += str(Counter(car_bus_train_win_df['win_label'])) + '\n'
    # Evaluate Triplet
    triplet_test_win_df = app_win_df_test.append(manual_win_df_test)

    # 0: walk or stationary
    # 2: mrt
    # 3: bus or car
    triplet_test_win_df.win_label[(triplet_test_win_df.win_label == 1)] = 0  # walk or stationary
    triplet_test_win_df.win_label[(triplet_test_win_df.win_label == 2)] = 1  # mrt
    triplet_test_win_df.win_label[(triplet_test_win_df.win_label == 3)] = 2  # bus or car
    triplet_test_win_df.win_label[(triplet_test_win_df.win_label == 4)] = 2  # bus or car

    write_trip, acc_trip = evaluate_single_model(triplet_model, folder_name, "triplet",
                                                 np.array(triplet_test_win_df.iloc[:, :-1]),
                                                 np.array(triplet_test_win_df['win_label']))

    # Evaluate walk VS stop
    walk_stop_test_win_df = app_win_df_test.copy()

    # 0: stationary
    # 1: walk
    walk_stop_test_win_df = walk_stop_test_win_df[(walk_stop_test_win_df.win_label == 0)
                                                  | (walk_stop_test_win_df.win_label == 1)]

    write_walk_stop, acc_walk_stop = evaluate_single_model(walk_stop_model, folder_name, "walkVSstop",
                                                           np.array(walk_stop_test_win_df.iloc[:, :-1]),
                                                           np.array(walk_stop_test_win_df['win_label']))

    # Evaluate car VS bus
    car_bus_test_win_df = app_win_df_test.append(manual_win_df_test)

    # 0: bus
    # 1: car
    car_bus_test_win_df = car_bus_test_win_df[(car_bus_test_win_df.win_label == 4)
                                              | (car_bus_test_win_df.win_label == 3)]

    car_bus_test_win_df.win_label[(car_bus_test_win_df.win_label == 3)] = 0  # bus
    car_bus_test_win_df.win_label[(car_bus_test_win_df.win_label == 4)] = 1  # car

    write_car_bus, acc_car_bus = evaluate_single_model(car_bus_model, folder_name, "carVSbus",
                                                       np.array(car_bus_test_win_df.iloc[:, :-1]),
                                                       np.array(car_bus_test_win_df['win_label']))

    write += write_trip + write_car_bus + write_walk_stop

    write += evaluate_overall_app(triplet_model, walk_stop_model, car_bus_model, np.array(app_win_df_test.iloc[:, :-1]),
                              np.array(app_win_df_test['win_label']))

    write += evaluate_overall_manual(triplet_model, walk_stop_model, car_bus_model,
                                     np.array(manual_win_df_test.iloc[:, :-1]),
                                     np.array(manual_win_df_test['win_label']))

    with open(folder_name + "report_acc_%0.2f_%0.2f_%0.2f.txt" % (acc_trip, acc_car_bus, acc_walk_stop), 'w') as f:
        f.truncate()
        f.write(write)
        f.close()

    print write


def train_model(train_win_df):
    features_train = train_win_df.iloc[:, 0:-1]
    labels_train = train_win_df['win_label']

    # create class weight for imbalanced classes
    class_weight = create_class_weight(Counter(labels_train))

    # one hot encode the output variable
    cat_labels_train = np_utils.to_categorical(labels_train)

    # build model
    input_shape = len(list(features_train))
    inputs = Input(shape=(input_shape,))

    layer1 = Dense(128, activation='relu', W_regularizer=l2(train_opt['l2']))(inputs)
    layer2 = Dense(96, activation='relu', W_regularizer=l2(train_opt['l2']))(layer1)
    layer3 = Dense(64, activation='relu', W_regularizer=l2(train_opt['l2']))(layer2)
    layer4 = Dense(32, activation='relu', W_regularizer=l2(train_opt['l2']))(layer3)
    drop1 = Dropout(0.2)(layer4)
    layer5 = Dense(16, activation='relu', W_regularizer=l2(train_opt['l2']))(drop1)

    predictions = Dense(len(set(labels_train)), activation='softmax', W_regularizer=l2(train_opt['l2']))(layer5)
    model = Model(input=inputs, output=predictions)
    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=5, min_lr=0.001)

    model.fit(np.array(features_train), np.array(cat_labels_train), verbose=2, nb_epoch=train_opt['nb_epoch'],
              batch_size=train_opt['batch_size'], class_weight=class_weight, callbacks=[reduce_lr])

    return model

main()