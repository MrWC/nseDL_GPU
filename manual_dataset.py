import logging
import pandas as pd
from feature_calc import DL_FEATURES, WIN_FEATURES, cal_win_features_special_trip_dict, cal_win_label_special_trip_dict
from normalization import normalize, win_normalize
import pickle


def get_manual_win_df(window_size):
    manual_features_pt = pd.DataFrame.from_csv('./manual/pt_df/unnormalized_pt_features_df.csv')
    manual_labels_pt = pd.DataFrame.from_csv('./manual/pt_df/unnormalized_pt_labels_df.csv')['pt_label'].tolist()
    with open("./manual/pt_df/trip_dict.txt", "rb") as fp:  # Unpickling
        trip_dict = pickle.load(fp)

    # normalized the point features
    manual_features_pt = normalize(manual_features_pt[DL_FEATURES])
    # features_pt is a Data frame

    print "only collect the manual labelled data with user_id = 1"

    labels_win = cal_win_label_special_trip_dict(manual_labels_pt, window_size, trip_dict, user_id=1)
    features_win = cal_win_features_special_trip_dict(manual_features_pt, window_size, trip_dict, user_id=1)

    # normalize the features for window level
    if len(WIN_FEATURES) > 0:
        features_win = win_normalize(features_win)

    # check whether the features match with labels
    if len(features_win) != len(labels_win):
        logging.warning("the windows features are not matched with labels!!!!!!!!!!!!!!!!!!!!!!")

    manual_win_df = pd.DataFrame(features_win)
    manual_win_df['win_label'] = pd.Series(labels_win)

    # remove the window with label mix
    manual_win_df = manual_win_df[manual_win_df.win_label != 5]
    manual_win_df = manual_win_df[manual_win_df.win_label != -1]
    # now the win_df is unbalanced and has 4 labels
    return manual_win_df
