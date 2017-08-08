import logging
import pandas as pd
from feature_calc import DL_FEATURES, cal_win_label, cal_win_features, WIN_FEATURES
from normalization import normalize, win_normalize
import pickle


def get_app_win_df(window_size):
    app_features_pt = pd.DataFrame.from_csv('./pt_df/unnormalized_pt_features_df.csv')
    app_labels_pt = pd.DataFrame.from_csv('./pt_df/unnormalized_pt_labels_df.csv')['pt_label'].tolist()
    with open("./pt_df/trip_dict.txt", "rb") as fp:  # Unpickling
        trip_dict = pickle.load(fp)

    # normalized the point features
    app_features_pt = normalize(app_features_pt[DL_FEATURES])
    # app_features_pt is a Dataframe

    labels_win = cal_win_label(app_labels_pt, window_size, trip_dict)
    features_win = cal_win_features(app_features_pt, window_size, trip_dict)

    # normalize the features for window level
    if len(WIN_FEATURES) > 0:
        features_win = win_normalize(features_win)

    # check whether the features match with labels
    if len(features_win) != len(labels_win):
        logging.warning("the windows features are not matched with labels!!!!!!!!!!!!!!!!!!!!!!")

    app_win_df = pd.DataFrame(features_win)
    app_win_df['win_label'] = pd.Series(labels_win)

    # remove the window with label mix
    app_win_df = app_win_df[app_win_df.win_label != 5]
    # now the win_df is unbalanced and has 5 labels
    return app_win_df
