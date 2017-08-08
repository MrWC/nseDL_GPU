""" all functions needed for calculating features for mode ID """

import numpy as np

# WALKING_ID_FEATURES = ['STDACC','MEANGYR','TIME_DELTA','STDMEAN_MAG_5WIN']
# WALKING_ID_FEATURES = ['STDACC', 'MOV_AVE_VELOCITY', 'TIME_DELTA']
# IS_VEHICLE_ID_FEATURES = ['STDACC','MEANGYR','MEANMAG','drMEAN_MAG_5WIN','MOV_AVE_VELOCITY','STDPRES_5WIN','NUM_AP']
# IS_VEHICLE_ID_FEATURES = ['STDACC', 'MEANMAG', 'STDMEAN_MAG_WIN', 'MOV_AVE_VELOCITY', 'NUM_AP']

# ['ID', 'NID', 'SGT', 'TIMESTAMP', 'HUMIDITY', 'LIGHT', 'MODE', 'CMODE', 'NOISE', 'PRESSURE', 'STEPS', 'TEMPERATURE',
#  'IRTEMPERATURE', 'MEANMAG', 'MEANGYR', 'STDGYR', 'STDACC', 'MAXACC', 'MAXGYR', 'MAC', 'WLATITUDE', 'WLONGITUDE',
#  'ACCURACY', 'ID_extra', 'lat_clean', 'lon_clean', 'triplabel', 'poilabel', 'ccmode', 'gt_mode_manual',
#  'gt_mode_google', 'gt_mode_app', 'ANALYZED_DATE', 'TIME_SGT', 'TIME_DELTA', 'STEPS_DELTA', 'DISTANCE_DELTA',
#  'VELOCITY', 'ACCELERATION', 'MOV_AVE_VELOCITY', 'MOV_AVE_ACCELERATION', 'BUS_DIST', 'METRO_DIST', 'STDMEAN_MAG_WIN',
#  'drMEAN_MAG_WIN', 'STDPRES_WIN', 'GT_MODE_APP', 'NUM_AP', 'is_localized']

# todo try other features.
# todo how to handle the mixed data

# todo time_delta
# todo previous
# todo

DL_FEATURES = ['MOV_AVE_VELOCITY', 'STDACC', 'MEANMAG', 'MAXGYR', 'PRESSURE', 'STDPRES_WIN', 'NUM_AP',
                'METRO_DIST','BUS_DIST', 'NOISE']
WIN_FEATURES = []


# TODO num_ap
# TODO choose the test sample before balancing
# TODO label_number


def cal_win_label_special_trip_dict(labels, window_size, trip_dict, user_id=-1):
    """
    calculate the window labels for a list of labels
    :param labels: list of labels
    :param window_size: the window size
    :return: list of window labels
    """
    if type(labels) is not np.ndarray:
        labels = np.array(labels)
    result = []
    keys = sorted(trip_dict.keys())
    for key_idx in range(len(keys) - 1):
        # print str(window_size - 1 + trip_dict[keys[key_idx]]) + " to " + str(trip_dict[keys[key_idx + 1]])
        if user_id != -1:
            if trip_dict[keys[key_idx + 1]][1] != user_id:
                continue
        for idx in xrange(window_size - 1 + trip_dict[keys[key_idx]][0], trip_dict[keys[key_idx + 1]][0]):
            result.append(get_win_label(labels[idx - window_size + 1:idx + 1]))
    return result


def cal_win_label(labels, window_size, trip_dict):
    """
    calculate the window labels for a list of labels
    :param labels: list of labels
    :param window_size: the window size
    :return: list of window labels
    """
    if type(labels) is not np.ndarray:
        labels = np.array(labels)
    result = []
    keys = sorted(trip_dict.keys())
    for key_idx in range(len(keys) - 1):
        # print str(window_size - 1 + trip_dict[keys[key_idx]]) + " to " + str(trip_dict[keys[key_idx + 1]])
        for idx in xrange(window_size - 1 + trip_dict[keys[key_idx]], trip_dict[keys[key_idx + 1]]):
            result.append(get_win_label(labels[idx - window_size + 1:idx + 1]))
    return result


def get_win_label(labels_in_win):
    """
    calculate the label for a whole window
    :param labels_in_win: all the point labels in the window
    :return: a label for the window
    """
    result = 5
    labels_in_win = labels_in_win.tolist()
    label_set = set(labels_in_win)
    if len(label_set) == 1:
        result = labels_in_win[0]
    return result


def cal_win_features_special_trip_dict(features, window_size, trip_dict, user_id=-1):
    """
    convert the point features to window features
    :param features: list of point features
    :param window_size: the window size
    :return: list of window features
    """

    if type(features) is not np.ndarray:
        features = np.array(features)
    results = []
    keys = sorted(trip_dict.keys())
    for key_idx in range(len(keys) - 1):
        # print str(window_size - 1 + trip_dict[keys[key_idx]]) + " to " + str(trip_dict[keys[key_idx + 1]])
        if user_id != -1:
            if trip_dict[keys[key_idx + 1]][1] != user_id:
                continue

        for idx in xrange(window_size - 1 + trip_dict[keys[key_idx]][0], trip_dict[keys[key_idx + 1]][0]):
            results.append(get_win_feature(features[idx - window_size + 1:idx + 1]))
    return results


def cal_win_features(features, window_size, trip_dict):
    """
    convert the point features to window features
    :param features: list of point features
    :param window_size: the window size
    :return: list of window features
    """

    if type(features) is not np.ndarray:
        features = np.array(features)
    results = []
    keys = sorted(trip_dict.keys())
    for key_idx in range(len(keys) - 1):
        # print str(window_size - 1 + trip_dict[keys[key_idx]]) + " to " + str(trip_dict[keys[key_idx + 1]])
        for idx in xrange(window_size - 1 + trip_dict[keys[key_idx]], trip_dict[keys[key_idx + 1]]):
            results.append(get_win_feature(features[idx - window_size + 1:idx + 1]))
    return results


def get_win_feature(features_list):
    """
    append all point features into one list
    :param features_list: a list of point features in a window
    :return: the features of a window in a list
    """
    results = []
    for features in features_list:
        for feature in features:
            results.append(feature)
    return results

