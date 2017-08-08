import numpy as np
import math
import pandas as pd
from feature_calc import WIN_FEATURES, DL_FEATURES
from util import aveWithNan

min_dict = {'MOV_AVE_VELOCITY': 0, 'STDACC': -1, 'MEANMAG': -1, 'MAXGYR': -20000, 'PRESSURE': 100000, 'STDPRES_WIN': 0,
               'is_localized': 0, 'NUM_AP': 0, 'LIGHT': 0, 'NOISE': 27, 'STEPS': 96, 'TIME_DELTA': 0}

max_dict = {'MOV_AVE_VELOCITY': 35, 'STDACC': 4700, 'MEANMAG': 5000, 'MAXGYR': 20000, 'PRESSURE': 102000,
            'STDPRES_WIN': 100, 'is_localized': 1, 'NUM_AP': 20, 'LIGHT': 2000, 'NOISE': 90, 'STEPS': 106500, 'TIME_DELTA': 300}

var_dict = {'METRO_DIST': 315000, 'BUS_DIST': 16600}


def normalize(features_df):
    """
    normalize the input features with min-max scaler, [0,1],
    except the bus_dist and metro_dist
    for valid bus_dist and metro_dist, we use Gaussian function to normalize it,
    invalid bus_dist and metro_dist will be put 0
    :param features_df: input feature data frames
    :return: the normalized features in np.array
    """

    scaled_features = pd.DataFrame(columns=list(features_df))
    if 'METRO_DIST' in list(features_df):
        mrt_dist_list = features_df['METRO_DIST'].tolist()

        valid_mrt_bool = list(i!=-1 for i in mrt_dist_list)
        valid_mrt_dist = filter(lambda a: a != -1.0, mrt_dist_list)
        scaled_valid_mrt = []
        scaled_mrt = [0] * len(mrt_dist_list)
        if len(valid_mrt_dist)>0:
            scaled_valid_mrt = gaussian_fun(valid_mrt_dist, 'METRO_DIST')
            # list(map(lambda x: math.exp(-x*x/(2*529.0213))*(x!=-1), mrt_dist_list))
            count = 0
            for i in range(len(scaled_mrt)):
                if valid_mrt_bool[i] is True:
                    scaled_mrt[i] = scaled_valid_mrt[count]
                    count +=1

        scaled_features['METRO_DIST'] = pd.Series(scaled_mrt)

    if 'BUS_DIST' in list(features_df):
        bus_dist_list = features_df['BUS_DIST'].tolist()

        valid_bus_bool = list(i != -1 for i in bus_dist_list)
        valid_bus_dist = filter(lambda a: a != -1.0, bus_dist_list)
        scaled_bus = [0] * len(bus_dist_list)
        if len(valid_bus_dist)>0:
            scaled_valid_bus = gaussian_fun(valid_bus_dist, 'BUS_DIST')
            count = 0
            for i in range(len(scaled_bus)):
                if valid_bus_bool[i] is True:
                    scaled_bus[i] = scaled_valid_bus[count]
                    count += 1

        scaled_features['BUS_DIST'] = pd.Series(scaled_bus)

    #  invalid nan lat or lon will put 0
    if 'WLATITUDE' in list(features_df):
        if 'LAT_WIN' in WIN_FEATURES:
            scaled_features['WLATITUDE'] = features_df['WLATITUDE']
        else:
            min_lat_sg = 1.235578
            max_lat_sg = 1.479055
            lat_list = features_df['WLATITUDE']
            lat_list = list(map(lambda x: (x-min_lat_sg)/(max_lat_sg - min_lat_sg) *(~np.isnan(x)), lat_list))

            #  remove nan in lat_list
            for i in range(len(lat_list)):
                if np.isnan(lat_list[i]):
                    lat_list[i] = 0

            scaled_features['WLATITUDE'] = lat_list

    if 'WLONGITUDE' in list(features_df):
        if 'LON_WIN' in WIN_FEATURES:
            scaled_features['WLONGITUDE'] = features_df['WLONGITUDE']
        else:
            min_lon_sg = 103.565276
            max_lon_sg = 104
            lon_list = features_df['WLONGITUDE']
            lon_list = list(map(lambda x: (x-min_lon_sg)/(max_lon_sg-min_lon_sg) * (~np.isnan(x)), lon_list))

            #  remove nan in lon_list
            for i in range(len(lon_list)):
                if np.isnan(lon_list[i]):
                    lon_list[i] = 0

            scaled_features['WLONGITUDE'] = lon_list

    for col in list(features_df):
        if col in ['BUS_DIST','METRO_DIST','WLONGITUDE','WLATITUDE']:
            continue
        # print "processing column " + str(col)
        # print "min of this column " + str(min_dict[col])
        # print "max of this column " + str(max_dict[col])
        tmp = min_max_normalize(features_df[col].tolist(), min_dict[col], max_dict[col])
        scaled_features[col] = pd.Series(tmp)

    return scaled_features


def cal(num, max, var):
    return math.exp(-(math.pow(num - max, 2)) / (2 * var))


def gaussian_fun(valid_dist_list, feature):
    var = var_dict[feature]
    return list(cal(i,0,var) for i in valid_dist_list)


# bus var 142.52226
# mrt var 529.0213

# WIN_FEATURES = ['LAT_WIN','LON_WIN']
def win_normalize(features_win):
    col_number = 0
    if ('LAT_WIN' in WIN_FEATURES) and ('LON_WIN' in WIN_FEATURES):
        col_number += 1
        #  get lat_idx and lon_idx
        lat_idx = []
        lon_idx = []
        tmp_lat = DL_FEATURES.index("WLATITUDE")
        tmp_lon = DL_FEATURES.index("WLONGITUDE")
        while (tmp_lat < len(features_win[0])):
            lat_idx.append(tmp_lat)
            lon_idx.append(tmp_lon)
            tmp_lat += len(DL_FEATURES)
            tmp_lon += len(DL_FEATURES)

        # add 'LAT_WIN' and 'LON_WIN into features_win
        for win_idx in range(len(features_win)):
            lats = list(features_win[win_idx][i] for i in lat_idx)
            lons = list(features_win[win_idx][i] for i in lon_idx)
            ave_tmp = aveWithNan(np.array(lats))
            features_win[win_idx].append(ave_tmp)

            ave_tmp = aveWithNan(np.array(lons))
            features_win[win_idx].append(ave_tmp)


        # remove 'WLATITUDE' and 'WLONGITUDE' from features_win
        not_latlon_idx = []
        for i in range(len(features_win[0])):
            if i in lat_idx:
                continue
            if i in lon_idx:
                continue
            not_latlon_idx.append(i)
        for win_idx in range(len(features_win)):
            features_win[win_idx] = list(features_win[win_idx][i] for i in not_latlon_idx)

        features_win = np.array(features_win)
        min_lat_sg = 1.235578
        max_lat_sg = 1.479055
        lat_list = features_win[:,-2]
        lat_list = list(map(lambda x: (x - min_lat_sg) / (max_lat_sg - min_lat_sg) * (~np.isnan(x)), lat_list))

        #  remove nan in lat_list
        for i in range(len(lat_list)):
            if np.isnan(lat_list[i]):
                lat_list[i] = 0

        features_win[:, -2] = lat_list

        min_lon_sg = 103.565276
        max_lon_sg = 104.405883
        lon_list = features_win[:,-1]
        lon_list = list(map(lambda x: (x - min_lon_sg) / (max_lon_sg - min_lon_sg) * (~np.isnan(x)), lon_list))

        #  remove nan in lon_list
        for i in range(len(lon_list)):
            if np.isnan(lon_list[i]):
                lon_list[i] = 0

        features_win[:,-1] = lon_list


    return features_win.tolist()


def min_max_normalize(before_normalize, min_value, max_value):
    # invalid value will be nan
    return list(map(lambda x: (x - min_value) / (max_value - min_value) * (~np.isnan(x)), before_normalize))