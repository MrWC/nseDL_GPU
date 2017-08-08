"""
Helper functions and data types.
"""

from itertools import tee, izip
import math
import pandas as pd
from operator import eq
import numpy as np
from sklearn.cluster import DBSCAN

# mean earth radius in kilometers
# https://en.wikipedia.org/wiki/Earth_radius
earth_radius = 6371.0

def great_circle_dist(a, b, unit="kilometers"):
    """
    compute great circle distance between two latitude/longitude coordinate pairs.
    Returns great cirlce distance in kilometers (default) or meters.
    https://en.wikipedia.org/wiki/Great-circle_distance
    """
    lat1, lon1 = a
    lat2, lon2 = b
    if (lat1==92) or (lat2==92):
        return -1 # invalid location gives invalid distance
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) * math.sin(dlat / 2.0) +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
            math.sin(dlon / 2.0) * math.sin(dlon / 2.0))
    c = 2.0 * math.asin(math.sqrt(a))
    dist_km = earth_radius * c
    if unit == "kilometers":
        return dist_km
    elif unit == "meters":
        return dist_km * 1000
    else:
        raise ValueError("Unknown unit: %s" % unit)

    
def dist_to_radians(x):
    return (x / earth_radius) * (180.0 / math.pi)


def sliding_window(iterable, size):
    """ Yield moving windows of size 'size' over the iterable object.
    
    Example:
    >>> for each in sliding_window(xrange(6), 3):
    >>>     print list(each)
    [0, 1, 1]
    [1, 2, 2]
    [2, 3, 3]
    [3, 4, 4]

    """
    iters = tee(iterable, size)
    for i in xrange(1, size):
        for each in iters[i:]:
            next(each, None)
    return izip(*iters)
    
                    
def print_full_dataframe(df):
    pd.set_option('display.max_rows', len(df))
    print(df)
    pd.reset_option('display.max_rows')


def chunks(iterable, include_values=False, equal=eq):
    """Given an iterable, yield tuples of (start, end) indices for chunks
    with equal items. If inlcude_values is True, each tuple will have
    the value of that chunk as a third element.

    Example:
    >>> list(chunks([1, 1, 1, 2, 2, 1]))
    [(0, 3), (3, 5), (5, 6)]

    """
    # return generator: use yield instead of return
    idx = None
    start_idx = 0
    for idx, item in enumerate(iterable):
        if idx == 0:
            previous = item
            continue
        if not equal(item, previous):
            if include_values:
                yield (start_idx, idx, previous)
            else:
                yield (start_idx, idx)
            start_idx = idx
            previous = item
    if idx is not None:
        if include_values:
            yield (start_idx, idx+1, previous)
        else:
            yield (start_idx, idx+1)

def chunks_real(iterable, include_values=False, equal=eq):
    """Given an iterable, yield tuples of (start, end) indices for chunks
    with equal items. If inlcude_values is True, each tuple will have
    the value of that chunk as a third element.

    Example:
    >>> list(chunks([1, 1, 1, 2, 2, 1]))
    [(0, 3), (3, 5), (5, 6)]

    """
    # return iterable: use return instead of yield
    results = []
    idx = None
    start_idx = 0
    for idx, item in enumerate(iterable):
        if idx == 0:
            previous = item
            continue
        if not equal(item, previous):
            if include_values:
                results.append([start_idx, idx, previous])
            else:
                results.append([start_idx, idx])
            start_idx = idx
            previous = item
    if idx is not None:
        if include_values:
            results.append([start_idx, idx+1, previous])
        else:
            results.append([start_idx, idx+1])
    return results

def moving_ave_velocity ( v_array, dt_array, LARGE_TIME_JUMP, window_size ):
    """ calculates moving average velocity for a given velocity array
        Point with large time jumps will block the moving average velocity calculation
    """
    
    moving_ave_v = []
    for idx in xrange(0,len(v_array)):
        if idx<window_size:
            cur_v_array = v_array[0:idx+1]
            cur_dt_array = dt_array[0:idx+1]
        else:
            cur_v_array = v_array[idx-window_size+1:idx+1]
            cur_dt_array = dt_array[idx-window_size+1:idx+1]
        if np.any(cur_dt_array>LARGE_TIME_JUMP):
            # if there's large time jump point
            moving_ave_v.append(aveWithNan(cur_v_array[np.where(cur_dt_array>LARGE_TIME_JUMP)[0][-1]:]))
        else:
            moving_ave_v.append(aveWithNan(cur_v_array))
    # process AVE_VELOCITY to remove np.nan
    ave_v_last = 0
    ave_v_new = []
    for ave_v in moving_ave_v:
        if not np.isnan(ave_v):
            ave_v_last = ave_v
        ave_v_new.append(ave_v_last)
    return ave_v_new

def moving_ave ( data_vec, window_size ):
    """ calculates moving average of a given vecter with given window_size
    """
    if type(data_vec) is not np.ndarray:
        data_vec = np.array(data_vec)
    ave_data_vec = []
    for idx in xrange(0,len(data_vec)):
        if idx<window_size:
            ave_data_vec.append(aveWithNan(data_vec[0:idx+1]))
        else:
            ave_data_vec.append(aveWithNan(data_vec[idx-window_size+1:idx+1]))
    return ave_data_vec

def aveWithNan( data_vec ):
    """ function used to calculate the average of a vector able to deal with vector which has NaN points.
    input must be array, output can be nan
    same as np.nanmean, not useful now
    """
    idx_not_nan = np.where(~np.isnan(data_vec))[0];
    if len(idx_not_nan) == 0:
        # all nan in data_to_cal
        mean_data = np.nan
    else:
        mean_data = np.mean(data_vec[idx_not_nan])
    return mean_data

def moving_std ( data_vec, window_size ):
    """ calculates moving std of a given vecter with given window_size
    """
    if type(data_vec) is not np.ndarray:
        data_vec = np.array(data_vec)
    std_data_vec = []
    for idx in xrange(0,len(data_vec)):
        if idx<window_size:
            std_data_vec.append(np.nanstd(data_vec[0:idx+1]))
        else:
            std_data_vec.append(np.nanstd(data_vec[idx-window_size+1:idx+1]))
    return std_data_vec

def moving_dr ( data_vec, window_size ):
    """ calculates moving dynamic range of a given vecter with given window_size
    """
    if type(data_vec) is not np.ndarray:
        data_vec = np.array(data_vec)
    dr_data_vec = []
    for idx in xrange(0,len(data_vec)):
        if idx<window_size:
            dr_data_vec.append(drWithNan(data_vec[0:idx+1]))
        else:
            dr_data_vec.append(drWithNan(data_vec[idx-window_size+1:idx+1]))
    return dr_data_vec

def drWithNan( data_vec ):
    """ function used to calculate the dynamic range of a vector
    input must be array, output can be nan
    """
    dynamic_range = np.nanmax(data_vec)-np.nanmin(data_vec)
    return dynamic_range

def moving_diff ( data_vec, window_size ):
    """ calculates moving difference of a given vecter with given window_size
    no nan values are allowed
    """
    diff_data_vec = []
    for idx in xrange(0,len(data_vec)):
        if idx<window_size:
            diff_data_vec.append(data_vec[idx]-data_vec[0])
        else:
            diff_data_vec.append(data_vec[idx]-data_vec[idx-window_size+1])
    return diff_data_vec

def get_hour_SGT(timestamp):
    """Get the hour of the day (0-23) in Singapore UTC+8 timezone from a
    unix timestamp. timestamp is a UNIX timestamp in UTC, return value is
    integer between 0 and 23 that represents the hour of the day in SGT
    time for that time stamp."""
    # add 8h (28800 sec) for SGT time
    temp_t = int(timestamp)+28800
    # mod 24h (86400 sec) and integer divide by hours (3600 sec)
    hour = round((temp_t%86400)/3600.0,3)
    return hour

def visualization_for_point(maps, point_list, color):
    '''For trip visualization'''
    for point in point_list:
        if ~np.isnan(point[0]):
            maps.addpoint(point[0],point[1],color)

def visualization_for_radpoint(maps, point_list, color, radias):
    '''For trip visualization'''
    for point in point_list:
        if ~np.isnan(point[0]):
            maps.addradpoint(point[0],point[1],radias,color)

def linePlot(mode_series,y):
    """ plot the given mode segments in one line with different colors
    """
    colors = {0:'ffb200', 1:'#eeefff', 3: 'c', 4: 'r', 5: 'b', 6: 'g'}
    # print mode_series
    for i in mode_series:
        col = colors.get(i, 'k') #'k' is default value if nothing gotten
        # print mode_series[i]
        for seg in mode_series[i]:
            # print seg
            plt.plot(seg, [y, y], color=col, linewidth=5.0)

def chunk2series(chunks_of_modes,ts):
    """ change the chunks of modes into pd.Series which can be imported into
    the line plotting function, also change the idx to sgt_hour_time
    e.g.: 
    input: chunks_of_modes = [(0,60,3),(60,61,None),(61,191,5),(191,193,None),
    (193,206,3)]
    output: series_of_modes = {3:[(6.5,6.7),(7.5,7.6)], 5:[(6.75,7.5)]}

    """
    series_of_modes = {0:[],1:[],3:[],4:[],5:[],6:[]}
    for chunk in chunks_of_modes:
        if chunk[2] is not None:
            series_of_modes[chunk[2]].append((get_hour_SGT(ts[chunk[0]]),\
                get_hour_SGT(ts[chunk[1]-1])))
    return series_of_modes


def apply_DBSCAN(latlon,cluster_radios,cluster_min_samples):
    """ function used to apply DBSCAN on a given list/array of latlon
        distance calculation uses great_circle_dist
    """
    num_pts = len(latlon)
    X = np.zeros([num_pts,num_pts])
    for i in range(num_pts):
        for j in range(num_pts):
            X[i,j] = great_circle_dist((latlon[i][0],latlon[i][1]),(latlon[j][0],latlon[j][1]),unit="meters")
    # clustering
    db = DBSCAN(eps= cluster_radios, min_samples = cluster_min_samples, metric = 'precomputed').fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    return core_samples_mask, db.labels_

