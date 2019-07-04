import numpy as np
import pandas as pd
from collections import OrderedDict

# constants
POSSIBLE_BAR_TYPES = ['tick', 'volume', 'dollar']


def sample_bar(df, bar_type, threshold='auto', datetime_col='date_time', 
               price_col='price', vol_col='volume', rounding=-2, auto_ratio=1/50):
    """Alternative bar sampling, such as tick, volume and dollar bars.
    
    Args:
        df (dataframe): tick data dataframe with at least datetime, price and volume columns for the asset
        bar_type (str): the bar type for sampling. Options are 'tick', 'volume' or 'dollar'
        threshold (str, optional): Specify a user threshold for sampling. If set to 'auto', 
        then the threshold will be calculated as 1/50 of daily average of the chosen bar type. Defaults to 'auto'.
        datetime_col (str, optional): column name where datetime values. Defaults to 'date_time'.
        price_col (str, optional): column name where price values. Defaults to 'price'.
        vol_col (str, optional): column name where volume values. Defaults to 'volume'.
        rounding (int, optional): decimal rounding for threshold. Defaults to -2.
        auto_ratio (float, optional): ratio for automatic threshold setting. Defaults to 1/50.
    
    Returns:
        dateframe: dataframe with chosen bar type
    """

    # check if cols exist in df
    assert datetime_col in df.columns, 'Missing {} columns in dataframe'.format(
        datetime_col)
    assert price_col in df.columns, 'Missing {} columns in dataframe'.format(
        price_col)
    assert vol_col in df.columns, 'Missing {} columns in dataframe'.format(
        vol_col)
    
    # check bar type
    assert bar_type in POSSIBLE_BAR_TYPES, 'Expected bar_type to be one of {}, but got {} instead'.format(
        POSSIBLE_BAR_TYPES, bar_type)
    
    # make a copy
    aux = df.copy(deep=True)
    aux['ticks'] = 1
    aux['dollar'] = aux[price_col] * aux[vol_col]
    if bar_type == 'tick':
        bars = _get_tick_bar(aux, threshold=threshold, rounding=rounding, auto_ratio=auto_ratio)
    elif bar_type == 'volume':
        bars = _get_volume_bar(aux, threshold=threshold, rounding=rounding, auto_ratio=auto_ratio)
    else:
        bars = _get_dollar_bar(aux, threshold=threshold, rounding=rounding, auto_ratio=auto_ratio)
    
    return bars


def _get_tick_bar(df, threshold=1000, rounding=-2, auto_ratio=1/50):
    """Tick bar sampling.
    
    Args:
        df (dataframe): tick data dataframe
        threshold (int, optional): threshold for sampling. Defaults to 1000.
        rounding (int, optional): decimal rounding for threshold. Defaults to -2.
        auto_ratio (float, optional): ratio for automatic threshold setting. Defaults to 1/50.
    
    Returns:
        dataframe: dataframe with tick bars
    """
    # assing col with number of ticks
    df['cum_ticks'] = df['ticks'].cumsum()
    
    # get groups
    groups = _assign_groups_threshold(
        df, threshold=threshold, tgt_col='ticks', cum_col='cum_ticks',
        rounding=rounding, auto_ratio=auto_ratio)
    
    # group by groups (of threshold ticks)
    bars = df.groupby(groups, as_index=False).agg(OrderedDict([
        ('date_time', 'last'),
        ('price', ['first', np.max, np.min, 'last']),
        ('ticks', np.sum),
        ('volume', np.sum),
        ('dollar', np.sum),
    ]))
    # rename cols
    bars.columns = bars.columns.droplevel(0)
    bars.columns = ['date_time', 'open', 'high', 'low', 'close', 
                    'ticks', 'volume', 'dollar']
    return bars


def _get_volume_bar(df, threshold='auto', rounding=-2, auto_ratio=1/50):
    """Volume bar sampling.
    
    Args:
        df (dataframe): volume data dataframe
        threshold (int, optional): threshold for sampling. Defaults to 1000.
        rounding (int, optional): decimal rounding for threshold. Defaults to -2.
        auto_ratio (float, optional): ratio for automatic threshold setting. Defaults to 1/50.
    
    Returns:
        dataframe: dataframe with volume bars
    """
    df['cum_vol'] = df['volume'].cumsum()
    
    # get groups
    groups = _assign_groups_threshold(
        df, threshold=threshold, tgt_col='volume', cum_col='cum_vol',
        rounding=rounding, auto_ratio=auto_ratio)
    
    # group by groups (of threshold ticks)
    bars = df.groupby(groups, as_index=False).agg(OrderedDict([
        ('date_time', 'last'),
        ('price', ['first', np.max, np.min, 'last']),
        ('ticks', np.sum),
        ('volume', np.sum),
        ('dollar', np.sum),
    ]))
    # rename cols
    bars.columns = bars.columns.droplevel(0)
    bars.columns = ['date_time', 'open', 'high', 'low', 'close', 
                    'ticks', 'volume', 'dollar']
    return bars


def _get_dollar_bar(df, threshold='auto', rounding=-2, auto_ratio=1/50):
    """Dollar bar sampling.
    
    Args:
        df (dataframe): tick data dataframe
        threshold (int, optional): threshold for sampling. Defaults to 1000.
        rounding (int, optional): decimal rounding for threshold. Defaults to -2.
        auto_ratio (float, optional): ratio for automatic threshold setting. Defaults to 1/50.
    
    Returns:
        dataframe: dataframe with dollar bars
    """
    df['cum_dol'] = df['dollar'].cumsum()
    
    # get groups
    groups = _assign_groups_threshold(
        df, threshold=threshold, tgt_col='dollar', cum_col='cum_dol',
        rounding=rounding, auto_ratio=auto_ratio)
    
    # group by groups (of threshold ticks)
    bars = df.groupby(groups, as_index=False).agg(OrderedDict([
        ('date_time', 'last'),
        ('price', ['first', np.max, np.min, 'last']),
        ('ticks', np.sum),
        ('volume', np.sum),
        ('dollar', np.sum),
    ]))
    # rename cols
    bars.columns = bars.columns.droplevel(0)
    bars.columns = ['date_time', 'open', 'high', 'low', 'close', 
                    'ticks', 'volume', 'dollar']
    return bars


def _get_auto_threshold(df, tgt_col, rounding=-2, auto_ratio=1/50):
    """Calculate automatic threshold, defined as the daily average of chosen tick
    bar, defined in `tgt_col`, adjusted by a ratio, defined in `auto_ratio`.
    
    Args:
        df (dataframe): tick data dataframe
        tgt_col (str): column name for average daily calculation
        rounding (int, optional): decimal rounding for threshold. Defaults to -2.
        auto_ratio (float optional): ratio for automatic threshold setting. Defaults to 1/50.
    
    Returns:
        int: calculated threshold
    """
    # ~1/50 of mean daily number of ticks
    mean_no_ticks = df.set_index('date_time').resample('B').sum()[tgt_col].mean()
    th = np.round(mean_no_ticks * auto_ratio, rounding) # round to the nearest hundred
    return th


def _assign_groups_threshold(df, threshold, tgt_col, cum_col, rounding=-2, auto_ratio=1/50):
    """Helper function for bar labeling. It helps to efficiently group bars by label
    and then get the open high low close values.
    
    Args:
        df (dataframe): tick data dataframe
        threshold (int): threshold for bar samplint.
        tgt_col (str): column name with values of interest.
        cum_col (str): column name with cumulative values of `tgt_col`.
        rounding (int, optional): decimal rounding for threshold. Defaults to -2.
        auto_ratio (float, optional): ratio for automatic threshold setting. Defaults to 1/50.
    
    Returns:
        array-like: array with groups labels
    """
    # make a copy
    tmp = df.copy(deep=True)
    
    # get auto threshold if needed
    if threshold == 'auto':
        th = _get_auto_threshold(df, tgt_col=tgt_col, rounding=rounding, auto_ratio=auto_ratio)
        print('Auto threshold set to {:,}'.format(th))
    else:
        th = threshold   
    
    # define empty group column
    tmp['group'] = np.nan
    
    # define comparison ref
    ref = 0
    group = 0
    ref_row = 0
    for row in range(df.shape[0]):
        if tmp.iloc[row][cum_col] >= th:
            ref = tmp.iloc[row][cum_col] # new ref
            # reset counting
            tmp[cum_col] -= ref
            tmp.iloc[ref_row:(row+1), -1] = group
            group += 1
            ref_row = row + 1
    return tmp['group'].values
