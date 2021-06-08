"""
 Date-related utilities.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import MonthBegin, MonthEnd, BMonthEnd

DCT_HOLIDAYS = dict()


def eobm(date, country=None):
    """
    Replace datetime values with business month end.

    Parameters
    ----------
    date : datetime or pd.Series
        Various dates in the month.
    country : str
        2-letter ISO country code.

    Returns
    -------
    datetime or pd.Series
        Corresponding business month end dates.
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    eobm_date = date + MonthBegin(-1) + BMonthEnd(0)
    if country is None:
        return eobm_date
    s_holidays = get_holidays(country)
    while (eobm_date in s_holidays.to_list()) or (datetime.isoweekday(eobm_date) >= 6):
        eobm_date += pd.Timedelta(-1, 'd')
    return eobm_date


def pbme(date, country=None):
    """
    Replace datetime values with previous business month end.

    Parameters
    ----------
    date : datetime or pd.Series
        Various dates in the month.
    country : str
        2-letter ISO country code.

    Returns
    -------
    datetime or pd.Series
        Corresponding business month end dates.
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    pbme_date = date + BMonthEnd(-1)
    if country is None:
        return pbme_date
    s_holidays = get_holidays(country)
    while (pbme_date in s_holidays.to_list()) or (datetime.isoweekday(pbme_date) >= 6):
        pbme_date += pd.Timedelta(-1, 'd')
    return pbme_date


def eom(date):
    """
    Replace pd.Series of datetime values with calendar month end.

    :param date: pd.Series of datetime with various dates in month
    :return: pd.Series of datetime with only calendar month end dates
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    return date + MonthEnd(0)


def pme(date):
    """
    Replace pd.Series of datetime values with previous calendar month end.

    :param date: pd.Series of datetime with various dates in month
    :return: pd.Series of datetime with only previous calendar month end dates
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    return date + MonthEnd(-1)


def today():
    """ Returns today's date as datetime.datetime object using pandas' datetime.now().
    """
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


def month_to_datetime(month):
    if isinstance(month, np.ndarray):
        out = pd.Series(index=month)
        for month_i in range(month.size):
            out.iloc[month_i] = datetime(int(month[month_i] / 100),
                                         month[month_i] - int(month[month_i] / 100) * 100, 28)
        return out.get_values()
    elif isinstance(month, pd.Series):
        return month_to_datetime(pd.to_numeric(month).get_values())
    elif isinstance(month, np.int64):
        return datetime(int(month / 100), month - int(month / 100) * 100, 28)
    else:
        print('WARNING: Input type ' + str(type(month)) + ' not handled.')


def sas_date_to_datetime(series, unit='s'):
    if isinstance(series, pd.Series):
        return pd.to_datetime('1960-01-01') + pd.to_timedelta(series, unit=unit)
    else:
        print('WARNING: Input type ' + str(type(series)) + ' not handled.')


def is_month_end(s_date):
    """ Return a Series with boolean value for whether the date is a month-end date.
    """
    s_month = s_date.apply(lambda x: x.month)
    s_month_end = ~s_month.eq(s_month.shift(-1))
    if not s_date.empty:
        d = s_date.iloc[-1]
        if (d != eom(d)) and (d != eobm(d)):
            s_month_end.iloc[-1] = False
        s_idx_chg = s_date.gt(s_date.shift(-1), fill_value=datetime(2100, 1, 1))
        if s_idx_chg.any():
            s_month_end[(s_date != eom(s_date)) & (s_date != eobm(s_date)) & s_idx_chg] = False
    return s_month_end


def mtd_to_idx(s_mtd_rtn):
    is_me = is_month_end(s_mtd_rtn.index.to_series())
    s_mtd_rtn_fac = 1 + s_mtd_rtn/100
    s_rtn_idx = pd.Series(index=s_mtd_rtn.index)
    s_rtn_idx[is_me] = s_mtd_rtn_fac[is_me].cumprod()
    s_rtn_idx = s_rtn_idx.ffill().fillna(1.)
    s_rtn_idx[~is_me] = s_rtn_idx[~is_me] * s_mtd_rtn_fac[~is_me]
    return s_rtn_idx


def idx_to_mtd(s_rtn_idx, starting_value=1.):
    is_me = is_month_end(s_rtn_idx.index.to_series())
    s_idx_pme = pd.Series(index=s_rtn_idx.index)
    s_idx_pme[is_me] = s_rtn_idx
    s_idx_pme = s_idx_pme.ffill().fillna(starting_value)
    s_idx_pme[is_me] = s_idx_pme.shift(1)
    return 100 * (s_rtn_idx / s_idx_pme - 1)


def idx_to_set(s_idx, index_set=None):
    """
    Convert a return index (cumulative return series, expressed as a factor) to 1-periods differences over the set of
    indices in index_set

    Parameters
    ----------
    s_idx : Series
        The return index series
    index_set : set, default: index of s_idx
        Set of indices corresponding to indices in s_idx

    Returns
    -------
    DataFrame
        A 1-period return series, expresed as a percentage (e.g. 1% = 1.0)
    """
    if index_set is None:
        index_set = set(s_idx.index)
    assert s_idx[index_set].isna().sum() == 0, "Missing values in input series."
    if min(index_set) == s_idx.index[0]:
        start_idx = 1
    else:
        start_idx = s_idx[s_idx.index < min(index_set)].iloc[-1]
    # s_idx_set = s_idx[map(lambda x: x in index_set, s_idx.index)]
    s_idx_set = s_idx[[i for i in s_idx.index if i in index_set]]
    return 100 * (s_idx_set / s_idx_set.shift(1).fillna(start_idx) - 1)


def get_holidays(country_code):
    """
    Returns a Series of holiday dates for a given country.

    Parameters
    ----------
    country_code : str
        2-letter ISO country code.

    Returns
    -------
    pd.Series
        Series of datetime holiday dates.
    """
    if country_code in DCT_HOLIDAYS:
        return DCT_HOLIDAYS[country_code]
    if 'all' not in DCT_HOLIDAYS:
        df_holidays = pd.read_csv(os.path.join(os.getenv('MKTDATDIR'), 'FactSet', 'FACTSET_MARKET_HOLIDAYS.csv'))
        df_holidays['date'] = pd.to_datetime(df_holidays['Date'], format='%Y%m%d')
        DCT_HOLIDAYS['all'] = df_holidays
    df_holidays = DCT_HOLIDAYS['all']
    s_holidays = df_holidays.loc[df_holidays['CountryCode'] == country_code, 'date']
    DCT_HOLIDAYS[country_code] = s_holidays
    return s_holidays
