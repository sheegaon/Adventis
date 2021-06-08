"""
 Miscellaneous fixed income tools that are helpful, but don't fit neatly into any other category.
 Suggested import as fi

"""
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime


def numerical_rating_idx(rating_str):
    """
    Convert Moody's, S&P, or Credit Suisse style letter rating to 'index-style' numerical rating on scale GOV = 1;
    AAA = 2 to D = 23; NR/WR = 24

    Parameters
    ----------
    rating_str : pd.Series
        Contains string ratings

    Returns
    -------
    pd.Series
        Contains numerical ratings
    """
    rating_str = rating_str.str.lower().str.replace('bbb', 'baa').str.replace('bb', 'ba').str.replace(
        'ccc', 'caa').str.replace('cc', 'ca').str.replace('+', '1', regex=False).str.replace(
        '-', '3', regex=False).str.replace('2', '', regex=False)
    dct_rtg = {'aaa': 2, 'aa1': 3, 'aa': 4, 'aa3': 5, 'a1': 6, 'a': 7, 'a3': 8, 'baa1': 9, 'baa': 10, 'baa3': 11,
               'ba1': 12, 'ba': 13, 'ba3': 14, 'split ba': 14.5, 'b1': 15, 'b': 16, 'b3': 17, 'split b': 17.5,
               'caa1': 18, 'caa': 19, 'caa3': 20, 'ca': 21, 'c': 22, 'd': 23, 'nr': 24, 'wr': 24}
    return rating_str.map(dct_rtg)


def baml_string_rating(rating_num):
    rating_str = ''
    if rating_num == 1:
        rating_str = 'GOV'
    elif rating_num == 2:
        rating_str = 'AAA'
    elif rating_num == 3:
        rating_str = 'AA1'
    elif rating_num == 4:
        rating_str = 'AA2'
    elif rating_num == 5:
        rating_str = 'AA3'
    elif rating_num == 6:
        rating_str = 'A1'
    elif rating_num == 7:
        rating_str = 'A2'
    elif rating_num == 8:
        rating_str = 'A3'
    elif rating_num == 9:
        rating_str = 'BBB1'
    elif rating_num == 10:
        rating_str = 'BBB2'
    elif rating_num == 11:
        rating_str = 'BBB3'
    elif rating_num == 12:
        rating_str = 'BB1'
    elif rating_num == 13:
        rating_str = 'BB2'
    elif rating_num == 14:
        rating_str = 'BB3'
    elif rating_num == 15:
        rating_str = 'B1'
    elif rating_num == 16:
        rating_str = 'B2'
    elif rating_num == 17:
        rating_str = 'B3'
    elif rating_num == 18:
        rating_str = 'CCC1'
    elif rating_num == 19:
        rating_str = 'CCC2'
    elif rating_num == 20:
        rating_str = 'CCC3'
    elif rating_num == 21:
        rating_str = 'CC'
    elif rating_num == 22:
        rating_str = 'C'
    elif rating_num == 23:
        rating_str = 'D'
    elif rating_num == 24:
        rating_str = 'NR'
    return rating_str


def moodys_string_rating(rating_num):
    rating_str = ''
    if rating_num == 1:
        rating_str = 'GOV'
    elif rating_num == 2:
        rating_str = 'AAA'
    elif rating_num == 3:
        rating_str = 'AA1'
    elif rating_num == 4:
        rating_str = 'AA2'
    elif rating_num == 5:
        rating_str = 'AA3'
    elif rating_num == 6:
        rating_str = 'A1'
    elif rating_num == 7:
        rating_str = 'A2'
    elif rating_num == 8:
        rating_str = 'A3'
    elif rating_num == 9:
        rating_str = 'BAA1'
    elif rating_num == 10:
        rating_str = 'BAA2'
    elif rating_num == 11:
        rating_str = 'BAA3'
    elif rating_num == 12:
        rating_str = 'BA1'
    elif rating_num == 13:
        rating_str = 'BA2'
    elif rating_num == 14:
        rating_str = 'BA3'
    elif rating_num == 15:
        rating_str = 'B1'
    elif rating_num == 16:
        rating_str = 'B2'
    elif rating_num == 17:
        rating_str = 'B3'
    elif rating_num == 18:
        rating_str = 'CAA1'
    elif rating_num == 19:
        rating_str = 'CAA2'
    elif rating_num == 20:
        rating_str = 'CAA3'
    elif rating_num == 21:
        rating_str = 'CA'
    elif rating_num == 22:
        rating_str = 'C'
    elif rating_num == 23:
        rating_str = 'D'
    elif rating_num == 24:
        rating_str = 'NR'
    return rating_str


def str_rating_flcl(numer_rating):
    """
    Using moody string rating function to get string rating from numerical rating
    :param numer_rating: float/list/pandas series
    :return: string or list of rating (floor/ceiling) corresponding to the input numerical rating
    """
    if isinstance(numer_rating, float):
        inp_rating = [numer_rating]
    else:
        inp_rating = numer_rating
    # Handles series/list
    str_rate_flr = [moodys_string_rating(int(x)) if not pd.isnull(x) else 'NA' for x in np.floor(inp_rating)]
    str_rate_clg = [moodys_string_rating(int(x)) if not pd.isnull(x) else 'NA' for x in np.ceil(inp_rating)]
    output_rating = [f"{x}/{y}" for x, y in zip(str_rate_flr, str_rate_clg)]
    if isinstance(numer_rating, float):
        output_rating = output_rating[0]

    return output_rating


def portfolio_rating(df_bonds, rating_col='rtg', simple_mean=False):
    """
    Returns portfolio rating for dataframe of bonds

    Parameters
    ----------
    df_bonds : pd.DataFrame
        Should contain weights (column='wt') and a column for numerical rating (arg rating_col)

    rating_col : str
        Column name to be used for numerical rating

    simple_mean : bool
        True: Arithmetic weight mean; False: rating = log(w*e^rating) as log of default probability is linear to rating

    Returns
    -------
    float
        Portfolio rating value
    """
    df_bonds.loc[(df_bonds[rating_col] == 24) | (df_bonds[rating_col] == -1), rating_col] = 16
    if simple_mean:
        port_rate = (df_bonds['wt'] * df_bonds[rating_col]).sum()
    else:
        port_rate = np.log((df_bonds['wt'] * np.exp(df_bonds[rating_col])).sum())
    return port_rate


def rtg_indicators(rtg_series):
    """
    Create rating indicators for IG, HY, and coarse letter rating groups.
    """
    # Using strict inequality as lcapx has rating of 14.5
    df_out = pd.DataFrame(index=rtg_series.index)
    df_out['hy'] = (rtg_series > 11) | rtg_series.isna()
    df_out['ig'] = ~df_out['hy']
    df_out['aaa'] = rtg_series <= 2
    df_out['aa'] = (rtg_series >= 3) & (rtg_series < 6)
    df_out['aa+'] = rtg_series < 6
    df_out['a'] = (rtg_series >= 6) & (rtg_series < 9)
    df_out['bbb'] = (rtg_series >= 9) & (rtg_series < 12)
    df_out['bb'] = (rtg_series >= 12) & (rtg_series < 15)
    df_out['b'] = (rtg_series >= 15) & (rtg_series < 18)
    df_out['ccc'] = (rtg_series >= 18) & (rtg_series < 21)
    df_out['ccc-'] = rtg_series >= 18
    df_out['cc-'] = rtg_series >= 21
    df_out['nr'] = rtg_series == 24
    return df_out


def get_frb_cmt_yld(extrapolate_y30=True):
    """
    Retrieve Constant Maturity Treasury yields from the Federal Reserve Board's H15 data.

    Note: Series M01 has an inception date of 2001-07-31. Series Y30 is missing from 2002-02-19 to 2006-02-08.
    This function optionally extrapolates the missing Y30 values from Y05, Y10, and Y20.

    Parameters
    ----------
    extrapolate_y30 : bool, default True
        Extrapolate values for Y30 from Y05, Y10, and Y20.

    Returns
    -------
    DataFrame
        Date and maturities from M01 to Y30
    """
    import urllib
    u = urllib.request.urlopen(
        r"https://www.federalreserve.gov/datadownload/Output.aspx?rel=H15&series=bf17364827e38702b42a58cf8eaa3f78"
        r"&lastobs=&from=&to=&filetype=csv&label=include&layout=seriescolumn&type=package")
    try:
        df_frb_cmt = pd.read_csv(u)
    except pd.errors.ParserError:
        logging.warning("Federal Reserve Board Constant Maturity Treasury data download is temporarily unavailable.")
        return None
    df_frb_cmt.columns = ['date'] + [x[15:18] for x in df_frb_cmt.iloc[3, 1:].values.tolist()]
    df_frb_cmt = df_frb_cmt.iloc[5:, :].copy()
    for n in range(1, df_frb_cmt.shape[1]):
        df_frb_cmt.iloc[:, n] = pd.to_numeric(df_frb_cmt.iloc[:, n], errors='coerce')
    df_frb_cmt['date'] = pd.to_datetime(df_frb_cmt['date'])
    df_frb_cmt.dropna(subset=['Y01', 'Y05', 'Y10'], how='all', inplace=True)

    if extrapolate_y30:
        # Coefficients are from an OLS regression using all data since 1995-01-01, (almost) rounded to nearest 0.01
        df_frb_cmt.loc[df_frb_cmt['Y30'].isna(), 'Y30'] = (
                0.93 * df_frb_cmt['Y10'] + 0.82 * df_frb_cmt['Y20'] - 0.75 * df_frb_cmt['Y07'])
        # Coefficients are from an OLS regression using all data since 1995-01-01, rounded to nearest 0.1
        df_frb_cmt.loc[df_frb_cmt['Y30'].isna(), 'Y30'] = (
                0.9 * df_frb_cmt['Y10'] + 0.6 * df_frb_cmt['Y20'] - 0.5 * df_frb_cmt['Y05'])

    return df_frb_cmt
