"""
 Miscellaneous equity tools that are helpful, but don't fit neatly into any other category.
 Suggested import as equ
"""
import pandas as pd
import qr_shared_src.data_tools.db as db


def gics_scheme(asofdate=None, level=None):
    """
    Get the GICS scheme as of a particular date.

    Parameters
    ----------
    asofdate : str
        yyyymmdd formatted date for which data is to be retrieved
    level : int
        1 - 4 from sector, industry group, industry, and subindustry
        If None, all levels are retrieved

    Returns
    -------
    pd.DataFrame
        ind_code, level, description, parent_ind_code

    """
    if asofdate is None:
        asofdate = pd.to_datetime('today').strftime("%Y%m%d")

    lev_str = '' if level is None else f" and level = {level} "

    con = db.get_conn('ISIS')
    sql = f""" select industry_code as ind_code, level, description, parent_industry_code as parent_ind_code 
               from industry_orig 
               where industry_scheme_id = 'GICS' and '{asofdate}' between from_date and to_date {lev_str}"""
    df = pd.read_sql_query(sql, con)
    con.close()

    return df


def is_reit(gics):
    """
    Pass a pandas Series containing the 8-character GICS subindustry code (a column from a dataframe, typically)
    and return a Series bool identifying REITs

    Parameters
    ----------
    gics : pd.Series
        Typically DataFrame column with GICS subindustry code as string.

    Returns
    -------
    pd.Series
        bool indicating REIT status
    """

    return (gics.str[:6] == '404020') | (gics == '40401010') | (gics.str[0:6] == '601010') | (gics.str[:6] == '402040')
