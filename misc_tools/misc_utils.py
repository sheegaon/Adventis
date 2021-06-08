"""
 $Id: //QR_ProductDesign/main/QRPDSrc/misc_tools/pymod/misc_utils.py#174 $
 $Change: 94696 $
 $DateTime: 2021/04/23 13:58:10 $
 $Author: tfishman $

 Miscellaneous tools that are helpful, but don't fit neatly into any other category.
 Suggested import as mu

"""
import pandas as pd
from qr_shared_src.misc_tools.misc_utils import *


def calc_return_statistics(
        df_monthly, ret_col_list=None, tenor_list=None, diff_list=None, rf_rate=None, annualize_factor=12,
        historical_scenarios=None, sample_periods=None, use_available=False, skip_summary=False):
    r"""
    Returns 2 DataFrames: (1) summary statistics and (2) a monthly DF with index matching df_monthly containing
    monthly and cumulative returns, rolling relative returns, volatilities, and information ratios.

    Differences are calculated by the index list in diff_list.

    Parameters
    ----------
    df_monthly : pd.DataFrame
        A monthly data frame containing columns specified in ret_col_list.
    ret_col_list : [str]
        A list of column names with returns for the design and benchmarks. Default all columns.
    tenor_list : [int]
        A list of numbers of months over which cumulative rolling returns are calculated.
    diff_list : typing.List
        A list of pairs of labels of ret_col_list
    rf_rate : str
        Column name containing risk-free rate expressed in monthly total return terms.
    annualize_factor : numeric
        Annualizing factor. Defaults to 12 if using monthly data
    historical_scenarios : [tuple]
        TODO Refactor all function calls to move this bit of code out of here and use
         RiskModel\proj_func.py:calc_scenario_rtn().
        List of historical scenarios over which to calculate cumulative returns. Each secenario
        consists of a tuple of (label, start date, end date). Optional fourth tuple argument of 'return per duration',
        return of BBG TSY index divided by index duration over that period.
    sample_periods : dict
        Sample period labels and period start/end dates (inclusive). If only one date is provided, it is presumed
        to be the start date and the last available date is taken as the end date.
    use_available : bool, default False
        Use all available data for sample periods with missing data.
    skip_summary : bool, default False
        Skip calculation of summary statistics (for speed).

    Returns
    -------
    [pd.DataFrame, pd.DataFrame]
        Summary statistics by ret_col_list, monthly returns and return-related statistics.
    """
    import warnings
    def cumret(x): return (1+x).prod()-1

    if 'date' in df_monthly.columns:
        df_mon = df_monthly.copy()
    else:
        df_mon = df_monthly.reset_index()

    if ret_col_list is None:
        ret_col_list = list(set(df_mon.columns) - {'date'})

    dt0 = df_mon['date'].min()
    df_mon.insert(1, 'period_yrs', (df_mon['date'] - dt0).astype('timedelta64[D]')/365)

    if rf_rate is None:
        df_mon['rf'] = 0
        rf_rate = 'rf'

    mon_cols = ['date', 'period_yrs'] + list(set(ret_col_list + [rf_rate]))
    if historical_scenarios is None:
        historical_scenarios = []
    else:
        warnings.warn('Use of historical_scenarios argument in misc_utils.py:calc_return_statistics() is superceded by '
                      r'tfi\RiskModel\proj_func.py:calc_scenario_rtn()', category=DeprecationWarning)
        for rc in ret_col_list:
            sjv = rc.replace('tot_ret_', '')
            if f'eff_dur_{sjv}' in df_mon.columns:
                mon_cols += [f'eff_dur_{sjv}']

    if tenor_list is None:
        tenor_list = [12, 36, 60]

    df_mon[list(set(ret_col_list + [rf_rate]))] /= 100
    df_mon = df_mon[mon_cols].copy()
    df_mon['cum_rf'] = 100*((1 + df_mon[rf_rate]).cumprod() - 1)

    df_summary = pd.DataFrame(data=ret_col_list, columns=['ret_col'])
    yrs = (df_mon.index.size - df_mon.loc[df_mon[rf_rate].notnull(), rf_rate].index[0]) / annualize_factor
    df_summary['cum_ann_rf_full_sample'] = 100*((1 + df_mon['cum_rf'].iloc[-1]/100)**(1/yrs) - 1)
    if sample_periods is None:
        sample_periods = {'post_crisis': datetime(2010, 1, 31)}
    sample_label = ['full_sample'] + list(sample_periods.keys())
    for rc in ret_col_list:
        assert df_mon[rc].notnull().any(), f"Error in calc_return_statistics: Returns are all NaN for {rc}."
        df_mon[f'cum_{rc}'] = 100*((1 + df_mon[rc]).cumprod() - 1)

        # cumulative arithmetic return (cumar)
        df_mon[f'cumar_{rc}'] = 100*df_mon[rc].cumsum()

        # CAGR (compound annual growth rate)
        df_mon[f'cagr_{rc}'] = 100 * ((1 + df_mon[f'cum_{rc}']/100) ** (1/df_mon['period_yrs']) - 1)
        # CAGR can be very misleading for periods less than a year, so ignore and set to missing
        df_mon.loc[df_mon['period_yrs'] < 1, f'cagr_{rc}'] = np.nan

        # Rolling window statistics
        for tnr in tenor_list:
            df_mon[f'cum_{rc}_{tnr}m'] = (
                    100 * df_mon[rc].rolling(center=False, window=tnr).apply(func=cumret, raw=True))
            df_mon[f'cum_rf_{tnr}m'] = (
                    100 * df_mon[rf_rate].rolling(center=False, window=tnr).apply(func=cumret, raw=True))

            yrs = tnr / annualize_factor
            df_mon[f'cum_ann_{rc}_{tnr}m'] = 100 * ((1 + df_mon[f'cum_{rc}_{tnr}m']/100)**(1/yrs) - 1)
            df_mon[f'cum_ann_exc_{rc}_{tnr}m'] = 100 * (
                    (1 + (df_mon[f'cum_{rc}_{tnr}m'] - df_mon[f'cum_rf_{tnr}m'])/100)**(1/yrs) - 1)

            df_mon[f'vol_{rc}_{tnr}m'] = 100 * np.sqrt(annualize_factor) * (
                    df_mon[rc].rolling(center=False, window=tnr).std())
            df_mon[f'ervol_{rc}_{tnr}m'] = 100 * np.sqrt(annualize_factor) * (
                    (df_mon[rc] - df_mon[rf_rate]).rolling(center=False, window=tnr).std())
            df_mon[f'sr_{rc}_{tnr}m'] = (
                    df_mon[f'cum_ann_exc_{rc}_{tnr}m'] / df_mon[f'ervol_{rc}_{tnr}m'])

        if skip_summary:
            continue
        # Returns and empirical haircut adjustment for historical scenarios
        for scen in historical_scenarios:
            if sum(df_mon['date'] == scen[1]) == 0:
                logging.warning(f'Missing date {scen[1]:%Y-%m-%d} in df_mon, skipping scenario {scen[0]}.')
                continue
            if sum(df_mon['date'] == scen[2]) == 0:
                logging.warning(f'Missing date {scen[2]:%Y-%m-%d} in df_mon, skipping scenario {scen[0]}.')
                continue
            cum_start = df_mon.loc[df_mon['date'] == scen[1], f'cum_{rc}'].iloc[0]
            cum_end = df_mon.loc[df_mon['date'] == scen[2], f'cum_{rc}'].iloc[0]
            cum_ret = 100 * ((1 + cum_end / 100.) / (1 + cum_start / 100.) - 1)
            df_summary.loc[df_summary['ret_col'] == rc, f'cum_{scen[0]}'] = cum_ret
            sjv = rc.replace('tot_ret_', '')
            if f'eff_dur_{sjv}' in df_mon.columns:
                eff_dur = df_mon.loc[df_mon['date'] == scen[1], f'eff_dur_{sjv}'].iloc[0]
                ret_per_dur = cum_ret / eff_dur
                if len(scen) > 3:
                    emp_hc = ret_per_dur / scen[3]
                    df_summary.loc[df_summary['ret_col'] == rc, f'emp_hc_{scen[0]}'] = emp_hc

        # Return statistics by historical sample
        sample_i = [df_mon.loc[df_mon[rc].notnull(), rc].index[0]]
        sample_j = [df_mon.loc[df_mon[rc].notnull(), rc].index[-1]]
        sample_hasmissing = [False]
        for dt in sample_periods.values():
            if isinstance(dt, tuple):
                dt, dt1 = dt
            else:
                dt1 = df_mon['date'].max()
            if (df_mon[rc].notnull() & (df_mon['date'] >= dt) & (df_mon['date'] <= dt1)).any():
                i = df_mon.loc[df_mon[rc].notnull() & (df_mon['date'] >= dt) & (df_mon['date'] <= dt1), rc].index[0]
                j = df_mon.loc[df_mon[rc].notnull() & (df_mon['date'] >= dt) & (df_mon['date'] <= dt1), rc].index[-1]
            else:
                i = df_mon.index[0]
                j = df_mon.index[-1]
            sample_i.append(i)
            sample_j.append(j)
            if (df_mon.loc[i, 'date'] - dt).days > 35 or (df_mon.loc[j, 'date'] - dt1).days > 35:
                sample_hasmissing.append(True)
            else:
                sample_hasmissing.append(False)

        warnings.filterwarnings(action='ignore', category=RuntimeWarning)
        for i, j, sample, hasmiss in zip(sample_i, sample_j, sample_label, sample_hasmissing):
            if (sample_label == 'full_sample') | (i == df_mon.index[0]):
                cum_start = 0
                cum_start_rf = 0
            else:
                cum_start = df_mon[f'cum_{rc}'].fillna(0).loc[df_mon.loc[:i].index[-2]]
                cum_start_rf = df_mon[f'cum_rf'].fillna(0).loc[df_mon.loc[:i].index[-2]]
            yrs = df_mon.loc[i:j].index.size / annualize_factor
            df_ret_sample = 100*((1 + 0.01*df_mon[f'cum_{rc}'].loc[j])/(1 + 0.01*cum_start) - 1)
            df_ret_rf_sample = 100*((1 + 0.01*df_mon['cum_rf'].loc[j])/(1 + 0.01*cum_start_rf) - 1)
            df_summary.loc[df_summary['ret_col'] == rc, f'cum_{sample}'] = df_ret_sample
            df_summary.loc[df_summary['ret_col'] == rc, f'cum_ann_{sample}'] = 100 * (
                    (1 + 0.01*df_ret_sample)**(1/yrs) - 1)
            df_summary.loc[df_summary['ret_col'] == rc, f'cum_ann_exc_{sample}'] = 100 * (
                    (1 + 0.01*(df_ret_sample - df_ret_rf_sample))**(1/yrs) - 1)
            df_summary.loc[df_summary['ret_col'] == rc, f'vol_{sample}'] = np.sqrt(annualize_factor) * 100 * (
                    df_mon.loc[i:j, rc].std())
            df_summary.loc[df_summary['ret_col'] == rc, f'ervol_{sample}'] = np.sqrt(annualize_factor) * 100 * (
                    (df_mon.loc[i:j, rc] - df_mon.loc[i:j, rf_rate]).std())
            df_summary.loc[df_summary['ret_col'] == rc, f'var99_{sample}'] = 100 * df_mon.loc[i:, rc].quantile(.01)

            # Drawdown statistics
            df_mon[f'running_max_{rc}'] = np.maximum.accumulate(df_mon.loc[i:j, f'cum_{rc}'])
            df_mon[f'drawdown_{rc}'] = -100.0*((1 + df_mon.loc[i:j, f'cum_{rc}']/100) /
                                               (1 + df_mon.loc[i:j, f'running_max_{rc}']/100) - 1)
            df_summary.loc[df_summary['ret_col'] == rc, f'dd_max_{sample}'] = df_mon[f'drawdown_{rc}'].max()
            df_summary.loc[df_summary['ret_col'] == rc, f'dd_99_{sample}'] = df_mon[f'drawdown_{rc}'].quantile(.99)
            dd_end = ((1+df_mon.loc[i:j, f'running_max_{rc}']/100)/(1+df_mon.loc[i:j, f'cum_{rc}']/100) - 1).idxmax()
            dd_start = (df_mon.loc[:dd_end, f'cum_{rc}']).idxmax()
            df_summary.loc[df_summary['ret_col'] == rc, f'dd_max_t_{sample}'] = dd_end - dd_start
            recovered = df_mon[f'cum_{rc}'] > df_mon.loc[dd_start, f'cum_{rc}']
            if sum(recovered) > 0:
                df_summary.loc[df_summary['ret_col'] == rc,
                               f'dd_t_recover_{sample}'] = df_mon[recovered & (df_mon.index > dd_end)].index[0] - dd_end
            else:
                df_summary.loc[df_summary['ret_col'] == rc, f'dd_t_recover_{sample}'] = np.nan
            for tnr in tenor_list:
                worst_tnr = np.nan
                num_neg_tnr = np.nan
                if df_mon.loc[i:j].shape[0] >= tnr:
                    worst_tnr = df_mon.loc[df_mon.loc[i:j].index[tnr - 1]:, f'cum_{rc}_{tnr}m'].min()
                    num_neg_tnr = sum(df_mon.loc[df_mon.loc[i:j].index[tnr - 1]:, f'cum_{rc}_{tnr}m'] < 0)
                df_summary.loc[df_summary['ret_col'] == rc, f'worst_{tnr}m_{sample}'] = worst_tnr
                df_summary.loc[df_summary['ret_col'] == rc, f'num_neg_{tnr}m_{sample}'] = num_neg_tnr
            if hasmiss and (not use_available):
                sample_cols = [c for c in df_summary.columns if sample in c]
                df_summary.loc[df_summary['ret_col'] == rc, sample_cols] = np.nan
        warnings.filterwarnings(action='default', category=RuntimeWarning)

    if not skip_summary:
        for sample in sample_label:
            df_summary[f'sr_{sample}'] = df_summary[f'cum_ann_exc_{sample}'] / df_summary[f'vol_{sample}']

    # Statistics for active (relative) returns as specified in diff_list
    if diff_list is None:
        diff_list = []
    for diff_set in diff_list:
        c1 = diff_set[0]
        c2 = diff_set[1]
        c1vc2 = f'{c1}_v_{c2}'

        # Cumulative returns over various periods
        df_mon[f'cum_{c1vc2}_full_period'] = df_mon[f'cum_{c1}'] - df_mon[f'cum_{c2}']
        for tnr in tenor_list:
            df_mon[f'cum_{c1vc2}_{tnr}m'] = df_mon[f'cum_{c1}_{tnr}m'] - df_mon[f'cum_{c2}_{tnr}m']
            df_mon[f'cum_ann_{c1vc2}_{tnr}m'] = df_mon[f'cum_ann_{c1}_{tnr}m'] - df_mon[f'cum_ann_{c2}_{tnr}m']

        # Full period and rolling average active (i.e. relative) returns,
        # volatility of active returns (a.k.a. tracking error), and information ratios
        df_mon[c1vc2] = df_mon[c1] - df_mon[c2]
        numer = f'avg_ann_{c1vc2}'
        df_mon[numer] = annualize_factor*100*df_mon[c1vc2].mean()
        denom = f'std_ann_{c1vc2}'
        df_mon[denom] = np.sqrt(annualize_factor)*100*df_mon[c1vc2].std()
        df_mon[f'ir_ann_{c1vc2}'] = df_mon[numer]/df_mon[denom]
        for tnr in tenor_list:
            numer = f'avg_ann_{c1vc2}_{tnr}m'
            df_mon[numer] = annualize_factor*100*df_mon[c1vc2].rolling(center=False, window=tnr).mean()
            denom = f'std_ann_{c1vc2}_{tnr}m'
            df_mon[denom] = np.sqrt(annualize_factor)*100*df_mon[c1vc2].rolling(center=False, window=tnr).std()
            df_mon[f'ir_ann_{c1vc2}_{tnr}m'] = df_mon[numer]/df_mon[denom]
        df_mon[c1vc2] = 100 * df_mon[c1vc2]

    df_mon[list(set(ret_col_list + [rf_rate]))] *= 100
    return df_summary, df_mon


def gen_peer_stats(s_peer, tenor_list=None):
    """
    Generate cumulative returns from a Series of peer returns indexed by date and an id variable.

    Parameters
    ----------
    s_peer : pd.Series
        Peer group returns indexed by a peer id variable and a date variable.
    tenor_list : [int], default [12, 36, 60]
        A list of numbers of months over which cumulative rolling returns are calculated.

    Returns
    -------
    pd.DataFrame
    """
    from pandas.api.types import is_datetime64_any_dtype

    if tenor_list is None:
        tenor_list = [12, 36, 60]

    # Assign column labels to variable names
    df_peer = s_peer.reset_index()
    idx1, idx2 = s_peer.index.names
    rtn_var = s_peer.name
    if is_datetime64_any_dtype(df_peer[idx1]):
        dt_var = idx1
        id_var = idx2
    elif is_datetime64_any_dtype(df_peer[idx2]):
        dt_var = idx2
        id_var = idx1
    else:
        raise ValueError(f"Neither {idx1} nor {idx2} are datetime64 objects.")

    df_peer = df_peer.drop_duplicates([id_var, dt_var]).reset_index(drop=True)
    df_peer_long = df_peer.pivot(index=dt_var, columns=id_var, values=rtn_var).reset_index().dropna(axis=1, how='all')
    peer_cols = df_peer_long.columns[1:].tolist()
    df_stat = calc_return_statistics(df_peer_long, ret_col_list=peer_cols, tenor_list=tenor_list, skip_summary=True)[1]
    df_stat_c = concat_wo_dup_col(df_peer_long, df_stat)
    peer_roll_list = [f'cum_ann_{pcl}_{tnr}m' for pcl in peer_cols for tnr in tenor_list]
    peer_roll_list += [f'cum_{pcl}_{tnr}m' for pcl in peer_cols for tnr in tenor_list]
    return df_stat_c[[dt_var] + peer_roll_list], peer_cols


def gen_peer_ranks(s_peer, df_rtn, tenor, tenor_freq='M', peer_criteria=None):
    """
    Generate rolling return ranks relative to a Series of peer group returns.

    Parameters
    ----------
    s_peer : pd.Series
        Peer group returns indexed by a peer id variable and a date variable.
    df_rtn : pd.DataFrame
        Return variables to generate ranks, indexed by date.
    tenor : int
        Number of periods (months) over which rolling ranks are calculated.
    tenor_freq : {'M', 'D', 'Y'}, default 'M'
        Frequency of tenor, default months.
    peer_criteria : pd.Series(dtype=bool)
        Optionally limit the peer group with a bool with index matching s_peer.

    Returns
    -------
    pd.DataFrame
        Return ranks indexed by date.
    """
    df_peer_cum, peer_cols = gen_peer_stats(s_peer, tenor_list=[tenor])
    df_peer_cum.set_index('date', inplace=True)
    df_peer_cum.dropna(how='all', inplace=True)

    prefix = 'cum_'
    tenor_str = f'{tenor}{tenor_freq.lower()}'
    if f"{prefix}{peer_cols[0]}_{tenor_str}" not in df_peer_cum.columns:
        prefix = 'cum_ann_'

    rtn_var_list = list(df_rtn.columns)
    df_rtn.dropna(how='all', inplace=True)
    df_rtn_cum = calc_return_statistics(df_rtn, tenor_list=[tenor], skip_summary=True)[1]
    rtn_roll_list = [f'{prefix}{rtn_var}_{tenor_str}' for rtn_var in rtn_var_list]
    df_rtn_cum.dropna(subset=rtn_roll_list, how='all', inplace=True)
    df_rtn_cum.set_index('date', inplace=True)

    # Apply (id_var, dt)-based criteria to limit peer group
    if peer_criteria is not None:
        for dt in df_peer_cum.index:
            for id_var in peer_cols:
                if (id_var, dt) in peer_criteria.index and (not peer_criteria[id_var, dt]):
                    df_peer_cum.loc[dt, f'{prefix}{id_var}_{tenor_str}'] = np.nan

    lst_100 = [.001] + list(np.arange(0.01, 1, .01)) + [.999]
    peer_roll_list = [f'{prefix}{id_var}_{tenor_str}' for id_var in peer_cols if
                      f'{prefix}{id_var}_{tenor_str}' in df_peer_cum.columns]
    lst_100_str = [f"{q:.3f}" for q in lst_100]
    df_qntl = df_peer_cum[peer_roll_list].quantile(lst_100, axis=1).transpose().dropna()
    df_qntl.columns = lst_100_str
    for v in lst_100_str:
        df_rtn_cum[v] = df_qntl[v]

    df_ranks = pd.DataFrame(index=df_rtn_cum.index)
    for i_rv, rtn_var in enumerate(rtn_var_list):
        df_ranks[f'rank_{rtn_var}'] = np.nan
        for j, row in df_rtn_cum.iterrows():
            if np.isnan(row[rtn_roll_list[i_rv]]):
                continue
            qi = pd.to_numeric(row[lst_100_str])
            df_ranks.loc[j, f'rank_{rtn_var}'] = 100 - 100 * np.interp(row[rtn_roll_list[i_rv]], qi.values, lst_100)

    return df_ranks


def set_display_options_for_pandas(max_columns=20, width=200):
    """
    Convenience function for setting display options for pandas, useful viewing dataframes while debugging. The
    default values for max_columns and width are not ideal for that purpose.

    Parameters
    ----------
    max_columns : int
    width : int

    Returns
    -------
    None

    """
    pd.options.display.max_columns = max_columns
    pd.options.display.width = width

    return
