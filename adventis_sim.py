import sys
import os
import logging
import random
import string
from datetime import timedelta, datetime
from time import perf_counter
import pickle

import pandas as pd
import numpy as np

import QRPDSrc.misc_tools.pymod.misc_utils as mu

# LST_TYPES = ['hurricane', 'tornado', 'earthquake', 'flood', 'drought']
LST_TYPES = ['hurricane', 'tornado']
LST_VINTAGES = [datetime(2020, 3, 31), datetime(2020, 6, 30), datetime(2020, 9, 30), datetime(2020, 12, 31),
                datetime(2021, 3, 31)]
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2021, 3, 31)
NUM_EPP = 110
MPP_SEED_FUND = 10000
DCT_STATE = dict()
PROJ_FOLDER = os.path.join(os.getenv('QRSHARE'),  "ProductDesign", "TFI", "OneOff", "data")


def main(argv):
    t0 = perf_counter()
    input_args = mu.arg_handler(argv, input_args=dict(
        nu=0.005, ec_power=2., mc_power=1.5, ec_premium_mult=1.1, mc_premium_mult=1.1, ec_half_life=14, mc_half_life=7,
        adv_fee_pct=0.0001, excess_loss_credit=0.5, event_mpp_fee=.1,
        recapitalization_threshold=.9),
                                name='OneOff/adventis_sim.py')
    mu.setup_logger(log_folder=os.path.join(os.getenv('QRSHARE'),  "ProductDesign", "TFI", "OneOff", "logs"),
                    filename=os.path.basename(__file__)[:-3], console_level=logging.INFO)

    logging.debug(f"Running with input_args={input_args}")

    # Potential extension: model a malicious actor who deliberately sells cheap insurance to himself

    all_dates = pd.date_range(start=START_DATE, end=END_DATE)
    init_epp_capital = 1e6
    init_mpp_capital = 1e7
    DCT_STATE['epp'] = pd.concat(
        flatten([[pd.DataFrame(
            {'bias': random.normalvariate(0, .5), 'noise': .05 * random.betavariate(2, 2), 'vintage': v,
             'available': init_epp_capital, 'staked': 0, 'active': True},
            index=[f'0xv{LST_VINTAGES.index(v)}e{i}']) for i in np.arange(NUM_EPP)] for v in LST_VINTAGES]))
    DCT_STATE['adv_price'] = pd.Series(
        index=all_dates, data=brownian(start_price=1., size=len(all_dates), bmu=0.001, sigma=.005))
    DCT_STATE['pregen_rand'] = np.random.rand(1000000)

    while os.path.exists(os.path.join(PROJ_FOLDER, "dct_state.pickle")):
        with open(os.path.join(PROJ_FOLDER, "dct_state.pickle"), 'rb') as pkl_handle:
            dct_state = pickle.load(pkl_handle)
        if dct_state['epp'].shape != DCT_STATE['epp'].shape:
            break
        for a in DCT_STATE.keys():
            DCT_STATE[a] = dct_state[a].copy()
        DCT_STATE['pregen_aes'] = dct_state['aes'].copy()
        DCT_STATE['epp'] = DCT_STATE['epp'].head(NUM_EPP * len(LST_VINTAGES))
        DCT_STATE['epp']['available'] = init_epp_capital
        DCT_STATE['epp']['staked'] = 0
        DCT_STATE['epp']['active'] = True
        break

    DCT_STATE['rand'] = DCT_STATE['pregen_rand'].copy()
    DCT_STATE['mpp'] = {'tokens': MPP_SEED_FUND, 'available': init_mpp_capital, 'staked': MPP_SEED_FUND}
    DCT_STATE['mpp_vintage_size'] = {v: 0 for v in LST_VINTAGES}
    DCT_STATE['mezz_coll'] = pd.Series(dtype=float)
    DCT_STATE['recap_tokens'] = 0
    DCT_STATE['n_adv_tokens'] = 1e7
    DCT_STATE['cashflow'] = pd.DataFrame(
        index=[0], data={'amount': -MPP_SEED_FUND, 'date': START_DATE, '0x': 'mpp', 'id': 0})
    for a in input_args.keys():
        DCT_STATE[a] = input_args[a]
    DCT_STATE['ec_decay'] = - np.log(2) / DCT_STATE['ec_half_life']
    DCT_STATE['mc_decay'] = - np.log(2) / DCT_STATE['mc_half_life']
    s_adv_avg_price = pd.concat(
        [pd.Series(data=[1.] * 30), DCT_STATE['adv_price']]).rolling(window=30).mean().tail(len(all_dates))
    DCT_STATE['adv_fee_tokens'] = pd.Series(data=np.nan, index=all_dates)
    sat_night = DCT_STATE['adv_fee_tokens'].index.weekday == 5
    DCT_STATE['adv_fee_tokens'].loc[sat_night] = input_args['adv_fee_pct'] / s_adv_avg_price
    DCT_STATE['adv_fee_tokens'] = DCT_STATE['adv_fee_tokens'].ffill().fillna(input_args['adv_fee_pct'])
    DCT_STATE['aes'] = pd.DataFrame()
    s_mpp_token_value = pd.Series(dtype=float)

    # ========================================
    # Run through simulation one day at a time
    # ========================================

    for dt in all_dates:
        logging.debug(f"Running simulation for date {dt:%Y-%m-%d}")
        for v in LST_VINTAGES:
            if dt >= v:
                continue
            for et in LST_TYPES:
                mint_aes(dt=dt, vintage=v, event_type=et)
        execute_events(dt)
        if dt.day == 1:
            DCT_STATE['mezz_coll'] *= np.nan

        # Wrap up pools on the expiration date of the contracts
        if dt in LST_VINTAGES:
            logging.info(f"End of vintage {dt:%Y-%m-%d}.")
            df_epp = DCT_STATE['epp'][DCT_STATE['epp']['vintage'] == dt].copy()
            for epp0x in df_epp.index:
                cfi = DCT_STATE['cashflow'].shape[0]
                DCT_STATE['cashflow'].loc[cfi, 'date'] = dt
                DCT_STATE['cashflow'].loc[cfi, '0x'] = epp0x
                DCT_STATE['cashflow'].loc[cfi, 'id'] = f'end of vintage {dt}'
                DCT_STATE['cashflow'].loc[cfi, 'amount'] = df_epp.loc[epp0x, 'staked']
            cfi = DCT_STATE['cashflow'].shape[0]
            DCT_STATE['cashflow'].loc[cfi, 'date'] = dt
            DCT_STATE['cashflow'].loc[cfi, '0x'] = 'mpp'
            DCT_STATE['cashflow'].loc[cfi, 'id'] = f'end of vintage {dt}'
            DCT_STATE['cashflow'].loc[cfi, 'amount'] = DCT_STATE['mpp_vintage_size'][dt]
            DCT_STATE['mpp']['tokens'] -= DCT_STATE['mpp_vintage_size'][dt] / mpp_token_value()
            DCT_STATE['mpp']['available'] += DCT_STATE['mpp_vintage_size'][dt]
            DCT_STATE['mpp']['staked'] -= DCT_STATE['mpp_vintage_size'][dt]
        s_mpp_token_value[dt] = mpp_token_value()
    DCT_STATE['s_mpp_token_value'] = s_mpp_token_value

    # Final cash flow from what's left in MPP
    cfi = DCT_STATE['cashflow'].shape[0]
    DCT_STATE['cashflow'].loc[cfi, 'date'] = END_DATE
    DCT_STATE['cashflow'].loc[cfi, '0x'] = 'mpp'
    DCT_STATE['cashflow'].loc[cfi, 'id'] = 'end of simulation'
    DCT_STATE['cashflow'].loc[cfi, 'amount'] = DCT_STATE['mpp']['staked']

    with open(os.path.join(PROJ_FOLDER, "dct_state.pickle"), 'wb') as pkl_handle:
        pickle.dump(DCT_STATE, pkl_handle)

    df_mpp = dct_state['s_mpp_token_value'].reset_index(name='value')
    df_mpp['eom'] = mu.eom(df_mpp['index'])
    df_mpp.drop_duplicates(keep='last', subset=['eom'])

    df_cf = DCT_STATE['cashflow']
    for x in df_cf['0x'].unique():
        pnl = df_cf.loc[df_cf['0x'] == x, 'amount'].sum()
        inv = - df_cf.loc[(df_cf['0x'] == x) & (df_cf['amount'] < 0), 'amount'].sum()
        if inv == 0:
            continue
        msg = f"Returned {100 * pnl / inv:.2f}%, ${pnl:.0f} P&L on net invested capital ${inv:.0f} for {x}"
        if x == 'mpp':
            logging.info(msg)
        else:
            logging.debug(msg)
    epp_pnl = df_cf.loc[df_cf['0x'] != 'mpp', 'amount'].sum()
    epp_inv = -df_cf.loc[(df_cf['0x'] != 'mpp') & (df_cf['amount'] < 0), 'amount'].sum()
    logging.info(f"EPPs returned {100 * epp_pnl / epp_inv:.2f}%, ${epp_pnl:.0f} P&L on ${epp_inv:.0f}")
    logging.info(f"EPP wallets abandoned: {(~DCT_STATE['epp']['active']).sum()}")
    logging.info(f"ADV tokens burned: {1e7 - DCT_STATE['n_adv_tokens']:.0f}")

    logging.info(f"Done. Time to run: {timedelta(seconds=perf_counter() - t0)}")
    return

# ==================
# Functions for main
# ==================


def mint_aes(dt, vintage=None, event_type=None, notional=None, epp0x=None):
    lst_v = [v for v in LST_VINTAGES if v > dt]
    if len(lst_v) == 0:
        return None
    if vintage is None:
        vintage = random.choice(lst_v)
    if event_type is None:
        event_type = random.choice(LST_TYPES)
    df_epp = DCT_STATE['epp'][(DCT_STATE['epp']['vintage'] == vintage) & DCT_STATE['epp']['active']]
    if epp0x is None:
        epp0x = random.choice(df_epp.index)

    if ('pregen_aes' in DCT_STATE.keys()) and (DCT_STATE['aes'].shape[0] < DCT_STATE['pregen_aes'].shape[0]):
        dct_aes = DCT_STATE['pregen_aes'].iloc[DCT_STATE['aes'].shape[0], :].to_dict()
        id16 = dct_aes['id']
        hazard = dct_aes['hazard']
        notional = dct_aes['notional']
    else:
        id16 = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
        while not DCT_STATE['aes'].empty and id16 in DCT_STATE['aes']['id'].tolist():
            id16 = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
        hazard = 0.001 * random.random()
        if notional is None:
            notional = random.randint(100, 10000)
    p_def_dt = 1 - np.exp(-hazard)
    days_to_expiry = (vintage - dt).days
    p_def_expiry = 1 - np.exp(-hazard * days_to_expiry)
    premium_pct = p_def_expiry * np.exp(random.normalvariate(df_epp.loc[epp0x, 'bias'], df_epp.loc[epp0x, 'noise']))
    dct_aes = {'id': id16, 'type': event_type, 'date_created': dt, 'vintage': vintage,
               'active': True, 'excess_loss': np.nan,
               'notional': notional, 'hazard': hazard, 'p_def': p_def_expiry, 'p_def_dt': p_def_dt,
               'premium_pct': premium_pct, 'equity_pool': epp0x}

    # Calculate collateral requirements
    df_aes = DCT_STATE['aes'].copy()
    for v in ['notional', 'equity_pool', 'active', 'excess_loss', 'date_created', 'premium_pct', 'id', 'vintage']:
        df_aes.loc[id16, v] = dct_aes[v]
    eq_coll = req_eq_coll(epp0x, df_aes, dt)
    mezz_coll = req_mezz_coll(df_aes, dt, new_eq_coll=pd.Series(index=[epp0x], data=[eq_coll]))
    addl_mezz_coll = mezz_coll - DCT_STATE['mpp']['staked']

    # Additional equity collateral required net of premium
    premium_amt = premium_pct * notional
    addl_eq_coll = eq_coll - df_epp.loc[epp0x, 'staked'] - premium_amt

    # Check EPP has sufficient available funds
    if df_epp.loc[epp0x, 'available'] < addl_eq_coll:
        logging.warning(f"EPP {epp0x} has insufficient free capital to execute a new AES")
        return mint_aes(dt, vintage, event_type, notional)

    # If additional collateral is greater than collateral for a new wallet, abandon this one
    if addl_eq_coll - premium_amt > notional:
        if (dt - DCT_STATE['aes'].loc[DCT_STATE['aes']['equity_pool'] == epp0x, 'date_created'].max()).days < 30:
            return mint_aes(dt, vintage, event_type, notional)
        # Add new EPP wallet with same parameters
        new_epp = DCT_STATE['epp'].loc[[epp0x]].copy()
        new_epp['staked'] = 0.
        new_epp.index = [f"0xv{LST_VINTAGES.index(vintage)}e"
                         f"{DCT_STATE['epp'][DCT_STATE['epp']['vintage'] == vintage].shape[0]}"]
        DCT_STATE['epp'] = pd.concat([DCT_STATE['epp'], new_epp])
        DCT_STATE['epp'].loc[epp0x, 'active'] = False
        return mint_aes(dt, vintage, event_type, notional, epp0x=new_epp.index[0])

    if addl_eq_coll > 0:
        DCT_STATE['epp'].loc[epp0x, 'staked'] += premium_amt + addl_eq_coll
        DCT_STATE['epp'].loc[epp0x, 'available'] -= addl_eq_coll

    # ADV tokens burned to pay protocol usage fee
    adv_fee_tokens = DCT_STATE['adv_fee_tokens'].loc[dt] * notional
    adv_fee = adv_fee_tokens * DCT_STATE['adv_price'].loc[dt]
    DCT_STATE['epp'].loc[epp0x, 'available'] -= adv_fee
    DCT_STATE['n_adv_tokens'] -= adv_fee_tokens

    df_cf = DCT_STATE['cashflow']
    cfie = df_cf.shape[0]
    df_cf.loc[cfie, 'amount'] = premium_amt - addl_eq_coll - adv_fee
    df_cf.loc[cfie, 'date'] = dt
    df_cf.loc[cfie, '0x'] = epp0x
    df_cf.loc[cfie, 'id'] = id16
    cfim = df_cf.shape[0]
    if addl_mezz_coll > 0:
        df_cf.loc[cfim, 'amount'] = - np.maximum(0, addl_mezz_coll)
        df_cf.loc[cfim, 'date'] = dt
        df_cf.loc[cfim, '0x'] = 'mpp'
        df_cf.loc[cfim, 'id'] = id16

    # Mezzanine collateral requirements
    assert DCT_STATE['mpp']['available'] > addl_mezz_coll, "MPP has insufficient inactive capital to execute a new AES."
    if addl_mezz_coll > 0:
        DCT_STATE['mpp']['tokens'] += addl_mezz_coll / mpp_token_value()
        DCT_STATE['mpp']['staked'] += addl_mezz_coll
        DCT_STATE['mpp']['available'] -= addl_mezz_coll
        DCT_STATE['mpp_vintage_size'][vintage] += addl_mezz_coll

    # Add new AES to global DF of AES
    DCT_STATE['aes'] = pd.concat([DCT_STATE['aes'], pd.DataFrame(dct_aes, index=[id16])])

    return dct_aes


def req_coll(epp0x, df_aes, dt, decay, power, premium_mult, excess_loss_credit):
    """
    Computes required collateral by EPP

    Sort all active AES written by epp0x. For each AES, equity collateral is highest of:
        1. Exponential decay as a function of time since AES was created
        2. Power function of sort order.
        3. Fixed multiple of premium.
        4. Minimum collateral per contract parameter (nu).

    Final amount is the sum across all AES contracts, less a collateral credit for excess losses incurred by the EPP.

    Parameters
    ----------
    epp0x : str
        Address of the equity protection provider (EPP).
    df_aes : pd.DataFrame
        DataFrame of all Adverse Event Swaps
    dt : datetime
        Current date for evaluating collateral (used for calculating decay since date AES created).
    decay
    power
    premium_mult
    excess_loss_credit

    Returns
    -------
    float
        Required equity collateral amount.
    """
    # Potential extension: take into account diversification across event types
    if df_aes.empty:
        return 0

    ix = df_aes['equity_pool'] == epp0x
    excess_loss = df_aes.loc[ix & ~df_aes['active'], 'excess_loss'].sum()
    df_active = df_aes[ix & df_aes['active']].sort_values(by='notional', ascending=False).reset_index(drop=True)
    coll = (np.max([np.exp(decay * (dt - df_active['date_created']).dt.days).values,
                    power ** (- df_active.index).values,
                    premium_mult * df_active['premium_pct'].values,
                    DCT_STATE['nu'] * np.ones(df_active.shape[0])], axis=0) * df_active['notional'].values).sum()

    return coll - excess_loss_credit * np.minimum(coll, excess_loss)


def req_eq_coll(epp0x, df_aes, dt):
    return req_coll(epp0x, df_aes[df_aes['vintage'] > dt], dt,
                    decay=DCT_STATE['ec_decay'],
                    power=DCT_STATE['ec_power'],
                    premium_mult=DCT_STATE['ec_premium_mult'],
                    excess_loss_credit=DCT_STATE['excess_loss_credit'])


def req_mezz_coll(df_aes, dt, new_eq_coll=None):
    if df_aes.empty:
        return 0

    active_epp = df_aes.loc[df_aes['active'] & (df_aes['vintage'] > dt), 'equity_pool'].unique()
    df_coll = pd.DataFrame(index=active_epp, columns=['mezz', 'eq', 'new_eq'])
    df_coll['mezz'] = DCT_STATE['mezz_coll']
    if new_eq_coll is None:
        new_eq_coll = 0
    df_coll['new_eq'] = new_eq_coll
    df_coll['eq'] = DCT_STATE['epp']['staked'] + df_coll['new_eq'].fillna(0)

    for epp0x in active_epp:
        if pd.notna(df_coll.loc[epp0x, 'mezz']):
            continue
        if df_coll.loc[epp0x, 'eq'] > .9 * df_aes.loc[df_aes['equity_pool'] == epp0x, 'notional'].sum():
            df_coll.loc[epp0x, 'mezz'] = 0
            continue
        df_coll.loc[epp0x, 'mezz'] = np.maximum(0, req_coll(
            epp0x, df_aes, dt,
            decay=DCT_STATE['mc_decay'],
            power=DCT_STATE['mc_power'],
            premium_mult=DCT_STATE['mc_premium_mult'],
            excess_loss_credit=0) - df_coll.loc[epp0x, 'eq'])

    s_mc = pd.concat([DCT_STATE['mezz_coll'], df_coll['mezz']])
    DCT_STATE['mezz_coll'] = s_mc.reset_index().drop_duplicates(subset=['index'], keep='last').set_index('index')[0]

    return df_coll['mezz'].sum()


def mpp_token_value():
    if DCT_STATE['mpp']['staked'] == 0:
        return 1.
    return DCT_STATE['mpp']['staked'] / DCT_STATE['mpp']['tokens']


def execute_events(dt):
    df_aes = DCT_STATE['aes']
    if 'pregen_rand' in DCT_STATE.keys():
        random_input = DCT_STATE['rand'][:df_aes.shape[0]]
        DCT_STATE['rand'] = DCT_STATE['rand'][df_aes.shape[0]:]
    else:
        random_input = np.random.rand(df_aes.shape[0])
    # Potential extension: event occurrence can be correlated across time and within events
    event_occurs = (df_aes['p_def_dt'] > random_input) & df_aes['active']
    df_events = df_aes[event_occurs].copy()
    if df_events.empty:
        return None
    prev_req_mezz_coll = req_mezz_coll(df_aes, dt)
    df_aes.loc[event_occurs, 'active'] = False
    cfi = DCT_STATE['cashflow'].shape[0]
    DCT_STATE['cashflow'].loc[cfi, 'date'] = dt
    DCT_STATE['cashflow'].loc[cfi, 'amount'] = 0
    cfm = None
    for event in df_events.itertuples():
        equity_stake = DCT_STATE['epp'].loc[event.equity_pool, 'staked']
        logging.debug(f"{event.type.capitalize()} event with id {event.id} occurred at {dt:%Y-%m-%d}. "
                      f"Equity pool {event.equity_pool} is impaired by ${event.notional:.0f}. "
                      f"EPP had ${equity_stake:.0f} staked.")
        df_aes.loc[event.id, 'excess_loss'] = -req_eq_coll(event.equity_pool, df_aes, dt) + req_eq_coll(
            event.equity_pool, df_aes[df_aes['id'] != event.id], dt) + df_aes.loc[event.id, 'notional']
        DCT_STATE['mezz_coll'].loc[event.equity_pool] = np.nan
        if pd.notna(DCT_STATE['cashflow'].loc[cfi, '0x']) and DCT_STATE['cashflow'].loc[cfi, '0x'] != event.equity_pool:
            cfi = DCT_STATE['cashflow'].shape[0]
            DCT_STATE['cashflow'].loc[cfi, 'date'] = dt
            DCT_STATE['cashflow'].loc[cfi, 'amount'] = 0
        DCT_STATE['cashflow'].loc[cfi, '0x'] = event.equity_pool
        mc_size = DCT_STATE['mpp']['staked']
        if equity_stake >= event.notional:
            DCT_STATE['epp'].loc[event.equity_pool, 'staked'] -= event.notional
            DCT_STATE['cashflow'].loc[cfi, 'amount'] -= event.notional
            event_mpp_fee = np.minimum(
                event.notional * DCT_STATE['event_mpp_fee'], DCT_STATE['epp'].loc[event.equity_pool, 'staked'])
            if event_mpp_fee > 0:
                DCT_STATE['epp'].loc[event.equity_pool, 'staked'] -= event_mpp_fee
                DCT_STATE['cashflow'].loc[cfi, 'amount'] -= event_mpp_fee
                DCT_STATE['mpp']['staked'] += event_mpp_fee
                if cfm is None:
                    cfm = DCT_STATE['cashflow'].shape[0]
                    DCT_STATE['cashflow'].loc[cfm, 'date'] = dt
                    DCT_STATE['cashflow'].loc[cfm, '0x'] = 'mpp'
                    DCT_STATE['cashflow'].loc[cfm, 'amount'] = 0.
                DCT_STATE['cashflow'].loc[cfm, 'amount'] += event_mpp_fee
        else:
            mc_impairment = event.notional - equity_stake
            assert mc_size > mc_impairment, f"Insufficient mezzanine collateral to pay off {event.id}."
            DCT_STATE['epp'].loc[event.equity_pool, 'staked'] = 0.
            DCT_STATE['cashflow'].loc[cfi, 'amount'] -= DCT_STATE['epp'].loc[event.equity_pool, 'staked']
            if cfm is None:
                cfm = DCT_STATE['cashflow'].shape[0]
                DCT_STATE['cashflow'].loc[cfm, 'date'] = dt
                DCT_STATE['cashflow'].loc[cfm, '0x'] = 'mpp'
                DCT_STATE['cashflow'].loc[cfm, 'amount'] = 0.
            DCT_STATE['mpp']['staked'] -= mc_impairment
            DCT_STATE['cashflow'].loc[cfm, 'amount'] -= mc_impairment
            logging.warning(
                f"Insufficient equity collateral to pay off event id {event.id}. "
                f"Mezzanine pool of ${mc_size:.0f} impaired by ${mc_impairment:.0f} "
                f"({100 * mc_impairment / mc_size:.2f}% of pool)")
        mezz_coll = req_mezz_coll(df_aes, dt)
        if mezz_coll < 25000:
            return
        mezz_cap_ratio = mc_size / mezz_coll
        logging.debug(f"Mezzanine required collateral went from ${prev_req_mezz_coll:.0f} to ${mezz_coll:.0f}. "
                      f"Pool size = ${mc_size:.0f}. New capitalization ratio = {mezz_cap_ratio:.2f}")
        if mezz_cap_ratio < DCT_STATE['recapitalization_threshold']:
            recapitalize_mezz_pool(mezz_coll)


def recapitalize_mezz_pool(mezz_coll):
    logging.warning('Initiating recapitalization of mezzanine collateral pool.')
    new_capital = mezz_coll * DCT_STATE['recapitalization_threshold'] - DCT_STATE['mpp']['staked']
    DCT_STATE['recap_tokens'] += new_capital / mpp_token_value()
    DCT_STATE['mpp']['tokens'] += new_capital / mpp_token_value()
    DCT_STATE['mpp']['staked'] += new_capital


def brownian(start_price, bmu, sigma, size):
    returns = np.random.normal(loc=bmu, scale=sigma, size=size)
    price = start_price * (1 + returns).cumprod()
    return price


def flatten(list_in):
    """Flatten a list of lists into a single list."""
    return [item for sublist in list_in for item in sublist]


def xnpv(rate, cashflows):
    chron_order = sorted(cashflows, key=lambda x: x[0])
    t0 = chron_order[0][0]
    return sum([cf / (1 + rate) ** ((t - t0).days / 365.0) for (t, cf) in chron_order])


def xirr(cashflows, guess=0.1):
    from scipy import optimize
    return optimize.newton(lambda r: xnpv(r, cashflows), guess)


def xirr_df(df_cf, guess=0.1):
    return xirr([(tpl.date, tpl.amount) for tpl in df_cf.itertuples()], guess)


if __name__ == "__main__":
    main(sys.argv.copy())
