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
NUM_RANDOM = 20
NUM_MALICIOUS = 3
NUM_FAIR = 60
MPP_SEED_FUND = 100000
DCT_STATE = dict()
PROJ_FOLDER = os.path.join(os.getenv('QRSHARE'),  "ProductDesign", "TFI", "OneOff", "data")


def main(argv):
    t0 = perf_counter()
    input_args = mu.arg_handler(argv, input_args=dict(
        nu=0.005, ec_power=2.5, mc_power=2., ec_premium_markup=.1, ec_half_life=14, mc_half_life=3,
        excess_loss_credit=0.5, event_mpp_fee=.1, min_prem_rate=.001, recapitalization_threshold=.9,
        adv_fee_pct=0.0001, oracle_fee=0., run_auction=False),
                                name='OneOff/adventis_sim.py')
    mu.setup_logger(log_folder=os.path.join(os.getenv('QRSHARE'),  "ProductDesign", "TFI", "OneOff", "logs"),
                    filename=os.path.basename(__file__)[:-3], console_level=logging.INFO)

    logging.debug(f"Running with input_args={input_args}")
    if input_args['run_auction']:
        for _ in np.arange(200):
            auction_simulator(True)
        logging.info(f"Done. Time to run: {timedelta(seconds=perf_counter() - t0)}")
        return

    all_dates = pd.date_range(start=START_DATE, end=END_DATE)
    init_epp_capital = 1e6
    init_mpp_capital = 1e7
    DCT_STATE['epp'] = pd.concat(
        flatten([[pd.DataFrame(
            {'type': 'random', 'bias': random.normalvariate(.75, .5), 'noise': .05 * random.betavariate(2, 2),
             'vintage': v, 'available': init_epp_capital, 'staked': 0, 'active': True},
            index=[f'0xv{LST_VINTAGES.index(v)}e{i}rnd']) for i in np.arange(NUM_RANDOM)] for v in LST_VINTAGES]))

    DCT_STATE['epp'] = pd.concat([DCT_STATE['epp']] + flatten(
        [[pd.DataFrame(
            {'type': 'malicious', 'vintage': v, 'available': init_epp_capital, 'staked': 0, 'active': True},
            index=[f'0xv{LST_VINTAGES.index(v)}e{i}mal']) for i in np.arange(NUM_MALICIOUS)] for v in LST_VINTAGES]))

    DCT_STATE['epp'] = pd.concat([DCT_STATE['epp']] + flatten(
        [[pd.DataFrame(
            {'type': 'fair', 'vintage': v, 'available': init_epp_capital, 'staked': 0, 'active': True},
            index=[f'0xv{LST_VINTAGES.index(v)}e{i}fai']) for i in np.arange(NUM_FAIR)] for v in LST_VINTAGES]))

    # Load previously generated random values, so that runs with slightly different parameters are comparable
    save_pickle = True
    if os.path.exists(os.path.join(PROJ_FOLDER, "dct_state.pickle")):
        with open(os.path.join(PROJ_FOLDER, "dct_state.pickle"), 'rb') as pkl_handle:
            dct_state = pickle.load(pkl_handle)
        if dct_state['epp'].shape == DCT_STATE['epp'].shape:
            save_pickle = False
            for k in dct_state.keys():
                DCT_STATE[k] = dct_state[k].copy()

    if save_pickle:
        DCT_STATE['adv_price'] = pd.Series(
            index=all_dates, data=brownian(start_price=1., size=len(all_dates), bmu=0.001, sigma=.005))
        DCT_STATE['pregen_rand'] = np.random.rand(2000000)
        DCT_STATE['pregen_aes'] = pd.concat([pd.DataFrame(index=[i], data={
            'id': ''.join(random.choices(string.ascii_lowercase + string.digits, k=16)),
            'p_def_expiry': 0.15 * random.random(),
            'notional': random.randint(100, 10000)}) for i in np.arange(10000)]).drop_duplicates(subset=['id'])
        with open(os.path.join(PROJ_FOLDER, "dct_state.pickle"), 'wb') as pkl_handle:
            pickle.dump(DCT_STATE, pkl_handle)

    DCT_STATE['rand'] = DCT_STATE['pregen_rand'].copy()
    DCT_STATE['mpp'] = {'tokens': MPP_SEED_FUND, 'available': init_mpp_capital, 'staked': MPP_SEED_FUND}
    DCT_STATE['mpp_vintage_size'] = {v: 0 for v in LST_VINTAGES}
    DCT_STATE['mezz_coll'] = pd.Series(dtype=float)
    DCT_STATE['recap_tokens'] = 0
    DCT_STATE['n_adv_tokens'] = 1e7
    DCT_STATE['cashflow'] = pd.DataFrame(
        index=[0], data={'amount': -MPP_SEED_FUND, 'date': START_DATE, '0x': 'mpp', 'id': 0})
    for k in input_args.keys():
        DCT_STATE[k] = input_args[k]
    DCT_STATE['ec_decay'] = - np.log(2) / DCT_STATE['ec_half_life']
    DCT_STATE['mc_decay'] = - np.log(2) / DCT_STATE['mc_half_life']
    DCT_STATE['min_prem'] = -np.log(1 - DCT_STATE['min_prem_rate']) / 365  # Must charge at least min_prem_rate per year
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
        for et in LST_TYPES:
            mint_aes(dt=dt, event_type=et)
            mint_aes(dt=dt, event_type=et)
            mint_aes(dt=dt, event_type=et)
            mint_aes(dt=dt, event_type=et)
            mint_aes(dt=dt, event_type=et)

        # Randomly generate event realizations and apply swap logic
        execute_events(dt)

        # Recalculate mezzanine collateral requirements monthly
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

    # Final cash flow from what's left in MPP
    cfi = DCT_STATE['cashflow'].shape[0]
    DCT_STATE['cashflow'].loc[cfi, 'date'] = END_DATE
    DCT_STATE['cashflow'].loc[cfi, '0x'] = 'mpp'
    DCT_STATE['cashflow'].loc[cfi, 'id'] = 'end of simulation'
    DCT_STATE['cashflow'].loc[cfi, 'amount'] = DCT_STATE['mpp']['staked']

    df_mpp = s_mpp_token_value.reset_index(name='value')
    df_mpp['eom'] = mu.eom(df_mpp['index'])
    df_mpp.drop_duplicates(keep='last', subset=['eom'])

    df_cf = DCT_STATE['cashflow']
    for x in df_cf['0x'].unique():
        pnl = df_cf.loc[df_cf['0x'] == x, 'amount'].sum()
        inv = - df_cf.loc[(df_cf['0x'] == x) & (df_cf['amount'] < 0), 'amount'].sum()
        if inv == 0:
            continue
        msg = f"Returned {100 * pnl / inv:.2f}%, ${pnl:,.0f} P&L on net invested capital ${inv:,.0f} for {x}"
        if x == 'mpp':
            logging.info(msg)
        else:
            logging.debug(msg)
    for epp_type in DCT_STATE['epp']['type'].unique():
        type_0x = DCT_STATE['epp'][DCT_STATE['epp']['type'] == epp_type].index
        type_ix = df_cf['0x'].isin(type_0x)
        if not type_ix.any():
            continue
        epp_pnl = df_cf.loc[type_ix, 'amount'].sum()
        epp_inv = -df_cf.loc[type_ix & (df_cf['amount'] < 0), 'amount'].sum()
        df_aes = DCT_STATE['aes'].copy()
        epp_paid = df_aes.loc[df_aes['equity_pool'].isin(type_0x) & (~df_aes['active']), 'notional'].sum()
        logging.info(f"{epp_type.capitalize()} EPPs returned {100 * epp_pnl / epp_inv:.2f}%, "
                     f"${epp_pnl:,.0f} P&L on ${epp_inv:,.0f} invested. Paid ${epp_paid:,.0f}")
    mu.df_to_files(df_cf, os.path.join(os.getenv('QRSHARE'),  "ProductDesign", "TFI", "OneOff", "data", "adventis_cf"),
                   to_csv=False, to_feather=False)
    logging.info(f"EPP wallets abandoned: {(~DCT_STATE['epp']['active']).sum()}")
    logging.info(f"ADV tokens burned: {1e7 - DCT_STATE['n_adv_tokens']:.0f}")

    logging.info(f"Done. Time to run: {timedelta(seconds=perf_counter() - t0)}")
    return

# ==================
# Functions for main
# ==================


def random_epp(p, bias, noise):
    return p * np.exp(random.normalvariate(bias, noise))


def malicious_epp(p):
    return np.minimum(0.001, p * 0.1)


def fair_epp(p):
    return p


def epp_premium_func(p, epp_info):
    if epp_info['type'] == 'random':
        return random_epp(p, bias=epp_info['bias'], noise=epp_info['noise'])
    elif epp_info['type'] == 'malicious':
        return malicious_epp(p)
    elif epp_info['type'] == 'fair':
        return fair_epp(p)
    else:
        raise ValueError(f"Invalid EPP type {epp_info['type']}")


def mint_aes(dt, vintage=None, event_type=None, notional=None, epp0x=None):
    # Move on to next vintage when within 15 days of expiration
    lst_v = [v for v in LST_VINTAGES if v > dt + pd.Timedelta(15, 'd')]
    if len(lst_v) == 0:
        return None
    if vintage is None:
        vintage = random.choice(lst_v)

    if event_type is None:
        event_type = random.choice(LST_TYPES)

    df_epp = DCT_STATE['epp'][(DCT_STATE['epp']['vintage'] == vintage) & DCT_STATE['epp']['active']]
    if epp0x is None:
        epp0x = random.choice(df_epp.index)

    if DCT_STATE['aes'].shape[0] < DCT_STATE['pregen_aes'].shape[0]:
        dct_aes = DCT_STATE['pregen_aes'].iloc[DCT_STATE['aes'].shape[0], :].to_dict()
        id16 = dct_aes['id']
        p_def_expiry = dct_aes['p_def_expiry']
        notional = dct_aes['notional']
    else:
        id16 = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
        while not DCT_STATE['aes'].empty and id16 in DCT_STATE['aes']['id'].tolist():
            id16 = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
        p_def_expiry = 0.15 * random.random()
        if notional is None:
            notional = random.randint(100, 10000)
    days_to_expiry = (vintage - dt).days
    hazard = - np.log(1 - p_def_expiry) / days_to_expiry
    if DCT_STATE['epp'].loc[epp0x, 'type'] == 'malicious':
        hazard *= 10
        p_def_expiry = 1 - np.exp(-hazard * days_to_expiry)
    p_def_dt = 1 - np.exp(-hazard)
    premium_pct = np.maximum(epp_premium_func(p_def_expiry, DCT_STATE['epp'].loc[epp0x]),
                             1 - np.exp(-DCT_STATE['min_prem'] * days_to_expiry))
    dct_aes = {'type': event_type, 'date_created': dt, 'vintage': vintage,
               'active': True, 'excess_loss': np.nan,
               'notional': notional, 'p_def': p_def_expiry, 'p_def_dt': p_def_dt,
               'premium_pct': premium_pct, 'equity_pool': epp0x}

    # Calculate collateral requirements
    df_aes = DCT_STATE['aes'].copy()
    for v in ['notional', 'equity_pool', 'active', 'excess_loss', 'date_created', 'premium_pct', 'vintage']:
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
        new_0x = f"0xv{LST_VINTAGES.index(vintage)}e{DCT_STATE['epp'][DCT_STATE['epp']['vintage'] == vintage].shape[0]}"
        new_epp.index = [new_0x]
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
    df_cf.loc[cfie, 'amount'] = - addl_eq_coll - adv_fee
    df_cf.loc[cfie, 'date'] = dt
    df_cf.loc[cfie, '0x'] = epp0x
    df_cf.loc[cfie, 'id'] = f"mint {id16}"
    cfim = df_cf.shape[0]
    if addl_mezz_coll > 0:
        df_cf.loc[cfim, 'amount'] = - np.maximum(0, addl_mezz_coll)
        df_cf.loc[cfim, 'date'] = dt
        df_cf.loc[cfim, '0x'] = 'mpp'
        df_cf.loc[cfim, 'id'] = f"mint {id16}"

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


def req_coll(epp0x, df_aes, dt, decay, power, premium_markup, excess_loss_credit):
    """
    Computes required collateral by EPP

    Sort all active AES written by epp0x. For each AES, equity collateral is notional times the highest of:
        1. Exponential decay from 100% as a function of time since AES was created
        2. Power function of sort order.
        3. Fixed markup over premium.
        4. Minimum collateral per contract parameter (nu).

    Final amount is the sum across all AES contracts, less a collateral credit for excess losses incurred.

    Potential extension: take into account diversification across event types

    Parameters
    ----------
    epp0x : str
        Address of the equity protection provider (EPP).
    df_aes : pd.DataFrame
        DataFrame of all Adverse Event Swaps
    dt : datetime
        Current date for evaluating collateral (used for calculating decay since date AES created).
    decay : float
        Exponential decay parameter.
    power : float
        Power function base.
    premium_markup : float
        Markup on premium setting minimum collateral.
    excess_loss_credit : float
        Factor applied to excess losses as a collateral credit.

    Returns
    -------
    float
        Required collateral amount.
    """
    epp_ix = df_aes['equity_pool'] == epp0x
    epp_ix_active = epp_ix & df_aes['active']
    if sum(epp_ix_active) == 1:
        return df_aes.loc[epp_ix_active, 'notional'].iloc[0]
    excess_loss = 0
    if excess_loss_credit > 0:
        excess_loss = df_aes.loc[epp_ix & ~df_aes['active'], 'excess_loss'].sum()
    df_active = df_aes[epp_ix & df_aes['active']].sort_values(by='notional', ascending=False).reset_index(drop=True)
    coll = (np.max([np.exp(decay * (dt - df_active['date_created']).dt.days).values,
                    power ** (- df_active.index).values,
                    (1 + premium_markup) * df_active['premium_pct'].values,
                    DCT_STATE['nu'] * np.ones(df_active.shape[0])], axis=0) * df_active['notional'].values).sum()

    return coll - excess_loss_credit * np.minimum(coll, excess_loss)


def req_eq_coll(epp0x, df_aes, dt):
    if df_aes.empty:
        return 0

    return req_coll(epp0x, df_aes[df_aes['vintage'] > dt], dt,
                    decay=DCT_STATE['ec_decay'],
                    power=DCT_STATE['ec_power'],
                    premium_markup=DCT_STATE['ec_premium_markup'],
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
            premium_markup=0,
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
    df_active = df_aes[df_aes['active']].copy()
    if df_events.empty:
        return None
    prev_req_mezz_coll = req_mezz_coll(df_aes, dt)
    df_aes.loc[event_occurs, 'active'] = False
    cfi = DCT_STATE['cashflow'].shape[0]
    DCT_STATE['cashflow'].loc[cfi, 'date'] = dt
    DCT_STATE['cashflow'].loc[cfi, 'amount'] = 0
    DCT_STATE['cashflow'].loc[cfi, 'id'] = "event payoff"
    cfm = None
    for event in df_events.itertuples():
        equity_stake = DCT_STATE['epp'].loc[event.equity_pool, 'staked']
        logging.debug(f"{event.type.capitalize()} event with id {event.Index} occurred at {dt:%Y-%m-%d}. "
                      f"Equity pool {event.equity_pool} is impaired by ${event.notional:,.0f}. "
                      f"EPP had ${equity_stake:,.0f} staked.")
        df_aes.loc[event.Index, 'excess_loss'] = -req_eq_coll(event.equity_pool, df_aes, dt) + req_eq_coll(
            event.equity_pool, df_aes[df_aes.index != event.Index], dt) + df_aes.loc[event.Index, 'notional']
        DCT_STATE['mezz_coll'].loc[event.equity_pool] = np.nan
        if pd.notna(DCT_STATE['cashflow'].loc[cfi, '0x']) and DCT_STATE['cashflow'].loc[cfi, '0x'] != event.equity_pool:
            cfi = DCT_STATE['cashflow'].shape[0]
            DCT_STATE['cashflow'].loc[cfi, 'date'] = dt
            DCT_STATE['cashflow'].loc[cfi, 'amount'] = 0
            DCT_STATE['cashflow'].loc[cfi, 'id'] = "event payoff"
        DCT_STATE['cashflow'].loc[cfi, '0x'] = event.equity_pool
        mc_size = DCT_STATE['mpp']['staked']
        if equity_stake >= event.notional:
            DCT_STATE['epp'].loc[event.equity_pool, 'staked'] -= event.notional
            DCT_STATE['cashflow'].loc[cfi, 'amount'] -= event.notional
            event_mpp_fee = np.minimum(
                event.notional * DCT_STATE['event_mpp_fee'], DCT_STATE['epp'].loc[event.equity_pool, 'staked'])
            equity_total_notional = df_active.loc[df_active['equity_pool'] == event.equity_pool, 'notional'].sum()
            if equity_stake == equity_total_notional:
                event_mpp_fee = 0
            if event_mpp_fee > 0:
                DCT_STATE['epp'].loc[event.equity_pool, 'staked'] -= event_mpp_fee
                DCT_STATE['cashflow'].loc[cfi, 'amount'] -= event_mpp_fee
                DCT_STATE['mpp']['staked'] += event_mpp_fee
                if cfm is None:
                    cfm = DCT_STATE['cashflow'].shape[0]
                    DCT_STATE['cashflow'].loc[cfm, 'date'] = dt
                    DCT_STATE['cashflow'].loc[cfm, '0x'] = 'mpp'
                    DCT_STATE['cashflow'].loc[cfm, 'amount'] = 0.
                    DCT_STATE['cashflow'].loc[cfm, 'id'] = "event payoff"
                DCT_STATE['cashflow'].loc[cfm, 'amount'] += event_mpp_fee
        else:
            mc_loss = event.notional - equity_stake
            assert mc_size > mc_loss, f"Insufficient mezzanine collateral to pay off {event.Index}."
            DCT_STATE['epp'].loc[event.equity_pool, 'staked'] = 0.
            DCT_STATE['cashflow'].loc[cfi, 'amount'] -= DCT_STATE['epp'].loc[event.equity_pool, 'staked']
            if cfm is None:
                cfm = DCT_STATE['cashflow'].shape[0]
                DCT_STATE['cashflow'].loc[cfm, 'date'] = dt
                DCT_STATE['cashflow'].loc[cfm, '0x'] = 'mpp'
                DCT_STATE['cashflow'].loc[cfm, 'amount'] = 0.
                DCT_STATE['cashflow'].loc[cfm, 'id'] = "event payoff"
            DCT_STATE['mpp']['staked'] -= mc_loss
            DCT_STATE['cashflow'].loc[cfm, 'amount'] -= mc_loss
            logging.warning(
                f"Insufficient equity to pay off {event.Index}. Mezzanine pool of ${mc_size:,.0f} lost ${mc_loss:,.0f} "
                f"({100 * mc_loss / mc_size:.2f}%). EPP type: {DCT_STATE['epp'].loc[event.equity_pool, 'type']}.")
        mezz_coll = req_mezz_coll(df_aes, dt)
        if mezz_coll < 25000:
            return
        mezz_cap_ratio = mc_size / mezz_coll
        logging.debug(f"Mezzanine required collateral went from ${prev_req_mezz_coll:,.0f} to ${mezz_coll:,.0f}. "
                      f"Pool size = ${mc_size:,.0f}. New capitalization ratio = {mezz_cap_ratio:.2f}")
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


def auction_simulator(speculative=False, df_orders=None):
    """
    Simulate the auction round of an AES

    Parameters
    ----------
    speculative : bool
        Is the proposal speculative or a real proposed trade
    df_orders : pd.DataFrame
        Price, amount, and side of orders. Side in {'b', 'o', 'p'} for bid, offers, and proposal, respectively.
        Generate randomly if not provided.
    """
    tick = 0.0025

    # Combined DF with bids (b), offers (o), and proposal (p)
    if df_orders is None:
        num_orders = random.randint(1, 50)
        df_orders = pd.DataFrame({
            'price': tick * np.random.randint(1, int(.15 / tick), num_orders),
            'amount': np.random.randint(1, 100, num_orders),
            'side': np.random.choice(['b', 'o'], num_orders)})
        df_orders.loc[0, 'side'] = 'p'
    df_orders['price'] = np.round(df_orders['price'], 4)
    bids = df_orders['side'] == 'b'
    offers = df_orders['side'] == 'o'
    num_bids = sum(bids)
    df_orders['fill'] = np.nan

    if num_bids == 0:
        logging.info(f"No alternative buyers. Proposed trade goes through.")
        df_orders.loc[0, 'fill'] = 0 if speculative else df_orders.loc[0, 'amount']
        df_orders.loc[0, 'buy_fill'] = 0 if speculative else df_orders.loc[0, 'amount']
        return df_orders
    
    # Cumulate bids and offers into bid and ask order books
    df_bidbook = df_orders[df_orders['side'] == 'b'].groupby('price')['amount'].sum().sort_index(
        ascending=False).to_frame()
    df_bidbook.index = [f'{x:.4f}' for x in df_bidbook.index]
    df_bidbook['cum_amount'] = df_bidbook['amount'].cumsum()
    df_all_offers = df_orders[df_orders['side'] != 'b'].copy()
    df_all_offers.loc[0, 'price'] += tick  # Proposal is included as an offer at 1 tick above proposed price
    df_all_offers['price'] = [f'{x:.4f}' for x in df_all_offers['price']]
    df_askbook = df_all_offers.groupby('price')['amount'].sum().to_frame()
    df_askbook['cum_amount'] = df_askbook['amount'].cumsum()

    # Generate combined order book
    df_book = pd.DataFrame(index=tick * (1 + np.arange(int(.15 / tick))))
    df_book.index = [f'{x:.4f}' for x in df_book.index]
    df_book['cum_bid'] = df_bidbook['cum_amount']
    df_book['cum_ask'] = df_askbook['cum_amount']
    df_book['cum_bid'].bfill(inplace=True)
    df_book['cum_ask'].ffill(inplace=True)

    # Check for bids higher than the lowest offer
    if not df_book['cum_bid'].le(df_book['cum_ask']).any():
        logging.info("Only very low bids entered. Proposed trade goes through.")
        df_orders.loc[0, 'fill'] = 0 if speculative else df_orders.loc[0, 'amount']
        df_orders.loc[0, 'buy_fill'] = 0 if speculative else df_orders.loc[0, 'amount']
        return
    
    # Results of double auction
    mkt_price = df_book.index[df_book['cum_bid'] <= df_book['cum_ask']][0]
    mkt_volume = df_book.loc[mkt_price, 'cum_bid']
    excess_offers = df_book.loc[mkt_price, 'cum_ask'] - mkt_volume
    mkt_price = float(mkt_price)
    
    # Calculate fills of bids
    df_orders.loc[bids, 'fill'] = 0
    df_orders.loc[(df_orders['price'] - mkt_price > -1e-4) & bids, 'fill'] = df_orders['amount']

    # Determine whether proposed trade goes through
    if mkt_volume < df_orders.loc[0, 'amount']:
        logging.info("Insufficient bidding above proposed price. Proposed trade goes through.")
        df_orders.loc[0, 'fill'] = mkt_volume if speculative else df_orders.loc[0, 'amount']
        df_orders.loc[0, 'buy_fill'] = 0 if speculative else df_orders.loc[0, 'amount']
        if speculative:
            df_orders.loc[(df_orders['price'] - mkt_price > -1e-4) & bids, 'fill'] = df_orders['amount']
        return
    if mkt_price - df_orders.loc[0, 'price'] > 1e-4:
        logging.info(f"Market clearing price ({mkt_price:.4f}) is higher than proposal ({df_orders.loc[0, 'price']}). "
                     "Original EPP is filled at a higher price. "
                     "Original buyer is not filled.")
        df_orders.loc[0, 'fill'] = df_orders.loc[0, 'amount']
        df_orders.loc[0, 'buy_fill'] = 0
    elif mkt_price - df_orders.loc[0, 'price'] < -1e-4:
        logging.info(f"Market clearing price ({mkt_price:.4f}) is lower than proposal ({df_orders.loc[0, 'price']}). "
                     f"Proposed trade goes through.")
        df_orders.loc[0, 'fill'] = 0 if speculative else df_orders.loc[0, 'amount']
        df_orders.loc[0, 'buy_fill'] = 0 if speculative else df_orders.loc[0, 'amount']
    else:
        logging.info("Market clearing price equals proposal. Proposed trade goes through.")
        df_orders.loc[0, 'fill'] = df_orders.loc[0, 'amount']
        df_orders.loc[0, 'buy_fill'] = 0 if speculative else df_orders.loc[0, 'amount']
        if speculative:
            excess_offers += df_orders.loc[0, 'amount']

    # Calculate fills of offers
    offer_full_fill = df_askbook.index[df_askbook['cum_amount'] <= mkt_volume - df_orders.loc[0, 'fill']]
    df_orders.loc[df_orders['price'].isin(offer_full_fill) & offers, 'fill'] = df_orders['amount']
    offer_partial_fill = df_orders['fill'].isna() & (df_orders['price'] - mkt_price < 1e-4)
    offer_partial_fill_ratio = 1 - excess_offers / df_orders.loc[offer_partial_fill, 'amount'].sum()
    df_orders.loc[offer_partial_fill, 'fill'] = offer_partial_fill_ratio * df_orders['amount']
    df_orders['fill'].fillna(0, inplace=True)

    # Resubmit low price offers that brough down market prices as a disincentive to skewing prices by submitting large
    # offers at low prices
    if mkt_price - df_orders.loc[0, 'price'] < -1e-4:
        df_next_proposal = df_orders[offer_partial_fill].copy()
        df_next_proposal['amount'] -= df_next_proposal['fill']
        df_next_proposal = df_next_proposal.groupby('price')['amount'].sum().reset_index().head(1)
        logging.info(f"Lowest price partial fills submitted as speculative proposal in the next auction.\n"
                     f"{df_next_proposal}")

    # Verify bid/offer fills match
    ofill = df_orders.loc[offers, 'fill'].sum() + df_orders.loc[0, 'fill']
    bfill = df_orders.loc[bids, 'fill'].sum() + df_orders.loc[0, 'buy_fill']
    assert abs(ofill - bfill) < 1e-6, f"Bid filled != offer filled, bid={bfill} offer={ofill}"

    return df_orders


if __name__ == "__main__":
    main(sys.argv.copy())
