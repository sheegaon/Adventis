"""
 Utilities for reading and writing files, such as feather, parquet, CSV, etc.
"""
import os
import logging
import pandas as pd
import numpy as np
import time
import importlib.util as imp_util


if imp_util.find_spec('feather') is None:
    logging.warning("Python environment does not have 'feather' package. Install using "
                    "'conda install feather-format -c conda-forge'.")


def check_overwrite(filename):
    """ Check whether filename exists and prompt before overwriting.
    """
    if os.path.isfile(filename):
        print(f"File exists: {filename}")
        nb = input('Do you want to overwrite it?')
        assert (str(nb)[0:1].upper() == 'Y'), 'Will not overwrite file.'
    return


def write_file_permissions(filename, make_read_only, write_func, **kwargs):
    import stat
    if os.path.isfile(filename):
        #  No need to check if file is read-only, it will be over written anyways
        os.chmod(filename, stat.S_IWRITE)
    write_func(filename, **kwargs)
    if make_read_only:
        os.chmod(filename, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return None


def safe_write(filename, write_func, make_read_only=False, **kwargs):
    import stat
    from datetime import datetime
    if make_read_only and os.path.isfile(filename):
        #  No need to check if file is read-only, it will be over written anyways
        os.chmod(filename, stat.S_IWRITE)
    try:
        write_func(filename + ".writing", **kwargs)
        if safe_remove(filename):
            os.rename(filename + ".writing", filename)
        else:
            fn = f"{filename}.{datetime.now():%Y%m%d_%M%H%S}"
            logging.warning(f"File path {filename} unavailable. Renaming {filename}.writing to {fn}")
            os.rename(filename + ".writing", fn)
        if make_read_only:
            os.chmod(filename, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        logging.debug(f"File path {filename} unavailable.")
        safe_remove(filename + ".writing")


def safe_remove(path, send_error=True):
    if path is None:
        return True
    attempts = 0
    while os.path.isfile(path):
        try:
            os.remove(path)
        except PermissionError:
            if attempts > 5:
                if send_error:
                    logging.error(f'Tried and failed removing {path}')
                else:
                    logging.debug(f'Tried and failed removing {path}')
                return False
            logging.debug(f'PermissionError while removing {path}')
            attempts += 1
            from random import random
            time.sleep(1 + attempts * 10 * random())
        except FileNotFoundError:
            pass
    return True


def lock_with_temp_file(fn, n_tries=200):
    """
    Search for a lock file, wait for its removal, then create a lock file in the specified location if none exists.
    The program must remove the lock file outside this function.
    It is recommended to use safe_remove to remove the lock file.

    Parameters
    ----------
    fn : str
        Path to lock file.
    n_tries : int
        Number of tries to wait for the lock file to be removed. Ignore lock files over an hour old.

    Returns
    -------
    bool
        Success or failure.
    """
    from datetime import datetime
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    success = True
    if os.path.isfile(fn):
        if n_tries > 0:
            last_modified = get_file_modified_time(fn)
            if (datetime.now() - last_modified).total_seconds() > 60 * 60:
                logging.info(f"Ignoring lock file {fn}, last modified {last_modified}.")
                safe_remove(fn)
                success = False
            else:
                from time import sleep
                from random import random
                sleep(1 + 4 * random())
                return lock_with_temp_file(fn, n_tries - 1)
        else:
            logging.warning(f"Reached max number of attempts waiting for {fn}, proceeding past the lock.")
            safe_remove(fn)
            success = False
    try:
        with open(fn, 'x') as out:
            out.truncate(1024)
    except FileExistsError:
        from time import sleep
        from random import random
        sleep(1 + 4 * random())
        logging.debug(f"FileExistsError {fn}")
        return lock_with_temp_file(fn, n_tries)
    return success


def get_file_modified_time(filename, n_attempts=12):
    """
    Try several times to get the file modified time. Return None if filename doesn't exist.

    Parameters
    ----------
    filename : str
        Path to file
    n_attempts : int
        Number of attempts

    Returns
    -------
    datetime
        Last modified time
    """
    last_update = pd.NaT
    attempt = 0
    from time import ctime, sleep
    from random import random
    while pd.isna(last_update) and attempt < n_attempts:
        try:
            last_update = pd.to_datetime(ctime(os.stat(filename).st_mtime))
        except (FileNotFoundError, PermissionError) as e:
            sleep(3 + 4 * random())
            attempt += 1
            logging.debug(f"File reading error {e}.")
    return last_update


def df_to_files(
        df_out, outfile, exclude_csv_output=False, print_time=False, index_csv=False, make_read_only=False,
        to_pkl=True, to_csv=True, to_feather=True, to_parquet=False, to_excel=False):
    """
    Write output files (parquet, feather, CSV or pickle) containing all data in df_out.

    :param df_out: pd.DataFrame containing output data.
    :param outfile: Filename (including location) to output, excluding ".csv" or ".pkl"
    :param exclude_csv_output: Use to_csv instead (bool flipped). Do not output to csv if True, default=False.
    :param print_time: show timing information, default=False
    :param index_csv: include the dataframe index in the csv output, default=False
    :param make_read_only: Make the files read-only, default False
    :param to_pkl: Write pickle file in addition to parquet, default=True
    :param to_csv: Write csv file in addition to parquet, default=True
    :param to_feather: Write feather file, default=True
    :param to_parquet: Write parquet file, default=False
    :param to_excel: Write xlsx file, uses index_csv to toggle index output, default=False
    :return:
    """
    from time import perf_counter
    from random import random
    if outfile.split('.')[-1] in ['feather', 'parquet', 'csv', 'pkl']:
        ftype = '.' + outfile.split('.')[-1]
        logging.warning(f'File name argument should not include a file type suffix. Removing {ftype} from name.')
        outfile = outfile[:-len(ftype)]

    if to_pkl:
        print("\nWriting " + outfile + ".pkl") if print_time else None
        attempts = 0
        while os.path.isfile(outfile + '.pkl.writing'):
            if attempts > 5:
                logging.debug(f"Attempt {1 + attempts} waiting for file {outfile}.pkl.writing to finish...")
            time.sleep(attempts + 10 * random())
        t3 = perf_counter()
        safe_write(outfile + '.pkl', df_out.to_pickle)
        print(f"Time to write pkl outfile {outfile}: {perf_counter() - t3:.3}s") if print_time else None

    if to_feather or to_parquet:
        df_na = df_out.copy()
        for c, v in df_na.isnull().all().iteritems():
            if v is True:
                df_na[c] = np.nan

        if to_feather:
            print("\nWriting " + outfile + ".feather") if print_time else None
            attempts = 0
            while os.path.isfile(outfile + '.feather.writing'):
                logging.debug(f"Attempt {1 + attempts} waiting for file {outfile}.feather.writing to finish...")
                time.sleep(attempts + 10*random())
            t1 = perf_counter()
            dff = df_na.reset_index()
            safe_write(outfile + '.feather', dff.to_feather)
            print(f"Time to write feather outfile {outfile}: {perf_counter() - t1:.3}s") if print_time else None

        if to_parquet:
            print("\nWriting " + outfile + ".parquet") if print_time else None
            attempts = 0
            while os.path.isfile(outfile + '.parquet.writing'):
                logging.debug(f"Attempt {1 + attempts} waiting for file {outfile}.parquet.writing to finish...")
                time.sleep(attempts + 10*random())
            t1 = perf_counter()
            dfp = df_na.copy()
            for col in dfp.dtypes[dfp.dtypes == "datetime64[ns]"].index:
                dfp[col] = dfp[col].astype("datetime64[ms]")
            safe_write(outfile + '.parquet', dfp.to_parquet)
            print(f"Time to write parquet outfile {outfile}: {perf_counter() - t1:.3}s") if print_time else None

    to_csv = False if exclude_csv_output else to_csv
    if to_csv:
        t2 = perf_counter()
        print("Writing " + outfile + ".csv") if print_time else None
        try:
            write_file_permissions(filename=outfile + ".csv", make_read_only=make_read_only,
                                   write_func=df_out.to_csv, index=index_csv)
        except OSError:
            print(f"File path {outfile + '.csv'} unavailable.")
        print(f"Time to write csv outfile {outfile}: {perf_counter() - t2:.3}s") if print_time else None

    if to_excel:
        t2 = perf_counter()
        print("Writing " + outfile + ".xlsx") if print_time else None
        try:
            write_file_permissions(filename=outfile + ".xlsx", make_read_only=make_read_only,
                                   write_func=df_out.to_excel, index=index_csv)
        except OSError:
            print(f"File path {outfile + '.csv'} unavailable.")
        print(f"Time to write xlsx outfile {outfile}: {perf_counter() - t2:.3}s") if print_time else None


def read_from_file(path, filetype, missing='ignore', retry=0, **kwargs):
    from datetime import datetime
    if path is None:
        return None
    if (not os.path.isfile(path)) and (not os.path.isfile(path + '.writing')):
        if missing == 'ignore':
            logging.debug(f'File of type {filetype} not found at {path}')
            return None if retry == 0 else read_from_file(path, filetype, missing, retry-1, **kwargs)
        elif missing == 'warn':
            logging.warning(f'File of type {filetype} not found at {path}')
            return None if retry == 0 else read_from_file(path, filetype, missing, retry-1, **kwargs)
        elif missing == 'error':
            logging.error(f'File of type {filetype} not found at {path}')
            if retry > 0:
                return read_from_file(path, filetype, missing, retry-1, **kwargs)
            raise FileNotFoundError

    df = None
    attempts = 0
    while df is None:
        if os.path.isfile(path + '.writing'):
            if attempts > 1440:
                logging.warning(f"Removing file {path}.writing after approximately 1 hour of attempting to wait.")
                os.remove(path + '.writing')
                continue
            logging.debug(f'File {path} being written. Waiting...')
            from random import random
            attempts += 1
            time.sleep(5 * random())
            continue
        if not os.path.isfile(path):
            if attempts > 10:
                return read_from_file(path, filetype, missing, **kwargs)
            logging.warning(f'File {path} not found after being written. Waiting...')
            attempts += 1
            time.sleep(5)
            continue
        if filetype == 'feather':
            from feather import read_dataframe
            if 'columns' in kwargs:
                kwargs['columns'] = ['index'] + kwargs['columns']
            df = read_dataframe(path, **kwargs).copy()
            df.set_index(df.columns[0], inplace=True)
            for c, v in df.isnull().all().iteritems():
                if v is True:
                    df[c] = None
            df.index.name = None if df.index.name == 'index' else df.index.name
            # Overwrite old feather files from before upgrade to pyarrow 3.0.0
            if (len(kwargs) == 0) and (get_file_modified_time(path) < datetime(2021, 2, 19)):
                df_to_files(df, path[:-8], to_csv=False, to_pkl=False)
            # END overwrite
        elif filetype == 'pickle':
            df = pd.read_pickle(path, **kwargs)
        elif filetype == 'parquet':
            df = pd.read_parquet(path, **kwargs)
        elif filetype == 'csv':
            df = pd.read_csv(path, **kwargs)
        else:
            print(f'misc_utils.read_from_file({path}, {filetype}): File type {filetype} not known.')
    return df


def read_feather(path, missing='ignore', retry=0, **kwargs):
    """
    Read a feather format file at path. The function expects a column 'index' and sets this column to be the index.

    NB: If path is None or file not found at path, function returns None and prints a warning message.

    :param path: String path to file.
    :param missing: {'ignore', 'warn', 'error'}
    :param retry: int, Retry file read if fails first time (handle common network errors when accessing the same file
        using mp). Number of retries to attempt.
    :param kwargs: keyword arguments to be passed to feather.read_dataframe
    :return: DataFrame with contents of feather file.
    """
    from pyarrow.lib import ArrowInvalid
    try:
        return read_from_file(path, 'feather', missing=missing, retry=retry, **kwargs)
    except ArrowInvalid as err:
        if 'Field named' in err.args[0] and 'columns' in kwargs:
            missing_field = err.args[0].split(' ')[2]
            kwargs['columns'].remove(missing_field)
            logging.warning(f"Field named {missing_field} is not found in {path}")
            return read_feather(path, missing, retry-1, **kwargs)
        if retry > 0:
            from time import sleep
            from random import random
            sleep(1 + 4 * random())
            return read_feather(path, missing, retry-1, **kwargs)
        else:
            raise err


def read_parquet(path, missing='ignore', retry=0, **kwargs):
    return read_from_file(path, 'parquet', missing=missing, retry=retry, **kwargs)


def read_pickle(path, missing='ignore', retry=0, **kwargs):
    return read_from_file(path, 'pickle', missing=missing, retry=retry, **kwargs)


def find_corrupt_feather(folder, start_date, only_check=""):
    """
    Walk through a folder, load all feather files modified after start_date, and selectively remove
    any that throw exceptions.

    Parameters
    ----------
    folder : str
        Search folder.
    start_date : str or datetime
        Skip files modified before start_date.
    only_check : str
        Only check files containing this string
    """
    start_date = pd.to_datetime(start_date)
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn[-8:] != '.feather':
                continue
            if only_check not in fn:
                continue
            full_fn = os.path.join(root, fn)
            mod_time = get_file_modified_time(full_fn)
            if mod_time > start_date:
                logging.debug(f'Attempting load of {full_fn}')
                try:
                    read_feather(full_fn)
                except OSError:
                    logging.warning(f"Found corrupt feather file {full_fn}. Attempting removal")
                    os.remove(full_fn)
