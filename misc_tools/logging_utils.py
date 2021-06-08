"""
 Utilities for logging.
"""
import os
import logging


def setup_logger(filename=None, log_folder=None, quietly=False, console_level=logging.WARNING,
                 include_process_name=False):
    """
    Set up the logger with convenient defaults.

    Creates three separate log files at levels DEBUG, INFO, and WARNING, plus console messages set to WARNING.

    Parameters
    ----------
    filename : str, optional
        Base file name for log files. Leave blank to log to console only.
    log_folder : str, optional
        Path to folder where log files will be created. Default is logs folder on PDDATDIR.
    quietly : bool, default False
        Don't print path to log file.
    console_level : [logging.DEBUG, logging.INFO, logging.WARNING]
        Logging level to the console.
    include_process_name : bool, default False
        Include the process name in the log output

    Returns
    -------
    datetime
        The start time by which all log file names are tagged.
    """
    from datetime import datetime
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    console_fmt = '%(asctime)s %(levelname)s [%(funcName)s()]: %(message)s'
    if filename is None:
        logging.basicConfig(level=console_level, format=console_fmt, datefmt='%H:%M:%S')
        return None

    import sys
    if log_folder is None:
        log_folder = os.path.join(os.getenv('PDDATDIR'), 'logs')
    os.makedirs(log_folder, exist_ok=True)
    start_time = datetime.now()
    logformat = '%(asctime)s %(levelname)-8s [%(filename)-15s - %(funcName)15s()]: %(message)s'
    if include_process_name:
        logformat = '%(asctime)s %(levelname)-8s [%(processName)-12s %(filename)-15s - %(funcName)15s()]: %(message)s'
    formatter = logging.Formatter(logformat, datefmt='%m/%d %H:%M:%S')
    logfilename = os.path.join(log_folder, f'{filename}_debug_{start_time:%Y-%m-%d_%H%M%S}.log')
    logging.basicConfig(filename=logfilename, datefmt='%m/%d %H:%M:%S', level=logging.DEBUG, format=logformat)
    logger = logging.getLogger('')
    logfilename = os.path.join(log_folder, f'{filename}_warnings_{start_time:%Y-%m-%d_%H%M%S}.log')
    fhw = logging.FileHandler(logfilename)
    fhw.setLevel(logging.WARNING)
    fhw.setFormatter(formatter)
    logger.addHandler(fhw)
    logfilename = os.path.join(log_folder, f'{filename}_{start_time:%Y-%m-%d_%H%M%S}.log')
    fh = logging.FileHandler(logfilename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    try:
        if 'svc' in os.getenv('USERNAME').lower():
            raise ModuleNotFoundError
        import coloredlogs
        coloredlogs.DEFAULT_FIELD_STYLES['funcName'] = {'color': 'cyan'}
        coloredlogs.install(fmt=console_fmt, datefmt='%H:%M:%S', level=console_level)
    except ModuleNotFoundError:
        formatter = logging.Formatter(console_fmt, datefmt='%H:%M:%S')
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    print(f"Logging to {logfilename}") if not quietly else None
    logging.info(f"Log file started at {logfilename}")
    logging.debug(f"Username {os.getenv('USERNAME').lower()} [{os.getlogin()}] on {os.getenv('COMPUTERNAME')} "
                  f"process ID {os.getpid()} [{os.getppid()}]")
    return start_time


def clean_log_folder(log_folder):
    """
    Selectively removes older log files.

    Parameters
    ----------
    log_folder : path
        Path to folder to clean
    """
    from datetime import datetime
    import pandas as pd

    from qr_shared_src.misc_tools.file_utils import get_file_modified_time, safe_remove
    from qr_shared_src.misc_tools.date_utils import today

    logging.debug(f'Cleaning up log folder: {log_folder}')
    for fn in os.listdir(log_folder):
        fmod = get_file_modified_time(os.path.join(log_folder, fn))
        # Keep all files generated in the last 2 days
        if fmod > today() - pd.Timedelta(1, 'd'):
            continue
        sz = os.stat(os.path.join(log_folder, fn)).st_size
        # Delete very small files older than 30 days
        if (datetime.now() - fmod).days > 30 and sz < 500:
            safe_remove(os.path.join(log_folder, fn), send_error=False)
            continue
        # Keep info files (unless very small and older than 30 days)
        if fn.count('_debug_') + fn.count('_warnings_') + fn.count('calc_mp_') == 0:
            continue
        # Delete blank files
        if sz == 0:
            safe_remove(os.path.join(log_folder, fn), send_error=False)
            continue
        # Delete debug/warnings/calc_mp files older than 14 days
        if (datetime.now() - fmod).days > 14:
            safe_remove(os.path.join(log_folder, fn), send_error=False)
