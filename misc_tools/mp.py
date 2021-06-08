"""
 Multiprocessing utils:
    -Multiprocessing logging
    -Subclass of Process which passes along exceptions

"""
import logging
import logging.handlers
import multiprocessing as mp
import traceback
from datetime import datetime


class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except (TypeError, ValueError, UnboundLocalError) as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            raise e
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            # raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


# Because you'll want to define the logging configurations for listener and workers, the
# listener and worker process functions take a configurer parameter which is a callable
# for configuring logging for that process. These functions are also passed the queue,
# which they use for communication.
def listener_configurer():
    from qr_shared_src.misc_tools.misc_utils import setup_logger
    setup_logger('mptest', include_process_name=True)


# This is the listener process top-level loop: wait for logging events
# (LogRecords)on the queue and handle them, quit when you get a None for a
# LogRecord.
def log_listener_fn(queue, configurer):
    configurer()
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys
            import traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise Exception


# The worker configuration is done at the start of the worker process run.
# Note that on Windows you can't rely on fork semantics, so each process
# will run the logging configuration code when it starts.
def configure_log_for_current_process(queue):
    if queue is None:
        print("Value 'None' for queue sent to configurer.")
        return
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    for rh in root.handlers:
        if isinstance(rh, logging.StreamHandler):
            root.removeHandler(rh)
    root.addHandler(h)
    # send all messages, for demo; no other level or filter logic applied.
    root.setLevel(logging.DEBUG)


def exception_handler(lst_proc, log_queue=None):
    from time import perf_counter
    t0 = perf_counter()
    error = None
    email_list = None
    p_tb_list = []
    while len(lst_proc) > 0:
        p = lst_proc.pop(0)
        p.join(1)
        if p.exception:
            import time
            error, tb = p.exception
            time.sleep(.5)
            logging.error(f"Error in {p.name} [{p.pid}]")
            logging.error(tb)
            time.sleep(.5)
            import os
            # Save error details to send an error e-mail at the end
            if not os.getenv('NOTPROD', False):  # and email_list is not None:
                email_list = [f"{x}@{os.getenv('USERDNSDOMAIN').lower()}" for x in os.getenv('EMAIL_LIST').split(',')]
                p_tb_list += [(p, tb)]
            continue
        if p.exitcode is None:
            lst_proc.append(p)
            if perf_counter() - t0 > 300:
                logging.debug(f"{len(lst_proc)} processes still running: {[p.name for p in lst_proc]}")
                t0 = perf_counter()
            continue
    if error is not None:
        if email_list is not None:
            from qr_shared_src.misc_tools.misc_utils import sendmail
            for p, tb in p_tb_list:
                sendmail(email_list, 'qrpd-auto@lordabbett.com',
                         f'Exception encountered in PROD process {p.name} [{p.pid}]',
                         f'{error}\n\n' + tb)
        if log_queue is not None:
            print(f"{datetime.now()} Logging queue passed in. Sending None to log_queue.")
            logging.warning('Logging queue passed in. Sending None to log_queue.')
            log_queue.put(None)

        raise error
    return lst_proc


def remove_done(lst_proc, log_queue=None):
    import numpy as np
    for _ in np.arange(len(lst_proc)):
        p = lst_proc.pop(0)
        p.join(1)
        if p.exception:
            import time
            error, tb = p.exception
            time.sleep(.5)
            logging.error(f"Error in {p.name} [{p.pid}]")
            logging.error(tb)
            if log_queue is not None:
                print(f"{datetime.now()} Logging queue passed in. Sending None to log_queue.")
                logging.warning('Logging queue passed in. Sending None to log_queue.')
                log_queue.put(None)
            raise error
        if p.exitcode is None:
            lst_proc.append(p)
    return lst_proc


# Arrays used for random selections in the logging demo
LEVELS = [logging.DEBUG, logging.INFO, logging.WARNING,
          logging.ERROR, logging.CRITICAL]

LOGGERS = ['a.b.c', 'd.e.f']

MESSAGES = [
    'Random message #1',
    'Random message #2',
    'Random message #3',
]


# This is the worker process top-level loop, which just logs ten events with
# random intervening delays before terminating.
# The print messages are just so you know it's doing something!
def worker_process(queue, configurer):
    import time
    from random import choice, random

    configurer(queue)
    name = mp.current_process().name
    print('Worker started: %s' % name)
    for i in range(10):
        time.sleep(random())
        logger = logging.getLogger(choice(LOGGERS))
        level = choice(LEVELS)
        message = choice(MESSAGES)
        logger.log(level, message)
    print('Worker finished: %s' % name)


# Here's where the demo gets orchestrated. Create the queue, create and start
# the listener, create ten workers and start them, wait for them to finish,
# then send a None to the queue to tell the listener to finish.
def demo_logging():
    log_queue = mp.Queue(-1)
    log_listener = mp.Process(target=log_listener_fn, args=(log_queue, listener_configurer))
    log_listener.start()
    workers = []
    for i in range(10):
        worker = mp.Process(target=worker_process, args=(log_queue, configure_log_for_current_process))
        workers.append(worker)
        worker.start()
    for w in workers:
        w.join()
    log_queue.put_nowait(None)
    log_listener.join()


def target():
    raise ValueError('Something went wrong...')


def demo_exception():
    p = Process(target=target)
    print('Starting')
    p.start()
    print('Joining')
    p.join()

    if p.exception:
        error, tb = p.exception
        print('Demo exception:')
        print(tb)


def run_parallel_func(func, args, n_pools, output_identifier=None):
    if output_identifier is not None:
        assert len(args) == len(output_identifier), "output_identifier len should match args length"
    else:
        output_identifier = args

    n_pools = min(n_pools, mp.cpu_count())
    n_jobs = len(args)
    pool = mp.Pool(processes=n_pools)
    job_args = [(func, i + 1, n_jobs, output_identifier[i], arg_x) for i, arg_x in enumerate(args)]
    results = pool.map(_func_arg_helper, job_args)
    return {ky: val for ky, val in zip(output_identifier, results)}


def _func_arg_helper(args):
    f = args[0]
    job_num = args[1]
    job_tot = args[2]
    identifier = args[3]
    f_args = args[4]
    logging.info(f'Running function {f.__name__} for job {job_num}/{job_tot}')
    try:
        return f(*f_args)
    except:
        logging.warning(f"Failed function {f.__name__} with identifier {identifier}")
        return None


if __name__ == '__main__':
    demo_logging()
    demo_exception()
