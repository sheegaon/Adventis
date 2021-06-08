"""
 Utilities for Jupyter and IPython notebooks.
"""
import os


def init():
    """ Run this at the start of any interactive session of IPython to load my defaults.
    """
    from qr_shared_src.reporting_tools.plotting import CB2_COLORS
    import matplotlib
    import logging
    import pandas as pd
    from qr_shared_src.misc_tools.logging_utils import setup_logger
    matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=CB2_COLORS)
    pd.options.display.max_columns = 20
    pd.options.display.width = 200
    setup_logger(console_level=logging.DEBUG)
    if os.getenv('USERNAME') is None:
        if os.getenv('USER') is not None:
            os.environ['USERNAME'] = os.getenv('USER')
        else:
            os.environ['USERNAME'] = os.getenv('USERPROFILE').split('\\')[-1]
    try:
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
    except:
        pass


def set_jupyter_lab_env(nas, data_folder='prod'):
    """ Set a handful of environment variables for use in JupyterLab
    """
    os.environ['DBCACHEDIR'] = os.path.join(nas, 'QuantPerf', 'qrpd')
    if data_folder == 'prod':
        os.environ['PDDATDIR'] = os.path.join(nas, 'QuantResearchProd', 'QR_ProductDesign', 'data')
    else:
        os.environ['PDDATDIR'] = os.path.join(nas, 'QuantPerf', data_folder, 'data')
    os.environ['MKTDATDIR'] = os.path.join(nas, 'InvestmentMarketData')
    os.environ['QRSHARE'] = os.path.join(nas, 'QuantResearch')


def jupyter_notebook_init():
    """ qgrid is no longer available, using D-Tale now
    """
    import importlib.util as imp_util
    import logging
    if imp_util.find_spec('dtale') is None:
        logging.warning("Python environment does not have 'dtale' package. Install using"
                        "'conda install dtale -c conda-forge'.")
    else:
        msg = """
        To use dtale in jupyter notebook, put this at the top:
        import dtale

        # useful function to reduce typing
        def dvw(df: pd.DataFrame):
            print(dtale.show(df))

        # if you want to make it so more than one output can be displayed per cell
        from IPython.core.interactiveshell import InteractiveShell
        InteractiveShell.ast_node_interactivity = 'all' """
        logging.warning(msg)


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def in_ipynb():
    try:
        cfg = get_ipython().config
        if 'IPKernelApp' in cfg.keys():
            return True
        else:
            return False
    except NameError:
        return False


def create_notebook(notebookfile, outputpath, output_name):
    """
    Create notebook template. Data is passed via environment variables

    :param notebookfile: template notebook file which will be run (with path)
    :param outputpath: output directory where to save .ipynb/html
    :param output_name: name of output python notebook/html file
    :return: None
    """
    from runipy.notebook_runner import NotebookRunner
    from nbformat import read, write
    import subprocess

    notebook = read(open(notebookfile), as_version=3)
    r = NotebookRunner(notebook)
    r.run_notebook()
    os.makedirs(outputpath, exist_ok=True)
    filename = outputpath + '//' + output_name + '.ipynb'
    write(r.nb, open(filename, 'w'))
    subprocess.call("jupyter nbconvert " + filename + " --to html")
    return None

