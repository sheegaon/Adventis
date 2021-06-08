"""
 Miscellaneous tools that are helpful, but don't fit neatly into any other category.
 Suggested import as mu
"""
from qr_shared_src.misc_tools.date_utils import *
from qr_shared_src.misc_tools.file_utils import *
from qr_shared_src.misc_tools.function_library import *
from qr_shared_src.misc_tools.logging_utils import *
from qr_shared_src.misc_tools.notebook_utils import *
import qr_shared_src.misc_tools.pandas_utils


def copy_figure(fig=None):
    """
    Copy a matplotlib figure to the clipboard.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, default: gcf()
        The figure to be copied to clipboard.

    """
    import io
    from PyQt5.Qt import QApplication, QImage

    if fig is None:
        import matplotlib.pyplot as plt

        fig = plt.gcf()
    axes = fig.get_axes()
    rt_edg = 1
    for ax in axes:
        if len(ax.texts) > 0:
            rt_edg = 0.96
    fig.tight_layout(rect=(0, 0, rt_edg, 1))
    buf = io.BytesIO()
    fig.savefig(buf)
    QApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))
    buf.close()


def is_iterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def weighted_percentile(values, percentiles, sample_weight=None, values_sorted=False, old_style=False):
    """
    Very close to numpy.percentile, but supports weights.

    NOTE: percentiles should be in [0, 1]!

    :param values: numpy.array with data
    :param percentiles: array-like with many percentiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed percentiles.
    """
    values = np.array(values)
    percentiles = np.array(percentiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(percentiles >= 0) and np.all(percentiles <= 1), 'percentiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]
    weighted_percentiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with np.percentile
        weighted_percentiles -= weighted_percentiles[0]
        weighted_percentiles /= weighted_percentiles[-1]
    else:
        weighted_percentiles /= np.sum(sample_weight)
    return np.interp(percentiles, weighted_percentiles, values)


def flatten(list_in):
    """Flatten a list of lists into a single list."""
    return [item for sublist in list_in for item in sublist]


def total_return(series_return):
    """Calculate total return of an entire series"""
    return 100 * ((1 + series_return * 0.01).prod() - 1)


def cum_return(series_return, date_range=None):
    """
    Calculate cumulative return series

    :param series_return: input series
    :param date_range: optional argument to calculate cumulative returns over a limited range of dates
    :return: cumulative return series
    """
    if date_range is None:
        return 100 * ((1 + series_return * 0.01).cumprod() - 1)
    else:
        series_out = series_return * np.nan
        series_out[date_range] = 100 * ((1 + series_return[date_range] * 0.01).cumprod() - 1)
        return series_out


def multi_period_return(series_return, delta: int, ident=None):
    """
    Construct multi-period returns from a pandas Series, e.g. monthly to quarterly

    :param series_return: input series
    :param delta: time aggregation unit
    :param ident: identifier variable series for long-form datasets
    :return: output series
    """
    series_cp = (1 + series_return * 0.01).cumprod()
    s_out = 100 * (series_cp / series_cp.shift(delta) - 1)
    s_out.iloc[delta - 1] = 100 * (series_cp.iloc[delta - 1] - 1)
    if ident is not None:
        s_out[ident != ident.shift(delta - 1)] = np.nan
    return s_out


def linreg(x_mat, y, add_constant=False):
    """Quick linear regression."""
    if isinstance(x_mat, pd.Series):
        x_mat = x_mat.to_frame()
    if isinstance(x_mat, pd.DataFrame):
        x_mat = x_mat.values
    if add_constant:
        if x_mat.size == 0:
            return np.array((1 + x_mat.shape[1]) * [0.0])
        from statsmodels.api import add_constant

        x_mat = add_constant(x_mat, has_constant='add')
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.values
    return np.linalg.pinv(x_mat).dot(y)


def linfit(x_mat, params, add_constant=False):
    """Get fit from a linear regression."""
    if add_constant:
        if isinstance(x_mat, pd.Series):
            x_mat = x_mat.to_frame()
        if isinstance(x_mat, pd.DataFrame):
            x_mat.insert(loc=0, column='const', value=1.0)
        else:
            from statsmodels.api import add_constant

            x_mat = add_constant(x_mat, has_constant='add')
    return x_mat.dot(params)


def linreg_plot(x, y):
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    # Running the linear regression
    model = sm.OLS(y, sm.add_constant(x)).fit()
    a = model.params[0]
    b = model.params[1]

    # Return summary of the regression and plot results
    X2 = np.linspace(x.min(), x.max(), 100)
    Y_hat = X2 * b + a
    plt.scatter(x, y, alpha=0.3)  # Plot the raw data
    plt.plot(X2, Y_hat, 'r', alpha=0.9)  # Add the regression line, colored in red
    plt.xlabel('x value')
    plt.ylabel('y value')
    return model.summary()


def renorm_wt(mv_series, date_series=None, pct=True):
    """
    Renormalize a series of market values or weights to sum to 100pct by weight.

    :param mv_series: ps.Series
        Market values OR weights.
    :param date_series: pd.Series (or None)
        Dates with same index as above.
    :param pct: return weights in percent, 0-100, if True, 0 to 1 if False

    :return: pd.Series
        Weights.
    """
    scale = 100.0 if pct else 1.0
    if date_series is None:
        mv_series.fillna(0, inplace=True)
        return scale * mv_series / mv_series.sum()
    else:
        df1 = pd.concat([mv_series, date_series], axis=1)
        df1[mv_series.name].fillna(0, inplace=True)
        return scale * df1[mv_series.name] / df1.groupby(date_series.name)[mv_series.name].transform(np.sum)


def make_dash_table(df):
    """Return a dash definition of an HTML table for a Pandas dataframe"""
    import dash_html_components as html

    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table


def sql_id_liststr(idlist):
    return ','.join(["'" + str(x) + "'" for x in idlist])


def sendmail(to_list, from_address, subject, body, files=None, mimetype='plain', server='postman.lordabbett.com'):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.application import MIMEApplication

    if isinstance(to_list, str):
        to_list = [to_list]
    for to in to_list:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = from_address
        msg['To'] = to
        msg.attach(MIMEText(body, mimetype))
        # attach files
        if files is not None:
            for f in files or []:
                with open(f, "rb") as fil:
                    part = MIMEApplication(fil.read(), Name=os.path.basename(f))
                # After the file is closed
                part['Content-Disposition'] = 'attachment; filename="%s"' % os.path.basename(f)
                msg.attach(part)
        # send email
        s = smtplib.SMTP(server)
        s.sendmail(from_address, to, msg.as_string())
        s.quit()
        s.close()
        logging.info('Email sent to ' + to)


def bulk_upload_df_to_db(df_in, table_name, sql_conn, delete_where=''):
    """
    Upload a DataFrame to a SQL table using bulk upload via CSV

    :param df_in: The DataFrame to upload
    :param table_name: The SQL table name
    :param sql_conn: SQLAlchemy connection object
    :param delete_where: Limit the DELETE FROM statement appending this statement after WHERE
    """
    import pathlib

    univ = df_in.universe.iloc[0]
    as_of_date = df_in.as_of_date.iloc[0]
    print(f'Detecting column format of DB table {table_name}')
    df_format = pd.read_sql_query(f'SELECT TOP 1 * FROM {table_name}', sql_conn)
    assert (
        len(set(df_in.columns) - set(df_format.columns)) == 0
    ), f'Found columns in input table not in {table_name}: {set(df_in.columns) - set(df_format.columns)}'
    assert (
        len(set(df_format.columns) - set(df_in.columns)) == 0
    ), f'Found columns in {table_name} not in input table: {set(df_format.columns) - set(df_in.columns)}'
    df_in = df_in[df_format.columns]
    print(f'Outputting to CSV and bulk inserting to DB table {table_name}')
    tmp = f"bulktemp_{datetime.now():%m%d_%H%M%S}"
    # Note: We have only been given permission to bulk upload files from this folder.
    tmp_file = os.path.join(os.getenv('MKTDATDIR'), 'BAML', 'tempcsv', tmp + '.csv')
    df_in.to_csv(tmp_file, index=False, header=False)
    sql_conn.execute(f"SELECT * INTO #{tmp} FROM {table_name} WHERE 1=2")
    sql_conn.execute(f"BULK INSERT #{tmp} FROM '{pathlib.Path(tmp_file)}' WITH (FIELDTERMINATOR=',')")
    os.remove(tmp_file)
    print(f'Copying {tmp} to {table_name}')
    sql_conn.execute(
        f"DELETE FROM {table_name} WHERE universe='{univ}' AND as_of_date='{as_of_date:%Y%m%d}'" + delete_where
    )
    col_list = ', '.join(df_in.columns)
    sql_conn.execute(f"INSERT INTO {table_name} ({col_list}) SELECT {col_list} FROM #{tmp}")
    sql_conn.execute(f"DROP TABLE #{tmp}")


def arg_handler(argin, input_args, name, args_help=None):
    input_args.update({'debug': False})
    input_args.update({'process_name': name})
    if args_help is None:
        args_help = dict()
    for v in argin[1:]:
        if (v == '?') or (v == '-help') or (v == '-h'):
            print(
                f"Help for {name}\nOptions syntax is \"option_name=option_value\". If option_value contains a\n"
                f"space, the entire expression must be surrounded by double quotes."
            )
            print('\nDefault options:')
            print('----------------\n')
            for opt in input_args:
                print(f"{opt}={input_args[opt]}")
                if opt in args_help:
                    print(f"\t{args_help[opt]}")
                elif opt == 'debug':
                    print(f"\tRun in 'debug' mode; may not be implemented by {name}.")
                elif opt == 'process_name':
                    print("\tThe name assigned to this process in log files and messages.")
                print("")
            print(
                "\nOptions set to False by default may also be set to True with syntax '-option_name'. Setting to\n"
                "True/False/None is not case-sensitive."
            )
            quit()
        if '=' in v:
            vs = v.split('=')
            if vs[1].lower() == 'true':
                input_args.update({vs[0]: True})
            elif vs[1].lower() == 'false':
                input_args.update({vs[0]: False})
            elif vs[1].lower() == 'none':
                input_args.update({vs[0]: None})
            elif not pd.isna(pd.to_numeric(vs[1], errors='coerce')):
                input_args.update({vs[0]: pd.to_numeric(vs[1])})
            else:
                input_args.update({vs[0]: vs[1]})
        elif v[0] == '-':
            input_args.update({v[1:]: True})
        elif pd.to_datetime(v, errors='coerce') is not pd.NaT:
            input_args.update({'input_date': pd.to_datetime(v)})
        elif (v == 'debug') or (v == '-debug'):
            input_args.update({'debug': True})
        else:
            print(f"Improper usage of {name}. Use 'python {name.split('/')[-1]} ?' for help.")
            quit()
    return input_args


def find_duplicates(lst):
    seen = {}
    dupes = []
    for x in lst:
        if x not in seen:
            seen[x] = 1
        else:
            if seen[x] == 1:
                dupes.append(x)
            seen[x] += 1
    return dupes


def total_size(obj, handlers=None, verbose=False):
    """
    Returns the approximate memory footprint of an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}
    """
    from sys import getsizeof, stderr
    from itertools import chain
    from collections import deque

    def dict_handler(d):
        chain.from_iterable(d.items())

    try:
        from reprlib import repr
    except ImportError:
        pass
    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    handlers = {} if handlers is None else handlers
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)
        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    sz = sizeof(obj)
    if sz < 1024:
        logging.info(f'Total size: {sz:.1f}B')
    elif sz < 1024 ** 2:
        logging.info(f'Total size: {sz/1024:.1f}KB')
    elif sz < 1024 ** 3:
        logging.info(f'Total size: {sz/1024**2:.1f}MB')
    else:
        logging.info(f'Total size: {sz/1024**3:.1f}GB')
    return sz


def create_indicators(test_series, value_list):
    """Creates a DataFrame with indicator variables based on test_series for all values in value_list."""
    df_out = pd.DataFrame(index=test_series.index)
    for v in value_list:
        df_out[v.lower()] = test_series == v
    return df_out


def create_bucket_weights(test_series, value_list, formatter):
    """Creates a DataFrame with bucket weight variables based on test_series for all values in value_list."""
    df_out = pd.DataFrame(index=test_series.index)
    for v in value_list:
        df_out[formatter.format(v)] = test_series < v
    return df_out


def copy_sheet(ws1, ws2):
    """Copies the contents of an entire worksheet from one Excel workbook to another. Does not copy formatting."""
    mr = ws1.UsedRange.Rows.Count
    mc = ws1.UsedRange.Columns.Count

    for i in range(1, mr + 1):
        for j in range(1, mc + 1):
            ws2.Cells(i, j).Value = ws1.Cells(i, j).Value


def concat_wo_dup_col(df1, df2):
    import pandas as pd

    cols_to_use = df2.columns.difference(df1.columns)
    return pd.concat([df1, df2[cols_to_use]], axis=1)
