"""
 Pandas DataFrame and Series utility methods.
"""
import pandas as pd


@pd.api.extensions.register_series_accessor("la")
class SeriesUtils:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        return True

    def is_month_end(self):
        """
        Return a Series with boolean value for whether the date is a month-end date.

        """
        from qr_shared_src.misc_tools.misc_utils import is_month_end
        return is_month_end(self._obj)

    def mtd_to_idx(self, inplace=False):
        from qr_shared_src.misc_tools.misc_utils import mtd_to_idx
        out_obj = mtd_to_idx(self._obj)
        if inplace:
            self._obj = out_obj
        else:
            return out_obj

    def idx_to_mtd(self, inplace=False):
        from qr_shared_src.misc_tools.misc_utils import idx_to_mtd
        out_obj = idx_to_mtd(self._obj)
        if inplace:
            self._obj = out_obj
        else:
            return out_obj

    def idx_to_set(self, index_set=None, inplace=False):
        """
        Convert a return index (cumulative return series, expressed as a factor) to 1-periods differences over the
        set of indices in index_set

        Parameters
        ----------
        self : Series
            The return index series
        index_set : set, default index of self
            Set of indices corresponding to indices in self
        inplace : bool, default False
            Update the data in place if possible.

        Returns
        -------
        DataFrame
            A 1-period return series, expresed as a percentage (e.g. 1% = 1.0)
        """
        from qr_shared_src.misc_tools.misc_utils import idx_to_set
        out_obj = idx_to_set(self._obj, index_set=index_set)
        if inplace:
            self._obj = out_obj
        else:
            return out_obj

    def convert_matlab_date(self, inplace=False):
        from datetime import datetime, timedelta
        out_obj = self._obj.apply(lambda x: datetime.fromordinal(int(x)) + timedelta(days=x % 1) - timedelta(days=366))
        if inplace:
            self._obj = out_obj
        else:
            return out_obj


@pd.api.extensions.register_dataframe_accessor("la")
class DataFrameUtils:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        return True

    def show(self, tmpdir=None, name=None, ed='e', s=',', idx=True):
        """
        Saves an input dataframe to a temporary file, and shows it in excel | ultra edit | notepad | chrome

        Parameters
        ----------
        tmpdir
            temporary directory, if None, will use tempfile module
        name
            root of file name (suffix .csv will be added), if tmpdir is specified
        ed : {'e', 'u', 'n', 'c'}
            editor to use e: excel(default), u: ultra-edit, n: notepad, c: chrome
        s : {',', '\t'}
            separator, ','(default), '\t' for tab-separated, etc
        idx : bool, default = True
            show dataframe index in output file
        """
        show(self._obj, tmpdir, name, ed, s, idx)

    def info(self, print_column_string=False):
        """
        Prints out the column names and index name of a dataframe nicely along with the number of nans/0s/min/max.

        Parameters
        ----------
        print_column_string : bool, default False
            Print out a string with all column names
        """
        info(self._obj, as_string=print_column_string)

    def total_size(self, verbose=False):
        """
        Returns the approximate memory footprint of all contents.

        Parameters
        ----------
        verbose

        Returns
        -------
        int
        """
        from qr_shared_src.misc_tools.misc_utils import total_size
        return total_size(self._obj, verbose=verbose)

    def to_code(self):
        """
        Returns code which generates the DataFrame.
        """
        import numpy as np
        index = self._obj.index
        columns = self._obj.columns
        data = np.array2string(self._obj.values, separator=', ')
        code = f"pd.DataFrame(index=pd.{index}, columns=pd.{columns}, data={data})"
        print(code)

    def plot_line(self, set_legend=None, add_mean=False, num_format=None, dec_places=None, **kwargs):
        """
        Equivalent to df.plot.line(**kwargs), but adds gridlines and plot notes at the right margin.
        Optionally adds a horizontal mean line

        Parameters
        ----------
            set_legend:
            add_mean: bool
                      add a horizontal mean value line for each plot
            num_format: string
                        number format for value boxes and mean value line text, e.g. {:.1f}
            dec_places: integer
                        number of decimal places, instead of num_format, 2 -> {:.2f}
            **kwargs: arguments to pass to DataFrame.plot.line

        Returns
        -------
            plot object
        """

        import qr_shared_src.reporting_tools.plotting as rtp
        if 'ax' in kwargs.keys():
            ax = kwargs.pop('ax')
        else:
            fig, ax = rtp.create_axes(1)
            ax = ax[0]
        f = self._obj.plot.line(ax=ax, **kwargs)
        ax.grid(color='k', ls='--', lw=1, alpha=0.2)
        if set_legend is not None:
            ax.legend(set_legend)

        if dec_places is not None:
            num_format = "{:." + str(dec_places) + "f}"

        num_format = rtp.val_boxes(ax=ax, num_format=num_format)

        if add_mean:
            rtp.add_mean_to_plot(ax=ax, mean_val=self._obj.mean().values, num_format=num_format)

        return f


def show(df, tmpdir=None, name=None, ed='e', s=',', idx=True):
    """
    Saves an input dataframe to a temporary file, and shows it in excel | ultra edit | notepad | chrome

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to show
    tmpdir : str
        Temporary directory. If None, will use tempfile module.
    name : str
        Root of file name (suffix .csv wil be added), if tmpdir is specified.
    ed : {'e', 'u', 'n', 'c'}, default 'e
        Editor to use. Excel = 'e' Ultra-edit = 'u', Notepad = 'n', Chrome = 'c'
    s : str, default ','
        Separator. '\t' for tab-separated.
    idx : bool, default True
        Show DataFrame index in output file.
    """
    import tempfile
    import os

    if ed == 'c':
        suffix = '.html'
    else:
        suffix = '.csv'

    if tmpdir is not None:
        os.makedirs(tmpdir, exist_ok=True)
        if name is None:
            name = 'df'
        name = os.path.join(tmpdir, name + suffix)
        name = os.path.normpath(name)
        if suffix == '.csv':
            df.to_csv(name, sep=s, index=idx)
        else:
            df.to_html(name)
    else:
        f = tempfile.NamedTemporaryFile(delete=False)
        name = f.name + suffix
        if suffix == '.csv':
            df.to_csv(name, sep=s, index=idx)
        else:
            df.to_html(name)
        f.close()

    if ed == 'e':
        os.system("start excel.exe /e {}".format(name))
    elif ed == 'u':
        os.system("start uedit32 {}".format(name))
    elif ed == 'n':
        os.system("start notepad {}".format(name))
    elif ed == 'c':
        os.system("start chrome {}".format(name))
    return


def info(df, as_string=False):
    """
    Prints out the column names and index name of a dataframe nicely along with the number of nans/0s/min/max.

    :param df : dataframe for which to print the columns.
    :param as_string : prints a row with all the columns in string format to copy paste in python code
    :return: NULL
    """
    from pandas.api.types import is_numeric_dtype, is_datetime64_dtype
    import numpy as np
    import pandas as pd
    print('_'*90)
    cnt_null = df.isna().sum()
    len_null = len(str(max(cnt_null)))
    null_format = '{:>'+str(len_null)+'}'

    dict_zero = {'holder': 0}
    dict_min = {'holder': 0}
    dict_max = {'holder': 0}
    len_zero = len_min = len_max = 0
    if not df.empty:
        if is_numeric_dtype(df.index.dtype):
            col_array = list(df.index)
            dict_zero['index'] = col_array.count(0)
            col_array = np.array(col_array)
            dict_min['index'] = np.min(np.ma.masked_array(col_array, np.isnan(col_array)), axis=0)
            dict_max['index'] = np.max(np.ma.masked_array(col_array, np.isnan(col_array)), axis=0)
            len_zero = max([len_zero, len(str(dict_zero['index']))])
            len_min = max([len_min, len(str(dict_min['index']))])
            len_max = max([len_max, len(str(dict_max['index']))])
        elif is_datetime64_dtype(df.index.dtype):
            dict_zero['index'] = 'N/A'
            dict_min['index'] = df.index.min()
            dict_max['index'] = df.index.max()
            len_zero = max([len_zero, len(str(dict_zero['index']))])
            len_min = max([len_min, len(str(dict_min['index']))])
            len_max = max([len_max, len(str(dict_max['index']))])
        for col in df.columns:
            assert isinstance(df[col], pd.Series), f"Multiple columns with key {col}"
            if is_numeric_dtype(df[col].dtype):
                col_array = list(df[col])
                dict_zero[col] = col_array.count(0)
                col_array = np.array(col_array)
                dict_min[col] = np.min(np.ma.masked_array(col_array, np.isnan(col_array)), axis=0)
                dict_max[col] = np.max(np.ma.masked_array(col_array, np.isnan(col_array)), axis=0)
                len_zero = max([len_zero, len(str(dict_zero[col]))])
                len_min = max([len_min, len(str(dict_min[col]))])
                len_max = max([len_max, len(str(dict_max[col]))])
            elif is_datetime64_dtype(df[col].dtype):
                dict_zero[col] = 'N/A'
                dict_min[col] = df[col].min()
                dict_max[col] = df[col].max()
                len_zero = max([len_zero, len(str(dict_zero[col]))])
                len_min = max([len_min, len(str(dict_min[col]))])
                len_max = max([len_max, len(str(dict_max[col]))])
    zero_format = '{:>'+str(len_zero)+'}'
    min_format = '{:>'+str(len_min)+'}'
    max_format = '{:>'+str(len_max)+'}'

    col_nr_str = str(len(df.columns))
    row_nr_str = str(len(df))
    index_format = '{:>'+str(len(col_nr_str))+'}'
    for count, name in enumerate(df.columns):
        print(index_format.format(str(count))
              + ' | nan: ' + null_format.format(str(cnt_null[name]))
              + ' | 0: ' + zero_format.format(str(dict_zero.get(name, ' '*len_zero)))
              + ' | min: ' + min_format.format(str(dict_min.get(name, ' '*len_min)))
              + ' | max: ' + max_format.format(str(dict_max.get(name, ' '*len_max)))
              + ' | ' + name)
    if 'index' in dict_zero.keys():
        index_name = 'None' if df.index.name is None else df.index.name
        print(index_format.format('index   ') + ' '*(len_null + len(col_nr_str))
              + ' | 0: ' + zero_format.format(str(dict_zero.get('index', ' ' * len_zero)))
              + ' | min: ' + min_format.format(str(dict_min.get('index', ' ' * len_min)))
              + ' | max: ' + max_format.format(str(dict_max.get('index', ' ' * len_max)))
              + ' | ' + index_name)
    max_len = max([len(col_nr_str), len(row_nr_str)])
    print('-' * (18+max_len))
    if as_string:
        col_str = '\''
        for name in df.columns:
            col_str = col_str+name+'\', \''
        if len(col_str) > 2:
            col_str = col_str[:-2]
        else:
            col_str = ''
        print(col_str)
        print('-' * (18+max_len))
    print('--- Columns = ' + col_nr_str + ((max_len-len(col_nr_str)) * ' ') + ' ---')
    print('--- Rows    = ' + row_nr_str + ((max_len-len(row_nr_str)) * ' ') + ' ---')
    print('_'*90)
