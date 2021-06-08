"""
 Code to get data from our relational databases.
"""
import logging


def get_conn(conn_str, **kwargs):
    if conn_str == 'FIPS':
        return get_db_connection(conn_str)
    return get_sqla_connection(conn_str, **kwargs)


def get_sqla_connection(db='ISIS', **kwargs):
    """
    This is the connection one should pass to pandas.read_sql or read_sql_query.

    :param db: 'ISIS' (default) or other database mnemonic
    :param kwargs: any additional kwargs you want to pass to the sqlalchemy engine that creates this connection.
                   A popular one would be something like echo=True for verbose logging
    :return: sqlalchemy connection to the specified db
    """
    import sqlalchemy as sa
    from sqlalchemy import event
    from sqlalchemy import exc
    import os

    # Make an sqlalchemy engine from the connection string.
    connstr = _get_conn_str(db)
    if 'postgres' in connstr:
        engine = sa.create_engine(connstr, **kwargs)
    else:
        # Need to url quote the string to make it work.
        # See here: http://docs.sqlalchemy.org/en/latest/dialects/mssql.html#module-sqlalchemy.dialects.mssql.pyodbc
        import urllib
        engine = sa.create_engine(
            "mssql+pyodbc:///?odbc_connect=%s" % urllib.parse.quote_plus(connstr), **kwargs)

    @event.listens_for(engine, "connect")
    def connect(dbapi_connection, connection_record):
        connection_record.info['pid'] = os.getpid()

    @event.listens_for(engine, "checkout")
    def checkout(dbapi_connection, connection_record, connection_proxy):
        pid = os.getpid()
        if connection_record.info['pid'] != pid:
            connection_record.connection = connection_proxy.connection = None
            raise exc.DisconnectionError(
                f"Connection record belongs to pid {connection_record.info['pid']}, "
                f"attempting to check out in pid {pid}")

    conn_obj = None
    try:
        conn_obj = engine.connect()
    except:
        logging.exception(f'db={db}')
    return conn_obj


def get_db_connection(db='IDA_QPD'):
    """
    Not recommended for use with pandas.read_sql or read_sql_query.  Use get_sqla_connection instead because it uses
    SQLAlchemy.
    :param db: 'IDA_QPD' (default), 'QUANT', 'EQRISK', etc
    :return: pyodbc connection to the specified db
    """
    import pyodbc
    conn_obj = None
    conn_str = _get_conn_str(db)
    try:
        conn_obj = pyodbc.connect(conn_str)
        if 'TIBCO DV ODBC Driver' in conn_str:
            conn_obj.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
            conn_obj.setencoding(encoding="utf-8")
            conn_obj.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-32')
    except pyodbc.OperationalError:
        logging.debug(f'db={db} OperationalError')
    except pyodbc.InterfaceError:
        logging.info(f'db={db} InterfaceError. FIPS DB driver not installed (but you may not need it).')
    return conn_obj


def _get_conn_str(db):
    """
    :param db: 'ISIS' (default), 'QUANT', or 'EQRISK'
    :return: a pyodbc-style connection string to use for the specified DB
    """
    import sys

    odbc_driver = '{SQL Server Native Client 11.0}'
    comp_driver = '{TIBCO(R) Data Virtualization 7.0}'
    if sys.platform == 'linux':
        odbc_driver = '{ODBC Driver 17 for SQL Server}'
        comp_driver = '{TIBCO DV ODBC Driver}'

    if db == 'ISIS':
        uid = 'quant'
        pwd = 'pquant'
        server = r'SQLPISIS\SQLPISIS'
        database = 'ISIS'
    elif db == 'ISIS-QA':
        uid = 'isis'
        pwd = 'qisis'
        server = r'SQLQISIS\SQLQISIS'
        database = 'ISIS'
    elif db == 'QUANT':
        uid = 'quant'
        pwd = 'pquant'
        server = r'SQLPQUANT\SQLPQUANT'
        database = 'QUANT'
    elif db == 'EQRISK':
        uid = 'QUANT_READONLY'
        pwd = 'readonly'
        server = r'SQLPEQRISK\SQLPEQRISK'
        database = 'EQRISK'
    elif db == 'SPX':
        uid = 'spxreader'
        pwd = 'r3ad3rSPX'
        server = r'SQLPSPX\SQLPSPX'
        database = 'SPXDATA'
    elif db == 'IDA_QPD':
        uid = 'product_design'
        pwd = 'pproduct_design'
        server = r'SQLPIDA\SQLPIDA'
        database = 'PRDTDESIGN'
    elif db == 'IDA_QPD-QA':
        uid = 'product_design'
        pwd = 'qproduct_design'
        server = r'SQLQIDA\SQLQIDA'
        database = 'PRDTDESIGN'
    elif db == 'IDA_STAGING':
        uid = 'product_user'
        pwd = 'pproduct_user'
        server = r'SQLPIDA\SQLPIDA'
        database = 'STAGING'
    elif db == 'IDA_RISK':
        uid = 'product_user'
        pwd = 'pproduct_user'
        server = r'SQLPIDA\SQLPIDA'
        database = 'RISK'
    elif db == 'IDA_RISK-QA':
        uid = 'ida'
        pwd = 'iD4'
        server = r'SQLQIDA\SQLQIDA'
        database = 'RISK'
    elif db == 'FIPS':
        uid = 'firs_fips_product_design'
        pwd = 'pfirs_fips_product_design'
        server = r'COMPP6-EIOV'
        datasource = 'FIPS_V001'
        domain = 'composite'
    elif db == 'FID-QA':
        uid = 'fidread'
        pwd = 'qFidread0'
        server = 'lxpgFIDQ1-jitv'
        database = 'bloomberg'
    elif db == 'FID-QA-staging':
        uid = 'fiddbo'
        pwd = 'qFiddbo0'
        server = 'lxpgFIDQ1-jitv'
        database = 'staging'
    elif db == 'FID-staging':
        uid = 'fidread'
        pwd = 'pFidread0'
        server = 'lxpgFIDP1-eitv'
        database = 'staging'
    elif db == 'FID':
        uid = 'fidread'
        pwd = 'pFidread0'
        server = 'lxpgFIDP1-eitv'
        database = 'bloomberg'
    elif db == 'LCAPX-DEV':
        uid = 'lcapx'
        pwd = 'dlcapx'
        server = 'pgdlcapx'
        database = 'lcapx'
    elif db == 'LCAPX-QA':
        uid = 'lcapx_read'
        pwd = 'qlcapxread'
        server = 'lxpgqlcapx-jitv'
        database = 'lcapx'
    elif db == 'LCAPX':
        uid = 'lcapx_irs'
        pwd = 'lcapxpirs'
        server = 'lxpgplcapx-eitv'
        database = 'lcapx'
    else:
        raise KeyError(
            "Unknown database: '{}'.  Known db's are ISIS, ISIS-QA, QUANT, EQRISK, SPX, IDA_QPD, IDA_STAGING, IDA_RISK,"
            " IDA_RISK-QA".format(db))

    if 'SQL' in server:
        connstr = f"APP=my_app_name;DRIVER={odbc_driver};SERVER={server};DATABASE={database};UID={uid};PWD={pwd}"
    elif 'COMP' in server:
        connstr = f"APP=my_app_name;DRIVER={comp_driver};SERVER={server};DOMAIN={domain};UID={uid};PWD={pwd};" \
                  f"DATASOURCE={datasource}"
    else:
        connstr = f"postgresql+psycopg2://{uid}:{pwd}@{server}/{database}"

    return connstr
