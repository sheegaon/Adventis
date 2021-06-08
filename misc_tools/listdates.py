"""
Module to return valid trading dates with knowledge of a lord abbett maintained holiday calendar.

Note that the current holiday calendar only contains holidays in this calendar year and the one prior,
so if you are testing dates outside those two calendar years, you will not capture holidays properly.

Stuff you can do:

    Compute trading days between two dates.  Note for the US calendar, you can go back much further than just 1 year.
    There are named holidays in the file back to 20030101, and holidays are listed back to 19850101.

    dg = trading_days_between(date_from_yyyymmdd('20100628'), date_from_yyyymmdd('20160801'), country='US')
    dts = [yyyymmdd(x) for x in dg]

    Get just the month end dates:
    dg2 = monthends_between(date_from_yyyymmdd('20100628'), date_from_yyyymmdd('20160801'), country='US')
    dt_mth_ends = [yyyymmdd(x) for x in dt_mth_ends]

    d = today()
    print("Today:",yyyymmdd(d))
    print("   Month End?", is_monthend_date(d))
    print("   Trading Day?", (not is_non_trading_day(d)))
    print("   Holiday?", is_market_holiday(d))
    d = yesterday()
    print("Yesterday",yyyymmdd(d))
    print("   Month End?", is_monthend_date(d))
    print("   Trading Day?", (not is_non_trading_day(d)))
    print("   Holiday?", is_market_holiday(d))
    d = day_before(yesterday())
    print("Before That",yyyymmdd(d))
    print("   Month End?", is_monthend_date(d))
    print("   Trading Day?", (not is_non_trading_day(d)))
    print("   Holiday?", is_market_holiday(d))

    print()
    print("Last 10 trading days:")
    for d in trading_days_in_range(today(), 0, 10):
        print("Date:",yyyymmdd(d))
        print("   Month End?", is_monthend_date(d))
        print("   Day of Week:",d.strftime("%A"))
        print()

    print("Last 10 monthend dates:")
    for d in monthends_in_range(today(), 0, 10):
        print("Date:",yyyymmdd(d))
        print("   Month End?", is_monthend_date(d))
        print("   Day of Week:",d.strftime("%A"))
        print()

    print("Next 10 monthend dates:")
    for d in monthends_in_range(today(), 0, -10):
        print("Date:",yyyymmdd(d))
        print("   Month End?", is_monthend_date(d))
        print("   Day of Week:",d.strftime("%A"))
        print()

    print("Next 10 monthend dates, in France:")
    for d in monthends_in_range(today(), 0, -10, country="FR"):
        print("Date:",yyyymmdd(d))
        print("   Month End?", is_monthend_date(d,country="FR"))
        print("   Day of Week:",d.strftime("%A"))
        print()

    print( "Prior Monthend:", yyyymmdd(n_monthends_before(today(),1)))
    print( "Before That:", yyyymmdd(n_monthends_before(today(),2)))
    print()

    print( "Next Monthend:", yyyymmdd(n_monthends_after(today(),1)))
    print( "after That:", yyyymmdd(n_monthends_after(today(),2)))
    print()

    print( "Last Close:", last_close_date())
    print( "Before That:", yyyymmdd(n_trading_days_before(today(),2)))
    print()

    print( "Next Close:", yyyymmdd(n_trading_days_after(today(),0)))
    print( "after That:", yyyymmdd(n_trading_days_after(today(),1)))
    print()

    print("Testing proc_dateargs")
    for d in proc_dateargs( range(50,52), [], (), "20150925", "20100909", "0",1,2,3,"4", ["20", 21, 22, "23", 24] ):
        print("Date:",d)

"""
import argparse
import os.path
from datetime import datetime, timedelta
import math
import pytz
import csv
import sys

# when we say "today" we mean 6am NYC time
_nyc_tz = pytz.timezone('America/New_York')
_now = datetime.now(tz=_nyc_tz)
_today = _nyc_tz.localize(datetime(_now.year, _now.month, _now.day, 6, 0, 0, tzinfo=_nyc_tz).replace(tzinfo=None), is_dst=None)

# Lord Abbett holiday calendar.
_holidays = {}
try:
    with open(os.path.abspath(os.environ["MKTDATDIR"]+'/FactSet/FACTSET_MARKET_HOLIDAYS.csv')) as holidayfile:
        for row in csv.DictReader(holidayfile):
            this_ctry = row["CountryCode"]
            this_date = row["Date"]
            _holidays[(this_ctry, this_date)] = True
            # catch US new year's and call that a holiday for "ALL"
            # Note, catching 12/31 is not strictly necessary because the convention thus far has been that
            # if new years day falls on a Saturday, then there is no observed holiday.
            if this_ctry == "US" and (this_date[4:] == "1231" or this_date[4:] == "0101" or this_date[4:] == "0102"):
                _holidays[("ALL",this_date)] = True

except Exception as e:
    print("ERROR, something went wrong loading holiday calendar:", e, file=sys.stderr)


def prev_busday_month_end(indate):
    #
    if isinstance(indate, str):
        indate = date_from_yyyymmdd(indate)

    if is_monthend_date(indate):
        return yyyymmdd(yesterday())
    else:
        return yyyymmdd(n_monthends_before(indate, 1))


def require_holiday_calendar():
    """ Raise an exception if the holiday calendar didn't load

    :return: True, or don't return at all
    """
    if not _holidays:
        raise FileNotFoundError("ERROR: Holiday calendar is required but not available")
    else:
        return True


def dayshift(date, days):
    assert isinstance(date, datetime)
    # I don't really know if I am doing this right but it is to handle DST transitions. See some useful info here:
    # http://stackoverflow.com/questions/441147/how-can-i-subtract-a-day-from-a-python-date
    return _nyc_tz.localize(date.replace(tzinfo=None) - timedelta(days), is_dst=None)


def day_before(date):
    return dayshift(date, 1)


def day_after(date):
    return dayshift(date, -1)


def today():
    # .replace() is called to return a copy of _today, not a reference to the actual _today object
    return _today.replace()


def yesterday():
    return day_before(_today)


def tomorrow():
    return day_after(_today)


def ymd_from_date(date):
    return date.year, date.month, date.day


def date_from_ymd(year, month, day):
    return _today.replace(year=int(year), month=int(month), day=int(day))


def yyyymmdd(date):
    return date.strftime("%Y%m%d")


def date_parts(dstring):
    return dstring[:4], dstring[4:6], dstring[6:]


def date_from_yyyymmdd(dstring):
    try:
        return date_from_ymd(*date_parts(dstring))
    except (ValueError, TypeError):
        pass
    raise TypeError("Tried to generate date from yyymmdd but passed incompatible type:", type(dstring))


def is_weekend(date, country="US"):
    # in weekday(), friday is 4, saturday is 5 and sunday is 6
    if country == "IS":
        # in israel, a weekend is friday/saturday as far as I know (mkt open only Sun-Thu)
        return date.weekday() == 4 or date.weekday() == 5
    else:
        # everywhere else a weekend is saturday/sunday as far as I know (mkt open Mon-Fri)
        return date.weekday() > 4


def is_market_holiday(date, country="US"):
    return _holidays.get((country, yyyymmdd(date)), False)


def is_non_trading_day(date, country="US"):
    return is_weekend(date, country=country) or is_market_holiday(date, country=country)


def days_between(date1, date2, skiptest):
    #print("Date1: {} Date2: {}".format(date1, date2))
    # figure out increment direction by checking if date1 is before or after date2
    diff = date1-date2
    inc = int(math.copysign(1, diff.days))
    date = date1
    # skip to the first valid date between date1 and date2
    while skiptest(date):
        date = dayshift(date, inc)

    # start yielding dates until we pass date2
    diff = date-date2
    while inc* diff.days >= 0:
        #print( "Date: {} Diff: {}".format(date,diff.days))
        yield date
        date = dayshift(date, inc)
        while skiptest(date):
            date = dayshift(date, inc)
        diff=date-date2


def days_in_range(date, start, end, skiptest):
    # first shift the date to start
    # print("- IN days_in_range - Reference date is", yyyymmdd(date))
    inc = int(math.copysign(1, start))
    for i in range(0, start, inc):
        #print("- IN days_in_range -    Skipping To Start ({0}), At {1}".format(start,i))
        date = dayshift(date, inc)
        while skiptest(date):
            date = dayshift(date, inc)

    # now yield the date range from start to end
    inc = int(math.copysign(1, (end - start)))
    # print("- IN days_in_range - Start date is", yyyymmdd(date))
    # print("- IN days_in_range - Range is (START {0}, END {1}, INC {2})".format(start,end,inc))
    for i in range(start, end, inc):
        #print("- IN days_in_range -    Iterating From Start ({0}) To End ({1}), At {2}, day increment = {3}".format(start,end,i,inc))
        date = dayshift(date, inc)
        while skiptest(date):
            date = dayshift(date, inc)
        #print("- IN days_in_range -    Next date generated is", yyyymmdd(date))
        yield date


def trading_days_between(date1, date2, country="US"):
    yield from days_between(date1, date2, lambda d: is_non_trading_day(d, country=country))


def trading_days_in_range(date, start, end, country="US"):
    # use "yield from" to delegate to a function that is a generator already (see PEP 380)
    yield from days_in_range(date, start, end, lambda d: is_non_trading_day(d, country=country))


def n_trading_days_before(date, n, country="US"):
    # make a one element generator and run next() on it to return the single value
    return next(trading_days_in_range(date, int(n)-1, int(n), country=country))


def n_trading_days_after(date, n, country="US"):
    # make a one element generator and run next() on it to return the single value
    return next(trading_days_in_range(date, -int(n)-1, -int(n), country=country))


def is_monthend_date(date, country="US"):
    """A monthend date is a date where the next trading day is in a different month

    :param date: date to check for monthend-ness
    :return:     True or False, depending on check
    """
    d = n_trading_days_after(date, 1, country=country)
    #print("Testing Dates For Monthend: Date({0}) vs Next Trading Date({1})".format(yyyymmdd(date),yyyymmdd(d)))
    return date.month != d.month


def monthends_between(date1, date2, country="US"):
    yield from days_between(date1, date2,
                        lambda d: not is_monthend_date(d, country=country) or is_non_trading_day(d, country=country))


def monthends_in_range(date, start, end, country="US"):
    # use "yield from" to delegate to a function that is a generator already (see PEP 380)
    yield from days_in_range(date, start, end,
                         lambda d: not is_monthend_date(d, country=country) or is_non_trading_day(d, country=country))


def n_monthends_before(date, n, country="US"):
    # make a one element generator and run next() on it to return the single value
    return next(monthends_in_range(date, int(n)-1, int(n), country=country))


def n_monthends_after(date, n, country="US"):
    # make a one element generator and run next() on it to return the single value
    return next(monthends_in_range(date, -int(n)-1, -int(n), country=country))


def last_close_date(country="US"):
    """ return the date of the last market close prior to today, in yyyymmdd format
    :param country: the courntry holiday calendar to use (defaults to US)
    :return: the last close date as YYYYMMDD
    """
    return yyyymmdd(n_trading_days_before(today(), 1, country=country))


def _procdate(d, country="US"):
    """Produce a date either from something castable to an integer or a YYYYMMDD date.
    If integer, interpret as a number of trading days before today, and return the YYYYMMDD string corresponding to that.
    If YYYYMMDD string, just return the string unchanged.

    This is meant to be a private method only called from proc_dateargs.

    :param d: the number of days or raw YYYYMMDD date
    :return: a YYYYMMDD string
    """
    # print("   _procdate: handling ", d)
    if d == 0 or d == "0":
        # handle 0 as a simple special case
        return yyyymmdd(today())
    else:

        # print("   _procdate:    not zero. checking yyyymmdd")
        try:
            # check for valid YYYYMMDD string:
            date_from_yyyymmdd(d)
            return d
        except TypeError:
            pass

        # print("   _procdate:    not yyyymmdd. checking int")
        # finally try to handle as n trading days ago:
        return yyyymmdd(n_trading_days_before(today(), int(d), country=country))


def proc_dateargs(*args, country="US"):
    """Generator for a list of YYYYMMDD date strings, one for each passed in argument.  The args are processed
    in order by the internal _procdate function.  Any object type that can be handled by that function will be.
    If args itself contains things that are iterable, then we iterate over them and pass the elements one by one
    to _procdate.  For example, proc_dateargs(1,2,3,4) is equivalent to proc_dateargs(1,range(2,5)).

    **Note**
    For a long list of dates, proc_dateargs is very slow. Instead, if you are trying to produce a
    listing of sequential dates, use trading_days_in_range()

    :param args: things to try to turn into dates
    :return: yields a YYYYMMDD string for each arg
    """
    for arg in args:
        # print("proc_dateargs: handling", arg)
        if isinstance(arg, str):
            yield _procdate(arg, country=country)
        else:
            try:
                # if this is an iterable but not a string, iterate
                # print("proc_dateargs: trying as iter {0}, {1}".format(arg,iter(arg)))
                for a in iter(arg):
                    # print("proc dateargs: iterating to element", a)
                    yield _procdate(a, country=country)
                # signal that the arg has been handled via iteration, by setting it to an empty list.
                # this is safe because empty lists would be a no-op anyway, even if they are explicitly passed
                # print("proc_dateargs got outside iter loop", arg)
                arg = ()
            except TypeError:
                # silently ignore TypeError and let arg be handled below as a single _procdate call
                pass

            # not iterable, not a string.  Try anyway.  Empty list is signal not even to try
            if arg != ():
                yield _procdate(arg, country=country)


def yield_dates(args):
    """
    given an already processed args list (assumed to contain all the args from main
    and possibly others, which are ignored), yield a list of dates


    :param args:
    :return:
    """

    if args.d:
        # if the -d flag is given, interpret num_days and any -y flag as being YYYYMMDD real dates, not relative
        # this means we use things like dates_between and monthends_between as date generators instead of dates_in_range
        if args.m:
            date_generator = monthends_between
        else:
            date_generator = trading_days_between

        # args.y is the stopping date so make that today unless specified
        if args.y == 0:
            args.y = yyyymmdd(today())

        # now fully parameterize the date generator for the forward and reverse cases
        if args.r:
            date_generator = date_generator(date_from_yyyymmdd(str(args.num_days)), date_from_yyyymmdd(str(args.y)), country=args.c)
        else:
            date_generator = date_generator(date_from_yyyymmdd(str(args.y)), date_from_yyyymmdd(str(args.num_days)), country=args.c)

    else:
        # the -d flag was not given so we interpret num_days and args.y in the usual way as days relative to the last
        # business day close
        if args.num_days < 0:
            args.r = not args.r

        refdt = today()
        if args.m:
            date_generator = monthends_in_range
            if not is_monthend_date(refdt,country=args.c):
                refdt = n_monthends_before(refdt, 1, country=args.c)
        else:
            date_generator = trading_days_in_range

        # if there is a stride we multiply the num_days
        args.num_days = args.num_days * args.s

        # now fully parameterize the date generator for the forward and reverse cases
        if args.r:
            date_generator = date_generator(refdt, args.num_days, args.y, country=args.c)
        else:
            date_generator = date_generator(refdt, args.y-1, args.num_days-1, country=args.c)

    # skip implements the stride length
    skip = 0

    # yield the dates requested
    for d in date_generator:
        if not skip:
            yield d
        skip = (skip + 1) % args.s


def main(argv):
    """
    handle commandline arguments and produce a date listing to stdout
    :param argv:
    :return:
    """
    parser = argparse.ArgumentParser(
        description=
        ("Produce a list of valid trading dates going back in time num_days. "
         "Stop either at today's date or a number of trading days prior to today, depending on the value of the -y option."))

    # handle args
    parser.add_argument('-m', help='list month end dates only', action='store_true')
    parser.add_argument('-r', help='reverse the list', action='store_true')
    parser.add_argument('-d', help='interpret num_days and Y as real dates in YYYYMMDD format, '
                                   'rather than relative ones', action='store_true')
    parser.add_argument('-y', help="stop generating dates Y days before today (defaults to 0)",
                        type=int, default=0)
    parser.add_argument('-c', help="use specified 2 letter country code for holiday calendar (defaults to %(default)s)",
                        default="US")
    parser.add_argument('-s', help='stride to use for listing dates (defaults to 1)', type=int, default=1)
    parser.add_argument('num_days', help='an integer number of dates to list.  Negative for future dates', type=int)
    args = parser.parse_args()

    for d in yield_dates(args):
        print(yyyymmdd(d))


if __name__ == '__main__':
    main(sys.argv.copy())
