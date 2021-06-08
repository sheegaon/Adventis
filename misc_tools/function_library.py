"""
 A library of commonly-used mathematical functions, intuitively parameterized for convenience and
 ease of use in optimization and statistical fitting (a.k.a. machine learning).
"""
import numpy as np


def sigmoid(x, ymin=0., ymax=1., shape=1., center=0.):
    """
    Returns a logistic (sigmoid) function with specified minimum, maximum, shape coefficient, and center.

    Note: slope = shape * (ymax - ymin) / 4

    Parameters
    ----------
    x
    ymin
    ymax
    shape
    center

    Returns
    -------

    """
    # return ymin + (ymax - ymin) / (1 + np.exp(-shape * (x - center)))
    # note expit(x) = 1/(1+exp(-x)), and is faster than np.exp and does not issue overflow warning for large x
    from scipy.special import expit
    return ymin + (ymax - ymin) * expit(shape*(x - center))


def softplus(x):
    """
    Returns the "softplus" function, a softened version of the function f(x) = max(x, 0), for an array of inputs.

    Parameters
    ----------
    x : np.array

    Returns
    -------
    np.array
    """
    out = x
    out[x < 30] = np.log(1 + np.exp(x[x < 30]))
    return out


def double_softplus(x=np.arange(-2, 2, .01), softness=.25, ylevel=0, downkink=-1, upkink=1, slope=1):
    """
    Returns a combination of a softplus and a negative softplus, with some parameters controlling the level,
    softness, and slope.

    Parameters
    ----------
    x : np.array
    softness : float
    ylevel : float
    downkink : float
    upkink : float
    slope : float

    Returns
    -------
    np.array
    """
    plus_up = softplus((x - upkink) / softness)
    plus_down = softplus(-(x - downkink) / softness)
    return ylevel + slope * softness * (plus_up - plus_down)


def piecewise_linear(x, ymin=0., ymax=1., shape=1., center=0.):
    """
    Returns a piecewise-linear function with specified minimum, maximum, shape coefficient, and center.

    Parallels sigmoid() function

    :param x: input data
    :param ymin:
    :param ymax:
    :param shape:
    :param center:
    :return:
    """
    slope = shape * (ymax - ymin) / 4
    return np.minimum(ymax, np.maximum(ymin, (ymax + ymin) / 2 + slope * (x - center)))


def hurst(p):
    """ Source: https://www.quantopian.com/posts/some-code-from-ernie-chans-new-book-implemented-in-python
    """
    tau = []
    lagvec = []
    #  Step through the different lags
    for lag in np.arange(2, 20):
        #  produce price difference with lag
        pp = np.subtract(p[lag:], p[:-lag])
        #  Write the different lags into a vector
        lagvec.append(lag)
        #  Calculate the variance of the difference vector
        tau.append(np.sqrt(np.std(pp)))
    #  linear fit to double-log graph (gives power)
    m = np.polyfit(np.log10(lagvec), np.log10(tau), 1)
    # calculate hurst
    h = m[0] * 2
    # plot lag vs variance
    return h
