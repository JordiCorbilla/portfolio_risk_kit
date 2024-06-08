"""
This module provides various financial functions and tools for analyzing and handling portfolio data learned from EDHEC Business School, 
computing statistical metrics, and optimizing portfolios based on different criteria. The main features include:
- Loading and formatting financial datasets (Fama-French, EDHEC Hedge Fund Index, etc.)
- Computing portfolio statistics (returns, volatility, Sharpe ratio, etc.)
- Running backtests on different portfolio strategies
- Efficient Frontier plotting
- Value at Risk (VaR) and Conditional Value at Risk (CVaR) computations
- Portfolio optimization based on different risk metrics

Dependencies: pandas, numpy, scipy, statsmodels
"""

import pandas as pd
import numpy as np
import scipy.stats
import statsmodels.api as sm
import math
from scipy.stats import norm
from scipy.optimize import minimize


def get_ffme_returns(file_path):
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    me_m = pd.read_csv(file_path, header=0, index_col=0, na_values=-99.99)
    returns = me_m[['Lo 10', 'Hi 10']]
    returns.columns = ['SmallCap', 'LargeCap']
    returns = returns / 100
    returns.index = pd.to_datetime(returns.index, format="%Y%m").to_period('M')
    return returns


def get_fff_returns(file_path):
    """
    Load the Fama-French Research Factor Monthly Dataset
    """
    returns = pd.read_csv(file_path, header=0, index_col=0, na_values=-99.99) / 100
    returns.index = pd.to_datetime(returns.index, format="%Y%m").to_period('M')
    return returns


def get_hfi_returns(file_path):
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period('M')
    return hfi


def get_ind_file(file_path, filetype, weighting="vw", n_inds=30):
    """
    Load and format the Ken French Industry Portfolios files
    Variant is a tuple of (weighting, size) where:
        weighting is one of "ew", "vw"
        number of inds is 30 or 49
    """    
    if filetype == "returns":
        name = f"{weighting}_rets"
        divisor = 100
    elif filetype == "nfirms":
        name = "nfirms"
        divisor = 1
    elif filetype == "size":
        name = "size"
        divisor = 1
    else:
        raise ValueError(f"filetype must be one of: returns, nfirms, size")

    ind = pd.read_csv(file_path, header=0, index_col=0, na_values=-99.99) / divisor
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def get_ind_returns(file_path, weighting="vw", n_inds=30):
    """
    Load and format the Ken French Industry Portfolios Monthly Returns
    """
    return get_ind_file(file_path, "returns", weighting=weighting, n_inds=n_inds)


def get_ind_nfirms(file_path, n_inds=30):
    """
    Load and format the Ken French 30 Industry Portfolios Average number of Firms
    """
    return get_ind_file(file_path, "nfirms", n_inds=n_inds)


def get_ind_size(file_path, n_inds=30):
    """
    Load and format the Ken French 30 Industry Portfolios Average size (market cap)
    """
    return get_ind_file(file_path, "size", n_inds=n_inds)


def get_ind_market_caps(nfirms_file_path, size_file_path, n_inds=30, weights=False):
    """
    Load the industry portfolio data and derive the market caps
    """
    ind_nfirms = get_ind_nfirms(nfirms_file_path, n_inds=n_inds)
    ind_size = get_ind_size(size_file_path, n_inds=n_inds)
    ind_mktcap = ind_nfirms * ind_size
    if weights:
        total_mktcap = ind_mktcap.sum(axis=1)
        ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
        return ind_capweight
    return ind_mktcap


def get_total_market_index_returns(nfirms_file_path, size_file_path, returns_file_path, n_inds=30):
    """
    Load the 30 industry portfolio data and derive the returns of a capweighted total market index
    """
    ind_capweight = get_ind_market_caps(nfirms_file_path, size_file_path, n_inds=n_inds)
    ind_return = get_ind_returns(returns_file_path, weighting="vw", n_inds=n_inds)
    total_market_return = (ind_capweight * ind_return).sum(axis="columns")
    return total_market_return


def skewness(returns):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_returns = returns - returns.mean()
    sigma_returns = returns.std(ddof=0)
    exp = (demeaned_returns**3).mean()
    return exp / sigma_returns**3


def kurtosis(returns):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_returns = returns - returns.mean()
    sigma_returns = returns.std(ddof=0)
    exp = (demeaned_returns**4).mean()
    return exp / sigma_returns**4


def compound(returns):
    """
    Returns the result of compounding the set of returns
    """
    return np.expm1(np.log1p(returns).sum())


def annualize_returns(returns, periods_per_year):
    """
    Annualizes a set of returns
    """
    compounded_growth = (1 + returns).prod()
    n_periods = returns.shape[0]
    return compounded_growth**(periods_per_year / n_periods) - 1


def annualize_volatility(returns, periods_per_year):
    """
    Annualizes the volatility of a set of returns
    """
    return returns.std() * (periods_per_year**0.5)


def sharpe_ratio(returns, riskfree_rate, periods_per_year):
    """
    Computes the annualized Sharpe ratio of a set of returns
    """
    rf_per_period = (1 + riskfree_rate)**(1 / periods_per_year) - 1
    excess_returns = returns - rf_per_period
    ann_excess_returns = annualize_returns(excess_returns, periods_per_year)
    ann_volatility = annualize_volatility(returns, periods_per_year)
    return ann_excess_returns / ann_volatility


def is_normal(returns, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(returns)
        return p_value > level


def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns.
    Returns a DataFrame with columns for
    the wealth index, 
    the previous peaks, and 
    the percentage drawdown
    """
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index, 
        "Previous Peak": previous_peaks, 
        "Drawdown": drawdowns
    })


def semideviation(returns):
    """
    Returns the semideviation aka negative semideviation of returns
    returns must be a Series or a DataFrame, else raises a TypeError
    """
    if isinstance(returns, pd.Series):
        is_negative = returns < 0
        return returns[is_negative].std(ddof=0)
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(semideviation)
    else:
        raise TypeError("Expected returns to be a Series or DataFrame")


def var_historic(returns, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(var_historic, level=level)
    elif isinstance(returns, pd.Series):
        return -np.percentile(returns, level)
    else:
        raise TypeError("Expected returns to be a Series or DataFrame")


def cvar_historic(returns, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(returns, pd.Series):
        is_beyond = returns <= -var_historic(returns, level=level)
        return -returns[is_beyond].mean()
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected returns to be a Series or DataFrame")


def var_gaussian(returns, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    z = norm.ppf(level / 100)
    if modified:
        s = skewness(returns)
        k = kurtosis(returns)
        z = (z +
             (z**2 - 1) * s / 6 +
             (z**3 - 3 * z) * (k - 3) / 24 -
             (2 * z**3 - 5 * z) * (s**2) / 36)
    return -(returns.mean() + z * returns.std(ddof=0))


def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns


def portfolio_volatility(weights, covmat):
    """
    Computes the volatility of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    volatility = (weights.T @ covmat @ weights)**0.5
    return volatility


def plot_ef2(n_points, expected_returns, cov, style):
    """
    Plots the 2-asset efficient frontier
    """
    if expected_returns.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, expected_returns) for w in weights]
    volatilities = [portfolio_volatility(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": volatilities
    })
    return ef.plot.line(x="Volatility", y="Returns", style=style)


def minimize_volatility(target_return, expected_returns, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = expected_returns.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    return_is_target = {'type': 'eq', 'args': (expected_returns,), 'fun': lambda weights, expected_returns: target_return - portfolio_return(weights, expected_returns)}
    weights = minimize(portfolio_volatility, init_guess, args=(cov,), method='SLSQP', options={'disp': False}, constraints=(weights_sum_to_1, return_is_target), bounds=bounds)
    return weights.x


def tracking_error(returns_a, returns_b):
    """
    Returns the Tracking Error between the two return series
    """
    return np.sqrt(((returns_a - returns_b)**2).sum())


def max_sharpe_ratio(riskfree_rate, expected_returns, cov):
    """
    Returns the weights of the portfolio that gives you the maximum Sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = expected_returns.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    def neg_sharpe(weights, riskfree_rate, expected_returns, cov):
        r = portfolio_return(weights, expected_returns)
        vol = portfolio_volatility(weights, cov)
        return -(r - riskfree_rate) / vol
    weights = minimize(neg_sharpe, init_guess, args=(riskfree_rate, expected_returns, cov), method='SLSQP', options={'disp': False}, constraints=(weights_sum_to_1,), bounds=bounds)
    return weights.x


def global_minimum_volatility(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return max_sharpe_ratio(0, np.repeat(1, n), cov)


def optimal_weights(n_points, expected_returns, cov):
    """
    Returns a list of weights that represent a grid of n_points on the efficient frontier
    """
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), n_points)
    weights = [minimize_volatility(target_return, expected_returns, cov) for target_return in target_returns]
    return weights


def plot_ef(n_points, expected_returns, cov, style='.-', legend=False, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    Plots the multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, expected_returns, cov)
    rets = [portfolio_return(w, expected_returns) for w in weights]
    volatilities = [portfolio_volatility(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": volatilities
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    if show_cml:
        ax.set_xlim(left=0)
        w_msr = max_sharpe_ratio(riskfree_rate, expected_returns, cov)
        r_msr = portfolio_return(w_msr, expected_returns)
        vol_msr = portfolio_volatility(w_msr, cov)
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
    if show_ew:
        n = expected_returns.shape[0]
        w_ew = np.repeat(1 / n, n)
        r_ew = portfolio_return(w_ew, expected_returns)
        vol_ew = portfolio_volatility(w_ew, cov)
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = global_minimum_volatility(cov)
        r_gmv = portfolio_return(w_gmv, expected_returns)
        vol_gmv = portfolio_volatility(w_gmv, cov)
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
    return ax


def run_cppi(risky_returns, safe_returns=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    dates = risky_returns.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = account_value
    if isinstance(risky_returns, pd.Series): 
        risky_returns = pd.DataFrame(risky_returns, columns=["R"])

    if safe_returns is None:
        safe_returns = pd.DataFrame().reindex_like(risky_returns)
        safe_returns.values[:] = riskfree_rate / 12
    account_history = pd.DataFrame().reindex_like(risky_returns)
    risky_w_history = pd.DataFrame().reindex_like(risky_returns)
    cushion_history = pd.DataFrame().reindex_like(risky_returns)
    floorval_history = pd.DataFrame().reindex_like(risky_returns)
    peak_history = pd.DataFrame().reindex_like(risky_returns)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)
        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1 - risky_w
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w
        account_value = risky_alloc * (1 + risky_returns.iloc[step]) + safe_alloc * (1 + safe_returns.iloc[step])
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    risky_wealth = start * (1 + risky_returns).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_returns": risky_returns,
        "safe_returns": safe_returns,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor": floorval_history
    }
    return backtest_result


def summary_stats(returns, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of returns
    """
    ann_returns = returns.aggregate(annualize_returns, periods_per_year=12)
    ann_volatility = returns.aggregate(annualize_volatility, periods_per_year=12)
    ann_sr = returns.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = returns.aggregate(lambda returns: drawdown(returns).Drawdown.min())
    skew = returns.aggregate(skewness)
    kurt = returns.aggregate(kurtosis)
    cf_var5 = returns.aggregate(var_gaussian, modified=True)
    hist_cvar5 = returns.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_returns,
        "Annualized Volatility": ann_volatility,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })


def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    dt = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year) + 1
    rets_plus_1 = np.random.normal(loc=(1 + mu)**dt, scale=(sigma * np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0 * pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1 - 1
    return ret_val


def regress(dependent_variable, explanatory_variables, alpha=True):
    """
    Runs a linear regression to decompose the dependent variable into the explanatory variables
    returns an object of type statsmodel's RegressionResults on which you can call
       .summary() to print a full summary
       .params for the coefficients
       .tvalues and .pvalues for the significance levels
       .rsquared_adj and .rsquared for quality of fit
    """
    if alpha:
        explanatory_variables = explanatory_variables.copy()
        explanatory_variables["Alpha"] = 1
    
    lm = sm.OLS(dependent_variable, explanatory_variables).fit()
    return lm


def portfolio_tracking_error(weights, ref_returns, bb_returns):
    """
    Returns the tracking error between the reference returns
    and a portfolio of building block returns held with given weights
    """
    return tracking_error(ref_returns, (weights * bb_returns).sum(axis=1))


def style_analysis(dependent_variable, explanatory_variables):
    """
    Returns the optimal weights that minimizes the tracking error between
    a portfolio of the explanatory variables and the dependent variable
    """
    n = explanatory_variables.shape[1]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    solution = minimize(portfolio_tracking_error, init_guess, args=(dependent_variable, explanatory_variables,), method='SLSQP', options={'disp': False}, constraints=(weights_sum_to_1,), bounds=bounds)
    weights = pd.Series(solution.x, index=explanatory_variables.columns)
    return weights


def ff_analysis(returns, factors):
    """
    Returns the loadings of returns on the Fama French Factors
    which can be read in using get_fff_returns()
    the index of returns must be a (not necessarily proper) subset of the index of factors
    returns is either a Series or a DataFrame
    """
    if isinstance(returns, pd.Series):
        dependent_variable = returns
        explanatory_variables = factors.loc[returns.index]
        tilts = regress(dependent_variable, explanatory_variables).params
    elif isinstance(returns, pd.DataFrame):
        tilts = pd.DataFrame({col: ff_analysis(returns[col], factors) for col in returns.columns})
    else:
        raise TypeError("returns must be a Series or a DataFrame")
    return tilts


def weight_ew(returns, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
    """
    Returns the weights of the EW portfolio based on the asset returns "returns" as a DataFrame
    If supplied a set of capweights and a capweight tether, it is applied and reweighted 
    """
    n = len(returns.columns)
    ew = pd.Series(1 / n, index=returns.columns)
    if cap_weights is not None:
        cw = cap_weights.loc[returns.index[0]]
        if microcap_threshold is not None and microcap_threshold > 0:
            microcap = cw < microcap_threshold
            ew[microcap] = 0
            ew = ew / ew.sum()
        if max_cw_mult is not None and max_cw_mult > 0:
            ew = np.minimum(ew, cw * max_cw_mult)
            ew = ew / ew.sum()
    return ew


def weight_cw(returns, cap_weights, **kwargs):
    """
    Returns the weights of the CW portfolio based on the time series of capweights
    """
    w = cap_weights.loc[returns.index[0]]
    return w / w.sum()


def backtest_ws(returns, estimation_window=60, weighting=weight_ew, verbose=False, **kwargs):
    """
    Backtests a given weighting scheme, given some parameters:
    returns : asset returns to use to build the portfolio
    estimation_window: the window to use to estimate parameters
    weighting: the weighting scheme to use, must be a function that takes "returns", and a variable number of keyword-value arguments
    """
    n_periods = returns.shape[0]
    windows = [(start, start + estimation_window) for start in range(n_periods - estimation_window)]
    weights = [weighting(returns.iloc[win[0]:win[1]], **kwargs) for win in windows]
    weights = pd.DataFrame(weights, index=returns.iloc[estimation_window:].index, columns=returns.columns)
    portfolio_returns = (weights * returns).sum(axis="columns", min_count=1)
    return portfolio_returns


def sample_covariance(returns, **kwargs):
    """
    Returns the sample covariance of the supplied returns
    """
    return returns.cov()


def weight_gmv(returns, cov_estimator=sample_covariance, **kwargs):
    """
    Produces the weights of the GMV portfolio given a covariance matrix of the returns 
    """
    est_cov = cov_estimator(returns, **kwargs)
    return global_minimum_volatility(est_cov)


def cc_covariance(returns, **kwargs):
    """
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
    """
    rhos = returns.corr()
    n = rhos.shape[0]
    rho_bar = (rhos.values.sum() - n) / (n * (n - 1))
    ccor = np.full_like(rhos, rho_bar)
    np.fill_diagonal(ccor, 1.)
    sd = returns.std()
    return pd.DataFrame(ccor * np.outer(sd, sd), index=returns.columns, columns=returns.columns)


def shrinkage_covariance(returns, delta=0.5, **kwargs):
    """
    Covariance estimator that shrinks between the Sample Covariance and the Constant Correlation Estimators
    """
    prior = cc_covariance(returns, **kwargs)
    sample = sample_covariance(returns, **kwargs)
    return delta * prior + (1 - delta) * sample


def risk_contribution(weights, cov):
    """
    Compute the contributions to risk of the constituents of a portfolio, given a set of portfolio weights and a covariance matrix
    """
    total_portfolio_var = portfolio_volatility(weights, cov)**2
    marginal_contrib = cov @ weights
    risk_contrib = np.multiply(marginal_contrib, weights.T) / total_portfolio_var
    return risk_contrib


def target_risk_contributions(target_risk, cov):
    """
    Returns the weights of the portfolio that gives you the weights such
    that the contributions to portfolio risk are as close as possible to
    the target_risk, given the covariance matrix
    """
    n = cov.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    def msd_risk(weights, target_risk, cov):
        w_contribs = risk_contribution(weights, cov)
        return ((w_contribs - target_risk)**2).sum()
    weights = minimize(msd_risk, init_guess, args=(target_risk, cov), method='SLSQP', options={'disp': False}, constraints=(weights_sum_to_1,), bounds=bounds)
    return weights.x


def equal_risk_contributions(cov):
    """
    Returns the weights of the portfolio that equalizes the contributions
    of the constituents based on the given covariance matrix
    """
    n = cov.shape[0]
    return target_risk_contributions(target_risk=np.repeat(1 / n, n), cov=cov)


def weight_erc(returns, cov_estimator=sample_covariance, **kwargs):
    """
    Produces the weights of the ERC portfolio given a covariance matrix of the returns 
    """
    est_cov = cov_estimator(returns, **kwargs)
    return equal_risk_contributions(est_cov)

def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time period t
    and r is the per-period interest rate
    returns a |t| x |r| Series or DataFrame
    r can be a float, Series or DataFrame
    returns a DataFrame indexed by t
    """
    discounts = pd.DataFrame([(r+1)**-i for i in t])
    discounts.index = t
    return discounts

def pv(flows, r):
    """
    Compute the present value of a sequence of cash flows given by the time (as an index) and amounts
    r can be a scalar, or a Series or DataFrame with the number of rows matching the num of rows in flows
    """
    dates = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis='rows').sum()

def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of a series of liabilities, based on an interest rate and current value of assets
    """
    return pv(assets, r)/pv(liabilities, r)

def inst_to_ann(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.log1p(r)

def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices

def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns the series of cash flows generated by a bond,
    indexed by the payment/coupon number
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupons = np.repeat(coupon_amt, n_coupons)
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows
    
def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year,
                                      discount_rate.loc[t])
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)

def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows, given a per-period discount rate
    """
    discounted_flows = discount(flows.index, discount_rate)*flows
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights=weights)

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns the weight W in cf_s that, along with (1-W) in cf_l will have an effective
    duration that matches cf_t
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s)
