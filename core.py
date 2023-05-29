import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime 
from typing import List, Dict, Tuple
import seaborn as sns
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from dateutil.relativedelta import relativedelta


#figsize 
FS=(6,4)

def get_stock_info(stock_tickers: List[str]) -> List[dict]:
    """Returns the stock's country, industry and sector as a dict
    Parameters:
        stock_tickers: list of strings of desired stock tickers
    Output:
        list of dicts with the required fields as keys
    """
    li = []
    tickers = yf.Tickers(stock_tickers)
    # to avoid issues with keys due to lower case
    for t in [el.upper() for el in stock_tickers]:
        temp_dict = {key: value for key, value in tickers.tickers[t].info.items() if key in ['country', 'industry', 'sector', 'beta']}
        temp_dict["ticker"] = t
        li += [temp_dict]
    return li


def get_stock_returns(stock_tickers: List[str], start: str=None, end: str=None, repair=True, prepost=False) -> pd.DataFrame:
    """Gathers price data for stocks using yf.download() and transforms closing prices to daily (annualized) returns
    Parameters:
        stock_tickers: list of strings describing stocks with their tickers
        start: string describing the start date (inclusive) in format YYYY-MM-DD (1900-01-01 by default)
        end: string describing the end date (exclusive) in format YYYY-MM-DD (today by default)
        repair: boolean describing whether to repair probably wrong price data (True by default)
        prepost: boolean describing whether to include pre/post market data (False by default)
    Output:
        pd.DataFrame with the stock data for available dates
    """
    df = yf.download(stock_tickers, start=start, end=end, prepost=prepost, repair=repair)["Close"]
    df_shifted = df.shift(periods=1)
    df_returns = ((df - df_shifted) / df_shifted).shift(periods=-1)
    if not (isinstance(df_returns, pd.DataFrame)):
        df_returns = pd.DataFrame(df_returns)
        df_returns.columns = stock_tickers
    return df_returns


def updater(stock_tickers: List[str], start: str=None, end: str=None, ret_file_name: str="stock_returns.csv", info_file_name: str="stock_info.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetches return data as well as extra stock information from Yahoo Finance and exports it in two .csv files
    
    Parameters:
        stock_tickers: list of strings describing stocks with their tickers
        start: string describing the start date (inclusive) in format YYYY-MM-DD (1900-01-01 by default)
        end: string describing the end date (exclusive) in format YYYY-MM-DD (today by default)
        ret_file_name: string describing the file name where the returns are saved (.csv format)
    
    Output:
        pd.DataFrame tuple, first element containing all the returns, second element with the extra information, both exported to a csv with df.to_csv()
    """
    # creation of return df
    df = get_stock_returns(stock_tickers, start=start, end=end)
    df.to_csv(ret_file_name)
    
    # creation of info df
    list_dicts = get_stock_info(stock_tickers)
    df_info = pd.DataFrame.from_dict(list_dicts).T
    df_info = df_info.rename(columns = df_info.loc["ticker"])
    df_info = df_info.drop("ticker")
    df_info.to_csv(info_file_name)
    
    return df, df_info


def get_rf_rate(ticker: str='^IRX', date: datetime.datetime=None) -> float:
    """Returns the annualized risk-free rate of the day from Yahoo Finance.
    
    Parameters:
        ticker: str with the rate ticker in YF (by default, the 3 month T-bill rate)
        date: datetime.date representing the required date (current day be default)

    Output:
        float representing the risk-free rate of the desired day
    """
    if date is None:
        today = datetime.datetime.today()
        # to make sure that even in the week-end we get the latest available date
        start_today = today - datetime.timedelta(days=7)
        end_today = today
    else:
        start_today = date - datetime.timedelta(days=1)
        end_today = date

    ret = yf.download(ticker, start=start_today, end=end_today)['Close'] / 100.0
    return ret.iloc[len(ret)-1]


def updater_rf_rate(ticker: str='^IRX', start: str=None, end: str=None, rf_file_name: str="rf_rate.csv"):
    """Fetches return data as well as extra stock information from Yahoo Finance and exports it in two .csv files
    
    Parameters:
        ticker: string describing the rf rate with its ticker
        start: string describing the start date (inclusive) in format YYYY-MM-DD (1900-01-01 by default)
        end: string describing the end date (exclusive) in format YYYY-MM-DD (today by default)
        rf_file_name: string describing the file name where the rate is saved (.csv format)
    
    Output:
        pd.DataFrame tuple, first element containing all the returns, second element with the extra information, both exported to a csv with df.to_csv()
    """

    df_rf = pd.DataFrame(yf.download(ticker, start=start, end=end)['Close'] / 100.0) # de-annualize
    df_rf.to_csv(rf_file_name)
    df_rf.columns = [ticker]
    return df_rf


def ext_and_merger_rf(rf_csv: str, start: str=None, end: str=None) -> pd.DataFrame:
    """Takes a csv file with time series of risk-free rate and extracts it
    
    Parameters:
        ret_csv: string describing the path to the csv file with the returns
        start: str (optional) describing the starting date from where we want the rate, if applicable
        end: str (optional) describing the end date from where we want the rate (included)
    
    Output:
        pd.DataFrame with the risk-free rate, for dates for which it exists
    """
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("Starting date can't be larger than end date.")
    
    loc_df = pd.read_csv(rf_csv, index_col=0)
    loc_df.index = pd.to_datetime(loc_df.index)

    ext_rf = loc_df.dropna()
    
    if start_date is not None:
        if end_date is not None:
            ext_rf = ext_rf.loc[start_date: end_date]
        else:
            ext_rf = ext_rf.loc[start_date:]
    elif end is not None:
        ext_rf = ext_rf.loc[:end_date]
    return ext_rf
    

def get_market_returns(ticker: str='SPY', start: str=None, end: str=None) -> pd.DataFrame:
    """Fetches return data for the market (SPY by default) from Yahoo Finance and exports it in a .csv file
    
    Parameters:
        ticker: string describing the market index with its ticker
        start: string describing the start date (inclusive) in format YYYY-MM-DD (1900-01-01 by default)
        end: string describing the end date (exclusive) in format YYYY-MM-DD (today by default)
    
    Output:
        pd.DataFrame, containing the daily returns, exported to a csv with df.to_csv()
    """
    df = get_stock_returns([ticker], start=start, end=end)
    df.to_csv("market_returns.csv")
    return df.dropna()


def ext_and_merger(ret_csv: str, tickers: List[str], start: str=None, end: str=None) -> pd.DataFrame:

    """Takes a csv file with returns and extracts the returns for the given tickers
    
    Parameters:
        ret_csv: string describing the path to the csv file with the returns
        tickers: list of strings describing stocks with their tickers
        start: str (optional) describing the starting date from where we want the stock information, if applicable
        end: str (optional) describing the end date from where we want the stock information (included)
    
    Output:
        pd.DataFrame with the returns for the given tickers, for dates for which all tickers have returns
    """
    
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("Starting date can't be larger than end date.")
    
    loc_df = pd.read_csv(ret_csv, index_col=0)
    loc_df.index = pd.to_datetime(loc_df.index)

    ext_stock_ret = loc_df[tickers].dropna()
    
    if start_date is not None:
        if end_date is not None:
            ext_stock_ret = ext_stock_ret.loc[start_date: end_date]
        else:
            ext_stock_ret = ext_stock_ret.loc[start_date:]
    elif end is not None:
        ext_stock_ret = ext_stock_ret.loc[:end_date]
    return ext_stock_ret


def ext_and_merger_info(info_csv: str, tickers: List[str]) -> pd.DataFrame:

    """Takes a csv file with qualitative information and extracts it for the provided tickers

    Parameters:
        info_csv: string describing the path to the csv file with the information
        tickers: list of strings describing stocks with their tickers

    Output:
        pd.DataFrame with the information for the given tickers
    """
    
    loc_df = pd.read_csv(info_csv, index_col=0)
    
    ext_stock_info = loc_df[tickers]

    return ext_stock_info



class MVE_Portfolio:
    def __init__(self, df_stock_ret: pd.DataFrame, df_stock_ret_total: pd.DataFrame, df_stock_ret_future: pd.DataFrame, df_stock_info: pd.DataFrame, Rf: float, Rf_returns: pd.DataFrame, curr_date: datetime.datetime, market_ret: pd.DataFrame, market_ret_future: pd.DataFrame, market_ticker: str='SPY', mu_targ: float=None, std_targ: float=None):
        
        self.df_stock_ret = df_stock_ret
        self.df_stock_ret_total = df_stock_ret_total.merge(Rf_returns, on="Date")
        self.df_stock_ret_future = df_stock_ret_future.merge(Rf_returns, on="Date")
        self.stock_info = df_stock_info
        self.Rf = Rf
        self.Rf_returns = Rf_returns
        self.market_ret = market_ret
        self.market_ret_future = market_ret_future
        self.market_ticker = market_ticker
        self.curr_date = curr_date
        
        # annualized mean return
        
        self.mu_vec = df_stock_ret.mean() * 252

        # covariance matrix from stock returns (annualized)

        self.sigma_mat = df_stock_ret.cov() * 252
        
        self.inv_sigma_mat = np.linalg.inv(self.sigma_mat)

        # vector of ones

        vec_ones = np.ones(df_stock_ret.shape[1])

        # A scalar

        self.A = vec_ones.T @ self.inv_sigma_mat @ vec_ones

        # B scalar

        self.B = vec_ones.T @ self.inv_sigma_mat @ self.mu_vec

        # C scalar

        self.C = self.mu_vec.T @ self.inv_sigma_mat @ self.mu_vec

        # D scalar

        self.D = self.A * self.C - self.B ** 2
        
        if mu_targ is not None:
            self.std_targ = target(self.A, self.B, self.C, self.Rf, mu_targ=mu_targ, std_targ=std_targ)
            self.mu_targ = mu_targ

        else:
            self.mu_targ = target(self.A, self.B, self.C, self.Rf, mu_targ=mu_targ, std_targ=std_targ)
            self.std_targ = std_targ

        ###### Optimization WITHOUT risk free asset

        # lagrangian multipliers

        self.lambda_vec = (self.C - self.mu_targ * self.B)/ self.D

        self.gamma_vec = (self.mu_targ * self.A - self.B) / self.D

        # GLOBAL MINIMUM VARIANCE PORTFOLIO

        self.w_glob_min = (self.inv_sigma_mat @ vec_ones) / self.A

        # GMVP RETURN

        self.mu_glob_min = self.B / self.A

        # GMVP VARIANCE

        self.var_glob_min = 1 / self.A

        # SLOPE PORTFOLIO

        self.w_slope = (self.inv_sigma_mat @ self.mu_vec) / self.B

        # SLOPE RETURN

        self.mu_slope = self.mu_vec

        # SLOPE VARIANCE

        self.var_slope = self.C / self.B**2

        ###### Optimization WITH risk free asset

        # tangency portfolio

        self.w_tang = (self.inv_sigma_mat @ (self.mu_vec - self.Rf * vec_ones)) / (self.B - self.Rf * self.A)

        # tangency portfolio return

        self.mu_tang = (self.C - self.B * self.Rf) / (self.B - self.A * self.Rf)

        # tangency portfolio variance

        self.var_tang = (self.C - 2 * self.B * self.Rf + self.A * self.Rf**2) / (self.B - self.A * self.Rf)**2

        # tangency portfolio sharpe ratio

        self.sharpe_tang = (self.mu_tang - self.Rf) / np.sqrt(self.var_tang)

        ##### TARGET PORTFOLIO
        
        # target portfolio weights
        self.w_targ = (self.B - self.A * self.Rf)*self.w_tang
        
        # risk-free weight
        self.w_rf = 1 - sum(self.w_targ)
        
        # actual PF returns time series (until the last possible date, i.e. today)
        self.returns_targ = self.df_stock_ret_total @ (np.append(self.w_targ, self.w_rf))
        self.returns_targ_future = self.df_stock_ret_future @ (np.append(self.w_targ, self.w_rf))
        
        self.sharpe_targ = (self.mu_targ - self.Rf) / self.std_targ
        
        # beta
        self.df_mkt_pf = pd.DataFrame(self.returns_targ).merge(self.market_ret, on='Date')
        self.mu_mkt = self.df_mkt_pf[market_ticker].mean() * 252
        self.beta_targ = self.stock_info.loc['beta'].apply(lambda x: float(x)).T @ self.w_targ
        
        # cumulative returns
        self.cumret_targ_total = (1 + self.returns_targ).cumprod()
        self.cumret_mkt_total = (1 + self.df_mkt_pf[market_ticker]).cumprod()
        
        self.cumret_targ_future = (1 + self.returns_targ_future).cumprod()
        self.cumret_mkt_future = (1 + self.market_ret_future).cumprod()
        
        
    def eff_frontier_plot(self, do_plot=True):
        """Plots the tangent portfolio, the CML, the target portfolio, the global MV portfolio as well as the efficient frontier for the current instance of the MVE Portfolio"""
        x = np.linspace(0, 0.5, 1500)
        if do_plot:
            x = np.linspace(0, 0.5, 1500)
            plt.figure(figsize=FS)
            plt.plot(np.sqrt((self.A*(x)**2 - 2*self.B*x + self.C)/self.D), x, label="Efficient Frontier") # Efficient Frontier no risk free asset
            plt.plot(x, self.sharpe_tang * x + self.Rf, label="CML") # Capital Market Line
            plt.plot(np.sqrt(self.var_tang), self.mu_tang, 'ro', label="tangent PF")
            plt.plot(self.std_targ, self.mu_targ, 'bo', label="target PF")
            plt.plot(np.sqrt(self.var_glob_min), self.mu_glob_min, 'go', label="global MV PF")
            plt.xlabel('Standard Deviation')
            plt.ylabel('Expected Return')
            plt.title('Efficient Frontier and Capital Market Line')
            plt.legend(loc='best')
            plt.show()

        fig = plt.Figure(figsize=FS)
        ax = fig.add_subplot(111)
        ax.plot(np.sqrt((self.A*(x)**2 - 2*self.B*x + self.C)/self.D), x, label="Efficient Frontier")
        ax.plot(x, self.sharpe_tang * x + self.Rf, label="CML")
        ax.plot(np.sqrt(self.var_tang), self.mu_tang, 'ro', label="tangent PF")
        ax.plot(self.std_targ, self.mu_targ, 'bo', label="target PF")
        ax.plot(np.sqrt(self.var_glob_min), self.mu_glob_min, 'go', label="global MV PF")
        ax.set_xlabel('Standard Deviation')
        ax.set_ylabel('Expected Return')
        ax.set_title('Efficient Frontier and Capital Market Line')
        ax.legend(loc='best')

        return fig

    
    def sml_plot(self, do_plot=True):
        """Plots the target portfolio and the SML for the current instance of the MVE Portfolio"""
        x = np.linspace(0, 5, 1500)
        if do_plot:
            x = np.linspace(0, 5, 1500)
            plt.figure(figsize=FS)
            plt.plot(x, (self.mu_mkt - self.Rf) * x + self.Rf, label='SML') # Security Market Line
            plt.plot(self.beta_targ, self.mu_targ, 'go', label="target PF")
            plt.xlabel('Beta')
            plt.ylabel('Expected Return')
            plt.title('Security Market Line')
            plt.legend()
            plt.show()

        fig = plt.Figure(figsize=FS)
        ax = fig.add_subplot(111)
        ax.plot(x, (self.mu_mkt - self.Rf) * x + self.Rf, label='SML')
        ax.plot(self.beta_targ, self.mu_targ, 'go', label="target PF")
        ax.set_xlabel('Beta')
        ax.set_ylabel('Expected Return')
        ax.set_title('Security Market Line')
        ax.legend()

        return fig

    
    def returns_plot(self, do_plot=True):
        """Plots the returns over the entire time period for the MVE Portfolio and the market"""
        if do_plot: 
            plt.figure(figsize=FS)
            plt.plot(self.df_mkt_pf[0], label='Target PF')
            plt.plot(self.df_mkt_pf[self.market_ticker], label='Market PF')
            plt.vlines(self.curr_date, min(min(self.df_mkt_pf[0]), min(self.df_mkt_pf[self.market_ticker])), max(max(self.df_mkt_pf[0]), max(self.df_mkt_pf[self.market_ticker])), color='black', label='Portfolio creation date')
            plt.xlabel('Time')
            plt.ylabel('Return')
            plt.title('Returns for the period')
            plt.legend()
            plt.show()

        fig = plt.Figure(figsize=FS)
        ax = fig.add_subplot(111)
        ax.plot(self.df_mkt_pf[0], label='Target PF')
        ax.plot(self.df_mkt_pf[self.market_ticker], label='Market PF')
        ax.vlines(self.curr_date, min(min(self.df_mkt_pf[0]), min(self.df_mkt_pf[self.market_ticker])), max(max(self.df_mkt_pf[0]), max(self.df_mkt_pf[self.market_ticker])), color='black', label='Portfolio creation date')
        ax.set_xlabel('Time')
        ax.set_ylabel('Return')
        ax.set_title('Returns for the period')
        ax.legend()

        return fig
    
    
    def cumret_plot_from_start(self, do_plot=True):
        """Plots the cumulative returns over the entire time period for the MVE Portfolio and the market"""
        if do_plot:
            plt.figure(figsize=FS)
            plt.plot(self.cumret_targ_total, label='Target PF')
            plt.plot(self.cumret_mkt_total, label='Market PF')
            plt.vlines(self.curr_date, 0, max(max(self.cumret_targ_total), max(self.cumret_mkt_total)), color='black', label='Portfolio creation date')
            plt.xlabel('Time')
            plt.ylabel('Return')
            plt.title('Cumulative Returns for the period')
            plt.legend()
            plt.show()

        fig = plt.Figure(figsize=FS)
        ax = fig.add_subplot(111)
        ax.plot(self.cumret_targ_total, label='Target PF')
        ax.plot(self.cumret_mkt_total, label='Market PF')
        ax.vlines(self.curr_date, 0, max(max(self.cumret_targ_total), max(self.cumret_mkt_total)), color='black', label='Portfolio creation date')
        ax.set_xlabel('Time')  
        ax.set_ylabel('Return')  
        ax.set_title('Cumulative Returns for the period')
        ax.legend()

        return fig
    
    
    def cumret_plot_from_today(self, do_plot=True):
        """Plots the cumulative returns after today for the MVE Portfolio and the market"""
        if do_plot:
            plt.figure(figsize=FS)
            plt.plot(self.cumret_targ_future, label='Target PF')
            plt.plot(self.cumret_mkt_future, label='Market PF')
            plt.xlabel('Time')
            plt.ylabel('Return')
            plt.title('Cumulative Returns for the future')
            plt.legend()
            plt.show()

        fig = plt.Figure(figsize=FS)
        ax = fig.add_subplot(111)
        ax.plot(self.cumret_targ_future, label='Target PF')
        ax.plot(self.cumret_mkt_future, label='Market PF')
        ax.set_xlabel('Time')  
        ax.set_ylabel('Return') 
        ax.set_title('Cumulative Returns for the future')
        ax.legend()

        return fig
    
    
    def portfolio_specs(self):
        """Provides the expected return, the standard deviation and the Sharpe Ratio of the portfolio in a pd.DataFrame"""

        portfolio_specs_df = pd.DataFrame({'Expected Return': float(self.mu_targ*100), 'Standard Deviation': float(self.std_targ*100), 'Sharpe Ratio': float(self.sharpe_targ*100)}, index= ['values (in %)'])

        return portfolio_specs_df
    
    
    def get_weights(self):
        """Provides the weight allocation for each stock in a pd.DataFrame"""
        
        df_weights = pd.DataFrame(self.w_targ, index=self.stock_info.columns)
        df_weights.columns = ['weight']
        
        return df_weights


    
def target(A, B, C, Rf, mu_targ: float=None, std_targ: float=None) -> float:
        """Computes a missing target volatility or return from the other (given) value

        Parameters:
            A, B, C: floats (represent A, B, C from the MVE Portfolio theory)
            Rf: float representing risk-free rate in the market
            mu_targ: float (default None) from which we get the matching level of volatility
            std_targ: float (default None) from which we get the matching level of return

        Output:
            float representing the target volatility or return"""

        if mu_targ is None and std_targ is not None:

            mu_targ = np.sqrt(C - 2*B * Rf + Rf**2*A)*std_targ + Rf

            return mu_targ

        elif std_targ is None and mu_targ is not None:

            std_targ = (mu_targ - Rf) / np.sqrt(C - 2*B * Rf + Rf**2*A)

            return std_targ

        else:
            raise ValueError("Both target return and volatility can't be None.")



class Momentum_Portfolio:
    def __init__(self, df_stock_ret: pd.DataFrame, df_stock_ret_total: pd.DataFrame, df_stock_ret_future: pd.DataFrame, df_stock_info: pd.DataFrame, Rf: float, Rf_returns: pd.DataFrame, curr_date: datetime.datetime, market_ret: pd.DataFrame, market_ret_future: pd.DataFrame, market_ticker: str='SPY', amount_winners: int=1, amount_losers: int=1, percentage_winners: float=None, percentage_losers: float=None, skip_week: bool=True, months: int=6):
        amount_stocks = len(df_stock_ret.columns) 
            
        if percentage_winners is not None and percentage_losers is not None:
            if percentage_winners > 0.5 or percentage_losers > 0.5:
                raise ValueError("Can't select over half of the stocks as winners or losers!")
                
            elif percentage_winners < 0.0 or percentage_losers < 0.0:
                raise ValueError("Can't have negative percentages for winners or losers!")

            self.amount_winners = int(max(1, percentage_winners * amount_stocks))
            self.amount_losers = int(max(1, percentage_losers * amount_stocks))
        elif amount_winners is not None and amount_losers is not None:
            if amount_winners > amount_stocks / 2 or amount_losers > amount_stocks / 2:
                raise ValueError("Can't select over half of the stocks as winners or losers!") 
            self.amount_winners = amount_winners
            self.amount_losers = amount_losers
        else:
            raise ValueError("Issue with inputs for winners or losers!")
        
        if months <= 0 or months is None:
            raise ValueError("Can't have negative amount of months (or no months)!")
        
        self.df_stock_ret_total = df_stock_ret_total
        self.df_stock_ret_future = df_stock_ret_future
        self.stock_info = df_stock_info
        self.Rf = Rf
        self.market_ret_future = market_ret_future
        self.market_ticker = market_ticker
        self.curr_date = curr_date
        
        # allows user to either skip last week or not
        if skip_week:
            end_date = curr_date - datetime.timedelta(days=7)
            start_date = end_date - relativedelta(months=months)
            self.df_stock_ret = df_stock_ret.loc[start_date: end_date]
            self.Rf_returns = Rf_returns.loc[start_date: end_date]
            self.market_ret = market_ret.loc[start_date: end_date]
        else:
            end_date = curr_date
            start_date = end_date - relativedelta(months=months)
            self.df_stock_ret = df_stock_ret.loc[start_date: end_date]
            self.Rf_returns = Rf_returns.loc[start_date: end_date]
            self.market_ret = market_ret.loc[start_date: end_date]
        
        self.period_return = (1 + self.df_stock_ret).cumprod().iloc[len(self.df_stock_ret) - 1]
        self.losers = self.period_return.sort_values()[0:self.amount_losers]
        self.winners = self.period_return.sort_values()[len(self.period_return) - self.amount_winners:len(self.period_return)]
        
        self.mu_targ = self.winners.mean() - self.losers.mean()
        # we pass lists as values to avoid issue with from_dict() below
        dict_momentum = {key: [1 / self.amount_winners] if key in self.winners.index else [-1 / self.amount_losers] if key in self.losers.index else [0] for key in df_stock_ret.columns}
        self.w_targ = pd.DataFrame.from_dict(dict_momentum)
        
        self.returns_targ = self.df_stock_ret_total @ self.w_targ.T
        self.returns_targ_future = self.df_stock_ret_future @ self.w_targ.T
        
        self.std_targ = (self.df_stock_ret @ self.w_targ.T).std()
        
        self.sharpe_targ = (self.mu_targ - self.Rf) / self.std_targ
        
        self.df_mkt_pf = pd.DataFrame(self.returns_targ).merge(self.market_ret, on='Date')
        self.mu_mkt = self.df_mkt_pf[market_ticker].mean() * 252
        self.beta_targ = self.stock_info.loc['beta'].apply(lambda x: float(x)) @ self.w_targ.T
        
        # cumulative returns
        self.cumret_targ_total = (1 + self.returns_targ).cumprod()
        self.cumret_mkt_total = (1 + self.df_mkt_pf[market_ticker]).cumprod()
        
        self.cumret_targ_future = (1 + self.returns_targ_future).cumprod()
        self.cumret_mkt_future = (1 + self.market_ret_future).cumprod()
        
        
    def returns_plot(self, do_plot=True):
        """Plots the returns over the entire time period for the Momentum Portfolio and the market"""
        if do_plot:
            plt.figure(figsize=FS)
            plt.plot(self.df_mkt_pf[0], label='Target PF')
            plt.plot(self.df_mkt_pf[self.market_ticker], label='Market PF')
            plt.vlines(self.curr_date, min(min(self.df_mkt_pf[0]), min(self.df_mkt_pf[self.market_ticker])), max(max(self.df_mkt_pf[0]), max(self.df_mkt_pf[self.market_ticker])), color='black', label='Portfolio creation date')
            plt.xlabel('Time')
            plt.ylabel('Return')
            plt.title('Returns for the period')
            plt.legend()
            plt.show()

        fig = plt.Figure(figsize=FS)
        ax = fig.add_subplot(111)
        ax.plot(self.df_mkt_pf[0], label='Target PF')
        ax.plot(self.df_mkt_pf[self.market_ticker], label='Market PF')
        ax.vlines(self.curr_date, min(min(self.df_mkt_pf[0]), min(self.df_mkt_pf[self.market_ticker])), max(max(self.df_mkt_pf[0]), max(self.df_mkt_pf[self.market_ticker])), color='black', label='Portfolio creation date')
        ax.set_xlabel('Time')
        ax.set_ylabel('Return') 
        ax.set_title('Returns for the period')
        ax.legend()

        return fig


    def cumret_plot_from_start(self, do_plot=True):
        """Plots the cumulative returns over the entire time period for the Momentum Portfolio and the market"""
        if do_plot:
            plt.figure(figsize=FS)
            plt.plot(self.cumret_targ_total, label='Target PF')
            plt.plot(self.cumret_mkt_total, label='Market PF')
            plt.vlines(self.curr_date, 0, max(max(self.cumret_targ_total), max(self.cumret_mkt_total)), color='black', label='Portfolio creation date')
            plt.xlabel('Time')
            plt.ylabel('Return')
            plt.title('Cumulative Returns for the period')
            plt.legend()
            plt.show()

        fig = plt.Figure(figsize=FS)
        ax = fig.add_subplot(111)
        ax.plot(self.cumret_targ_total, label='Target PF')
        ax.plot(self.cumret_mkt_total, label='Market PF')
        ax.vlines(self.curr_date, 0, max(max(self.cumret_targ_total), max(self.cumret_mkt_total)), color='black', label='Portfolio creation date')
        ax.set_xlabel('Time')  
        ax.set_ylabel('Return')  
        ax.set_title('Cumulative Returns for the period')
        ax.legend()

        return fig

    
    def cumret_plot_from_today(self, do_plot=True):
        """Plots the cumulative returns after today for the Momentum Portfolio and the market"""
        if do_plot:
            plt.figure(figsize=FS)
            plt.plot(self.cumret_targ_future, label='Target PF')
            plt.plot(self.cumret_mkt_future, label='Market PF')
            plt.xlabel('Time')
            plt.ylabel('Return')
            plt.title('Cumulative Returns for the future')
            plt.legend()
            plt.show()

        fig = plt.Figure(figsize=FS)
        ax = fig.add_subplot(111)
        ax.plot(self.cumret_targ_future, label='Target PF')
        ax.plot(self.cumret_mkt_future, label='Market PF')
        ax.set_xlabel('Time')  
        ax.set_ylabel('Return') 
        ax.set_title('Cumulative Returns for the future')
        ax.legend()

        return fig
    
    
    def portfolio_specs(self):
        """Provides the expected return, the standard deviation and the Sharpe Ratio of the portfolio in a pd.DataFrame"""
        
        portfolio_specs_df = pd.DataFrame({'Expected Return': float(self.mu_targ*100), 'Standard Deviation': float(self.std_targ*100), 'Sharpe Ratio': float(self.sharpe_targ*100)}, index= ['values (in %)'])
        
        return portfolio_specs_df
    
    
    def get_weights(self):
        """Provides the weight allocation for each stock in a pd.DataFrame"""
        
        df_weights = pd.DataFrame(self.w_targ).T
        df_weights.columns = ['weight']
        return df_weights
        


class BAB_Portfolio:
    def __init__(self, df_stock_ret_total: pd.DataFrame, df_stock_info: pd.DataFrame, Rf: float, Rf_returns: pd.DataFrame, market_ret: pd.DataFrame, market_ticker: str='SPY', amount_high: int=1, amount_low: int=1, percentage_high: float=None, percentage_low: float=None):
        amount_stocks = len(df_stock_ret_total.columns) 
            
        if percentage_high is not None and percentage_low is not None:
            if percentage_high > 0.5 or percentage_low > 0.5:
                raise ValueError("Can't select over half of the stocks as high or low beta stocks!")
                
            elif percentage_high < 0.0 or percentage_low < 0.0:
                raise ValueError("Can't have negative percentages for winners or losers!")

            self.amount_high = int(max(1, percentage_high * amount_stocks))
            self.amount_low = int(max(1, percentage_low * amount_stocks))

        elif amount_high is not None and amount_low is not None:
            if amount_high > amount_stocks / 2 or amount_low > amount_stocks / 2:
                raise ValueError("Can't select over half of the stocks as high or low beta stocks!")

            self.amount_high = amount_high
            self.amount_low = amount_low
        else:
            raise ValueError("Issue with inputs for high or low beta stocks!")
        
        self.df_stock_ret_total = df_stock_ret_total
        self.stock_info = df_stock_info
        self.Rf = Rf
        self.market_ret = market_ret
        self.market_ticker = market_ticker
        
        betas = self.stock_info.loc['beta']
        self.low = betas.sort_values()[0:self.amount_low]
        self.high = betas.sort_values()[len(betas) - self.amount_high:len(betas)]
        
        # we pass lists as values to avoid issue with from_dict() below
        dict_momentum = {key: [1 / self.amount_low] if key in self.low.index else [-1 / self.amount_high] if key in self.high.index else [0] for key in df_stock_ret_total.columns}
        self.w_targ = pd.DataFrame.from_dict(dict_momentum)
        
        self.returns_targ = self.df_stock_ret_total @ self.w_targ.T
        
        self.mu_targ = (self.df_stock_ret_total @ self.w_targ.T).mean()
        self.std_targ = (self.df_stock_ret_total @ self.w_targ.T).std()
        
        self.sharpe_targ = (self.mu_targ - self.Rf) / self.std_targ

        self.df_mkt_pf = pd.DataFrame(self.returns_targ).merge(self.market_ret, on='Date')
        self.mu_mkt = self.df_mkt_pf[market_ticker].mean() * 252
        
        # avoid datatype issues
        self.low = pd.to_numeric(self.low, errors='coerce')
        self.high = pd.to_numeric(self.high, errors='coerce')
        self.beta_targ = self.low.mean() - self.high.mean()
        
        # cumulative returns
        self.cumret_targ_total = (1 + self.returns_targ).cumprod()
        self.cumret_mkt_total = (1 + self.df_mkt_pf[market_ticker]).cumprod()
        
        
    def returns_plot(self, do_plot=True):
        """Plots the returns over the entire time period for the BAB Portfolio and the market"""
        if do_plot:
            plt.figure(figsize=FS)
            plt.plot(self.df_mkt_pf[0], label='Target PF')
            plt.plot(self.df_mkt_pf[self.market_ticker], label='Market PF')
            plt.xlabel('Time')
            plt.ylabel('Return')
            plt.title('Returns for the period')
            plt.legend()
            plt.show()
        
        fig = plt.Figure(figsize=FS)
        ax = fig.add_subplot(111)
        ax.plot(self.df_mkt_pf[0], label='Target PF')
        ax.plot(self.df_mkt_pf[self.market_ticker], label='Market PF')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Return')
        ax.set_title('Returns for the period')
        ax.legend()

        return fig

    
    def cumret_plot_from_start(self, do_plot=True):
        """Plots the cumulative returns over the entire time period for the BAB Portfolio and the market"""
        if do_plot:
            plt.figure(figsize=FS)
            plt.plot(self.cumret_targ_total, label='Target PF')
            plt.plot(self.cumret_mkt_total, label='Market PF')
            plt.xlabel('Time')
            plt.ylabel('Return')
            plt.title('Cumulative Returns for the period')
            plt.legend()
            plt.show()

        fig = plt.Figure(figsize=FS)
        ax = fig.add_subplot(111)
        ax.plot(self.cumret_targ_total, label='Target PF')
        ax.plot(self.cumret_mkt_total, label='Market PF')
        ax.set_xlabel('Time')
        ax.set_ylabel('Return')
        ax.set_title('Cumulative Returns for the period')
        ax.legend()

        return fig
    
    
    def portfolio_specs(self):
        """Provides the expected return, the standard deviation and the Sharpe Ratio of the portfolio in a pd.DataFrame"""

        portfolio_specs_df = pd.DataFrame({'Expected Return': float(self.mu_targ*100), 'Standard Deviation': float(self.std_targ*100), 'Sharpe Ratio': float(self.sharpe_targ*100)}, index= ['values (in %)'])

        return portfolio_specs_df
    
    
    def get_weights(self):
        """Provides the weight allocation for each stock in a pd.DataFrame"""
        
        df_weights = pd.DataFrame(self.w_targ).T
        df_weights.columns = ['weight']
        return df_weights


    





