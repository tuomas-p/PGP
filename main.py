import graphical_interface as gi
import core
import os

# dict of stock ticker constants with their permnos and company names
sp500_firms = {
    'AAPL': (14593, 'Apple Inc.'),
    'MSFT': (10107, 'Microsoft Corporation'),
    'GOOGL': (26687, 'Alphabet Inc. Class A'),
    'AMZN': (84788, 'Amazon.com, Inc.'),
    'TSLA': (73107, 'Tesla, Inc.'),
    'JPM': (40772, 'JPMorgan Chase & Co.'),
    'JNJ': (22342, 'Johnson & Johnson'),
    'PG': (21282, 'Procter & Gamble Company'),
    'V': (41779, 'Visa Inc.'),
    'UNH': (79191, 'UnitedHealth Group Incorporated'),
    'MA': (47261, 'Mastercard Incorporated'),
    'HD': (32581, 'The Home Depot, Inc.'),
    'DIS': (23764, 'The Walt Disney Company'),
    'PYPL': (71547, 'PayPal Holdings, Inc.'),
    'CMCSA': (21508, 'Comcast Corporation Class A'),
    'VZ': (14809, 'Verizon Communications Inc.'),
    'KO': (13082, 'The Coca-Cola Company'),
    'ADBE': (60574, 'Adobe Inc.'),
    'INTC': (24506, 'Intel Corporation'),
    'PFE': (21682, 'Pfizer Inc.'),
    'T': (11474, 'AT&T Inc.'),
    'ABT': (10004, 'Abbott Laboratories'),
    'MRK': (26147, 'Merck & Co., Inc.'),
    'NFLX': (65225, 'Netflix, Inc.'),
    'GS': (35920, 'The Goldman Sachs Group, Inc.'),
    'IBM': (19920, 'International Business Machines Corporation'),
    'CSCO': (14053, 'Cisco Systems, Inc.'),
    'HON': (18408, 'Honeywell International Inc.'),
    'CVX': (18542, 'Chevron Corporation'),
    'ORCL': (18749, 'Oracle Corporation'),
    'BA': (11033, 'The Boeing Company'),
    'MMM': (12603, '3M Company'),
    'AXP': (16547, 'American Express Company'),
    'MCD': (16605, "McDonald's Corporation"),
    'JCI': (33467, 'Johnson Controls International plc'),
    'UNP': (16790, 'Union Pacific Corporation'),
    'ACN': (68319, 'Accenture plc Class A'),
    'NEE': (71922, 'NextEra Energy, Inc.'),
    'UPS': (88457, 'United Parcel Service, Inc.'),
    'XOM': (12060, 'Exxon Mobil Corporation'),
    'GE': (11703, 'General Electric Company')
}


if __name__ == "__main__":
    #first we call an update to import files if there are not present
    do_update = gi.updater_user()
    list_files = ["rf_rate.csv","stock_info.csv","stock_returns.csv"]
    for file in list_files:
         if not(os.path.isfile(file)): 
                do_update=True
    if do_update:
        core.updater(list(sp500_firms.keys()))
        core.updater_rf_rate("^IRX")


    #then the windows for the stocks selection is open
    selected_stock = gi.stock_selector(sp500_firms)
    ext_stock_info = core.ext_and_merger_info("stock_info.csv", selected_stock)


    #select the present date
    selected_date = gi.date_selector()

    #formalise all available and needed informations for the construction of portfolio
    curr_date = selected_date
    ext_stock_ret = core.ext_and_merger("stock_returns.csv", selected_stock, start='2010', end=curr_date)
    ext_stock_ret_total = core.ext_and_merger("stock_returns.csv", selected_stock, start='2010')
    ext_stock_ret_future = core.ext_and_merger("stock_returns.csv", selected_stock, start=curr_date)
    ext_stock_info = core.ext_and_merger_info("stock_info.csv", selected_stock)
    Rf = core.get_rf_rate()
    Rf_returns = core.ext_and_merger_rf("rf_rate.csv", start='2000-01-01')
    market_ret = core.get_market_returns()
    market_ret_future = core.get_market_returns(start=curr_date)



    #select action: portfolio and infos
    action=None
    while(action!="EXIT"):
        action = gi.action_selector()
        if action=="MVE":
            gi.call_mve(ext_stock_ret, ext_stock_ret_total, ext_stock_ret_future, ext_stock_info, Rf, Rf_returns,curr_date, market_ret, market_ret_future)
        if action=="MOM":
            gi.call_mom(ext_stock_ret, ext_stock_ret_total, ext_stock_ret_future, ext_stock_info, Rf, Rf_returns, curr_date, market_ret, market_ret_future)
        if action=="BAB":
            gi.call_bab(ext_stock_ret, ext_stock_ret_total, ext_stock_ret_future, ext_stock_info, Rf, Rf_returns, curr_date, market_ret, market_ret_future)
        if action=="INFO":
            gi.call_info(ext_stock_ret, ext_stock_ret_total, ext_stock_ret_future, ext_stock_info, Rf, Rf_returns, curr_date, market_ret, market_ret_future)
