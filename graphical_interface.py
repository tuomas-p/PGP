import core
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import re
import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def stock_selector(sp500_firms):
    """define the menu window where you can select the stocks available according to name, sector, indusrty, country """
    # Function to add a stock to the list
    def add_stock(stock):
        if stock not in selected_stocks:
            selected_stocks.append(stock)

    # Function to add all stocks to the list
    def add_all_stocks():
        for stock in sp500_firms:
            add_stock(stock)
    
    # Function to close the window
    def close_window():
        if not selected_stocks and not detail_list_stocks:
            error_message.config(text="Please select at least one stock or category.", fg="red")
        else:
            root.destroy()

    # Function to add a detail (country/industry/sector) to the list
    def add_detail(detail):
        if detail not in detail_list_stocks:
            detail_list_stocks.append(detail)

    # Import info of stocks
    df = pd.read_csv('stock_info.csv')

    available_country = list(set(df.iloc[0].tolist()))
    available_country.remove('country')
    available_country = [x for x in available_country if isinstance(x, str)]

    available_industry = list(set(df.iloc[1].tolist()))
    available_industry.remove('industry')
    available_industry = [x for x in available_industry if isinstance(x, str)]

    available_sector = list(set(df.iloc[2].tolist()))
    available_sector.remove('sector')
    available_sector = [x for x in available_sector if isinstance(x, str)]

    # Create the main window
    root = tk.Tk()
    root.title("Stock Selector")

    # Create a list to store selected stocks and details
    selected_stocks = []
    detail_list_stocks = []

    # Create buttons for each stock
    row = 0
    col = 0
    for stock, (_, name) in sp500_firms.items():
        button = tk.Button(root, text=name, command=lambda stock=stock: add_stock(stock))
        button.grid(row=row, column=col, padx=10, pady=5)

        col += 1
        if col > 2:
            col = 0
            row += 1

    # Create a separator for the vertical line
    separator = ttk.Separator(root, orient='vertical')
    separator.grid(row=0, column=3, rowspan=row+1, padx=10, pady=5, sticky='ns')

    # Create labels for each detail type
    country_label = tk.Label(root, text="Country")
    country_label.grid(row=0, column=4, padx=10, pady=5, sticky='n')

    industry_label = tk.Label(root, text="Industry")
    industry_label.grid(row=0, column=6, padx=10, pady=5, sticky='n')

    sector_label = tk.Label(root, text="Sector")
    sector_label.grid(row=0, column=9, padx=10, pady=5, sticky='n')

    # Create buttons for available countries
    col_countries = 4
    row_countries = 1
    for country in available_country:
        button = tk.Button(root, text=country, command=lambda country=country: add_detail(country))
        button.grid(row=row_countries, column=col_countries, padx=10, pady=5, sticky='we')
        row_countries += 1

    # Create a separator for the vertical line
    separator = ttk.Separator(root, orient='vertical')
    separator.grid(row=0, column=5, rowspan=row+1, padx=10, pady=5, sticky='ns')

    # Create buttons for available industries (2 columns)
    col_industries = 6
    row_industries = 1
    for i, industry in enumerate(available_industry):
        button = tk.Button(root, text=industry, command=lambda industry=industry: add_detail(industry))
        button.grid(row=row_industries, column=col_industries + i % 2, padx=10, pady=5, sticky='we')
        row_industries = (row_industries + 1) if i % 2 == 1 else row_industries

    # Create a separator for the vertical line
    separator = ttk.Separator(root, orient='vertical')
    separator.grid(row=0, column=8, rowspan=row+1, padx=10, pady=5, sticky='ns')

    # Create buttons for available sectors
    col_sectors = 9
    row_sectors = 1
    for sector in available_sector:
        button = tk.Button(root, text=sector, command=lambda sector=sector: add_detail(sector))
        button.grid(row=row_sectors, column=col_sectors, padx=10, pady=5, sticky='we')
        row_sectors += 1

    # Create a separator for the vertical line
    separator = ttk.Separator(root, orient='vertical')
    separator.grid(row=0, column=11, rowspan=row+1, padx=10, pady=5, sticky='ns')

    # Create a label for display text
    dt1 = "Select the stocks you want to invest in.\n\n"
    dt2 = "The button 'Add All Stocks' adds all available stocks to your portfolio.\n\n"
    dt3 = "Stocks can be selected by the country, industry or sector they belong to. \n\n"
    dt4 = "Click on OK when all stocks of interest have been selected.\n\n"
    display_text=dt1+dt2+dt3+dt4
    
    display_label = tk.Label(root, text=display_text, anchor='n', justify='left')
    display_label.grid(row=0, column=12, rowspan=row+1, padx=10, pady=5, sticky='nw')

    # Create a button to add all stocks
    add_all_button = tk.Button(root, text="Add All Stocks", command=add_all_stocks)
    add_all_button.grid(row=row+1, column=1, padx=5, pady=5, sticky='we')
 
    
    # Create a label for displaying an error message
    error_message = tk.Label(root, text="", fg="red")
    error_message.grid(row=5, column=12, padx=10, pady=5, sticky='nw')


    # Create a button to close the window
    close_button = tk.Button(root, text="OK", command=close_window, width=10)
    close_button.grid(row=4, column=12, columnspan=1, padx=5, pady=5, sticky='nw')
    
    
    

    # Run the main event loop
    root.mainloop()
    
    #add stocks by the country, industry and/or sector selected:
    matching_headers = []
    for header in df.columns:
        for string in detail_list_stocks:
            if string in df[header].values:
                matching_headers.append(header)
                break
    selected_stocks=selected_stocks+matching_headers
    
    
    # Remove duplicate stocks and details from the selected lists
    selected_stocks = list(set(selected_stocks))
    detail_list_stocks = list(set(detail_list_stocks))

    return selected_stocks


def is_valid_date(date_string):
    try:
        datetime.datetime.strptime(date_string, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def date_selector():
    """define the date window where you can input the present date """
    def submit_date():
        year = year_entry.get()
        month = month_entry.get()
        day = day_entry.get()
        if len(month)==1: 
            month="0"+month
        if len(day)==1:
            day="0"+day
            
        date = f"{year}-{month}-{day}"
        if is_valid_date(date):
            window.date = date
            window.destroy()
        else:
            error_label.config(text="Invalid date", fg="red")

    # Create the main window
    window = tk.Tk()

    # Set window title
    window.title("Date Entry")

    # Create the display text
    display_text = tk.Label(window, text="Enter Date (YYYY-MM-DD): \n\n (tip: try 2023-01-01)")
    display_text.grid(row=0, column=0, padx=10, pady=10)

    # Create the rectangle frames
    frame1 = tk.Frame(window, width=200, height=50, bg="gray")
    frame2 = tk.Frame(window, width=200, height=50, bg="gray")
    frame3 = tk.Frame(window, width=200, height=50, bg="gray")

    # Position the rectangles using grid layout
    frame1.grid(row=1, column=0, padx=10, pady=10)
    frame2.grid(row=2, column=0, padx=10, pady=10)
    frame3.grid(row=3, column=0, padx=10, pady=10)

    # Create the entry fields inside the rectangles
    year_label = tk.Label(frame1,  text="Year     :")
    year_entry = tk.Entry(frame1)
    year_label.pack(side=tk.LEFT)
    year_entry.pack(side=tk.LEFT, padx=10)

    month_label = tk.Label(frame2, text="Month :")
    month_entry = tk.Entry(frame2)
    month_label.pack(side=tk.LEFT)
    month_entry.pack(side=tk.LEFT, padx=10)

    day_label = tk.Label(frame3,   text="Day     :")
    day_entry = tk.Entry(frame3)
    day_label.pack(side=tk.LEFT)
    day_entry.pack(side=tk.LEFT, padx=10)

    # Create the submit button
    submit_button = tk.Button(window, text="Submit", command=submit_date)
    submit_button.grid(row=4, column=0, padx=10, pady=10)

    # Create the label for displaying date or error message
    error_label = tk.Label(window, text="", fg="red")
    error_label.grid(row=5, column=0, padx=10, pady=10)

    # Run the main event loop
    window.mainloop()

    if hasattr(window, "date"):
        return window.date
    else:
        return None


def updater_user():
    """first window that appears to ask the user if they want to update the database"""
    # Create the main window
    window = tk.Tk()

    # Set window title
    window.title("UPDATER")

    # Create a label for the question
    question_label = tk.Label(window, text="Do you want to update the database? \n\n (it can take a minute.)")
    question_label.pack(padx=10, pady=10)

    # Function to handle button click
    def button_click(value):
        window.answer = value
        window.destroy()

    # Create a frame to hold the buttons
    button_frame = tk.Frame(window)
    button_frame.pack(padx=10, pady=10)

    # Create the "Yes" button
    yes_button = tk.Button(button_frame, text="Yes", command=lambda: button_click(True))
    yes_button.pack(side=tk.LEFT, padx=5)

    # Create the "No" button
    no_button = tk.Button(button_frame, text="No", command=lambda: button_click(False))
    no_button.pack(side=tk.LEFT, padx=5)

    # Run the main event loop
    window.mainloop()

    # Return the user's answer
    return window.answer


def define_mu_or_sigma():
    """define the window in which the user can choose between constraining the 
    MVE portfolio according to target return or target risk"""
    # Create the main window
    window = tk.Tk()

    # Set window title
    window.title("Return or Risk selection")

    # Create a label for the question
    question_label = tk.Label(window, text="Do you want to constrain your MVE portfolio based on return or risk?")
    question_label.pack(padx=10, pady=10)

    # Function to handle button click
    def button_click(value):
        window.answer = value
        window.destroy()

    # Create a frame to hold the buttons
    button_frame = tk.Frame(window)
    button_frame.pack(padx=10, pady=10)

    # Create the "Yes" button
    yes_button = tk.Button(button_frame, text="Return", command=lambda: button_click(True))
    yes_button.pack(side=tk.LEFT, padx=5)

    # Create the "No" button
    no_button = tk.Button(button_frame, text="Risk", command=lambda: button_click(False))
    no_button.pack(side=tk.LEFT, padx=5)

    # Run the main event loop
    window.mainloop()

    # Return the user's answer
    return window.answer

def define_return():
    """define the window in which the user can input a value for the return"""
    def submit_value():
        # Get the entered value from the entry widget
        value = entry.get()

        # Check if the value is a valid double
        try:
            float_value = float(value)
            message_label.config(text="")
            return_value.set(float_value)  # Set the return value
            window.destroy()  # Close the window
        except ValueError:
            message_label.config(text="Invalid return value", fg="red")

    # Create the main window
    window = tk.Tk()
    window.title("Enter Double Value")

    # Create an entry widget to enter the value
    entry = tk.Entry(window)
    entry.pack(padx=10, pady=10)

    # Create a submit button
    submit_button = tk.Button(window, text="Submit", command=submit_value)
    submit_button.pack(padx=10, pady=5)

    # Create a label to display error message
    message_label = tk.Label(window, fg="red")
    message_label.pack(padx=10, pady=5)

    # StringVar to store the return value
    return_value = tk.StringVar()

    # Run the main event loop
    window.mainloop()

    # Return the input value as a float
    return float(return_value.get())
    
      
def define_risk():
    """define the window in which the user can input a value for the risk"""
    def submit_value():
        # Get the entered value from the entry widget
        value = entry.get()

        # Check if the value is a valid double
        try:
            float_value = float(value)
            message_label.config(text="")
            return_value.set(float_value)  # Set the return value
            window.destroy()  # Close the window
        except ValueError:
            message_label.config(text="Invalid risk value", fg="red")

    # Create the main window
    window = tk.Tk()
    window.title("Enter Double Value")

    # Create an entry widget to enter the value
    entry = tk.Entry(window)
    entry.pack(padx=10, pady=10)

    # Create a submit button
    submit_button = tk.Button(window, text="Submit", command=submit_value)
    submit_button.pack(padx=10, pady=5)

    # Create a label to display error message
    message_label = tk.Label(window, fg="red")
    message_label.pack(padx=10, pady=5)

    # StringVar to store the return value
    return_value = tk.StringVar()

    # Run the main event loop
    window.mainloop()

    # Return the input value as a float
    return float(return_value.get())


def action_selector():
    """define the window in which the user can choose choose the portfolio or info he want to see (see the different button)"""
    def select_button(value):
        window.destroy()
        action_selector.selected_value = value

    window = tk.Tk()
    window.title("Portfolio and Informations")

    display_text = "Please select the portfolio or information you wish to review:"
    display_label = tk.Label(window, text=display_text)
    display_label.pack(padx=10, pady=10)

    button1 = tk.Button(window, text="Mean-Variance Efficient Portfolio", command=lambda: select_button("MVE"))
    button1.pack(padx=10, pady=5)

    button2 = tk.Button(window, text="Momentum Portfolio", command=lambda: select_button("MOM"))
    button2.pack(padx=10, pady=5)

    button3 = tk.Button(window, text="Betting Against Beta Portfolio", command=lambda: select_button("BAB"))
    button3.pack(padx=10, pady=5)

    button4 = tk.Button(window, text="Information about Portfolios", command=lambda: select_button("INFO"))
    button4.pack(padx=10, pady=5)
    
    button4 = tk.Button(window, text="Exit", command=lambda: select_button("EXIT"))
    button4.pack(padx=10, pady=5)

    action_selector.selected_value = None

    window.mainloop()

    return action_selector.selected_value


def call_mve(ext_stock_ret_, ext_stock_ret_total_, ext_stock_ret_future_, ext_stock_info_, Rf_, Rf_returns_,curr_date_, market_ret_, market_ret_future_):
    """create a Mean-Variance efficient portfolio and display it"""
    
    #"""Parameters : see the constructor of the associated class for the portfolio"""
        
        #choose either risk or return
    decision = define_mu_or_sigma()
    if decision:
        mu_targ = define_return()
        std_targ = None
    else:
        mu_targ = None
        std_targ = define_risk()

    PF = core.MVE_Portfolio(ext_stock_ret_, ext_stock_ret_total_, ext_stock_ret_future_, ext_stock_info_, Rf_, Rf_returns_, datetime.datetime.strptime(curr_date_, '%Y-%m-%d'), market_ret_, market_ret_future_, market_ticker='SPY', mu_targ=mu_targ, std_targ=std_targ)

    
    # Create the main window
    window = tk.Tk()
    window.title("Mean-Variance Efficient Portfolio")


    # Create a frame for the first plot
    frame1 = tk.Frame(window)
    frame1.grid(row=0, column=0)

    # Create the efficient frontier portfolio plot
    x = np.linspace(0, np.sqrt(PF.var_tang)+1, 1500)
    fig1 = Figure(figsize=(5, 4), dpi=100)
    ax1 = fig1.add_subplot()
    
    
    ax1.plot(np.sqrt((PF.A*(x)**2 - 2*PF.B*x + PF.C)/PF.D), x, label="Efficient Frontier")
    ax1.plot(x, abs(PF.sharpe_tang) * x + PF.Rf, label="CML")
    ax1.plot(np.sqrt(PF.var_tang), abs(PF.mu_tang), 'ro', label="tangent PF")
    ax1.plot(PF.std_targ, PF.mu_targ, 'bo', label="target PF")
    ax1.plot(np.sqrt(PF.var_glob_min), PF.mu_glob_min, 'go', label="global MV PF")
    ax1.set_xlabel('Standard Deviation')
    ax1.set_ylabel('Expected Return')
    ax1.set_title('Efficient Frontier and Capital Market Line')
    ax1.legend()

    canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
    canvas1.draw()
    toolbar1 = NavigationToolbar2Tk(canvas1, frame1)
    toolbar1.update()
    canvas1.get_tk_widget().pack()
    toolbar1.pack()

    # Create a frame for the second plot
    frame2 = tk.Frame(window)
    frame2.grid(row=0, column=1)

    # Create the security market line plot
    
    fig2 = Figure(figsize=(5, 4), dpi=100)
    ax2 = fig2.add_subplot()
    
    x = np.linspace(0, PF.beta_targ, 1500)
    ax2.plot(x, (PF.mu_mkt - PF.Rf) * x + PF.Rf, label='SML')
    ax2.plot(PF.beta_targ, PF.mu_targ, 'go', label="target PF")
    
    
    list_stock=ext_stock_info_.columns.tolist()
    for i in list_stock:
        beta=ext_stock_info_[i].iloc[3]
    
    
    ax2.set_xlabel('Beta')  
    ax2.set_ylabel('Expected Return')  
    ax2.set_title('Security Market Line')
    ax2.legend()

    canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
    canvas2.draw()
    toolbar2 = NavigationToolbar2Tk(canvas2, frame2)
    toolbar2.update()
    canvas2.get_tk_widget().pack()
    toolbar2.pack()

    # Create a frame for the third plot
    frame3 = tk.Frame(window)
    frame3.grid(row=0, column=2)

    # Create the return plot
    fig3 = Figure(figsize=(5, 4), dpi=100)
    ax3 = fig3.add_subplot()
    
    ax3.plot(PF.df_mkt_pf[0], label='Target PF')
    ax3.plot(PF.df_mkt_pf[PF.market_ticker], label='Market PF')
    ax3.vlines(PF.curr_date, min(min(PF.df_mkt_pf[0]), min(PF.df_mkt_pf[PF.market_ticker])), max(max(PF.df_mkt_pf[0]), max(PF.df_mkt_pf[PF.market_ticker])), color='black', label='Portfolio creation date')
    ax3.set_xlabel('Time')  
    
    ax3.set_ylabel('Return')  
    ax3.set_title('Returns for the period')
    ax3.legend()
    

    canvas3 = FigureCanvasTkAgg(fig3, master=frame3)
    canvas3.draw()
    toolbar3 = NavigationToolbar2Tk(canvas3, frame3)
    toolbar3.update()
    canvas3.get_tk_widget().pack()
    toolbar3.pack()

    # Create a frame for the fourth plot
    frame4 = tk.Frame(window)
    frame4.grid(row=1, column=0)

    # Create the cumulative return plot for the past
    fig4 = Figure(figsize=(5, 4), dpi=100)
    ax4 = fig4.add_subplot()
    
    ax4.plot(PF.cumret_targ_total, label='Target PF')
    ax4.plot(PF.cumret_mkt_total, label='Market PF')
    ax4.vlines(PF.curr_date, 0, max(max(PF.cumret_targ_total), max(PF.cumret_mkt_total)), color='black', label='Portfolio creation date')
    ax4.set_xlabel('Time')  
    ax4.set_ylabel('Return')  
    ax4.set_title('Cumulative Returns for the period')
    ax4.set_yscale("log")
    ax4.legend()

    canvas4 = FigureCanvasTkAgg(fig4, master=frame4)
    canvas4.draw()
    toolbar4 = NavigationToolbar2Tk(canvas4, frame4)
    toolbar4.update()
    canvas4.get_tk_widget().pack()
    toolbar4.pack()

    # Create a frame for the fifth plot
    frame5 = tk.Frame(window)
    frame5.grid(row=1, column=1)

    # Create the cumulative return plot for the future
    fig5 = Figure(figsize=(5, 4), dpi=100)
    ax5 = fig5.add_subplot()
    
    ax5.plot(PF.cumret_targ_future, label='Target PF')
    ax5.plot(PF.cumret_mkt_future, label='Market PF')
    ax5.set_xlabel('Time')  
    ax5.set_ylabel('Return') 
    ax5.set_title('Cumulative Returns for the future')
    ax5.set_yscale("log")
    ax5.legend()

    canvas5 = FigureCanvasTkAgg(fig5, master=frame5)
    canvas5.draw()
    toolbar5 = NavigationToolbar2Tk(canvas5, frame5)
    toolbar5.update()
    canvas5.get_tk_widget().pack()
    toolbar5.pack()

    # Run the main window loop
    window.mainloop()


def call_mom(ext_stock_ret, ext_stock_ret_total, ext_stock_ret_future, ext_stock_info, Rf, Rf_returns, curr_date, market_ret, market_ret_future):
    """create a Momentum portfolio and display it"""
    
    #"""Parameters : see the constructor of the associated class for the portfolio"""
    PF = core.Momentum_Portfolio(ext_stock_ret, ext_stock_ret_total, ext_stock_ret_future, ext_stock_info, Rf, Rf_returns, datetime.datetime.strptime(curr_date, '%Y-%m-%d'), market_ret, market_ret_future, market_ticker='SPY',percentage_winners=0.3, percentage_losers=0.3)
    
    
    # Create the main window
    window = tk.Tk()
    window.title("Momentum Portfolio Results")


    # Create a frame for the second plot
    frame2 = tk.Frame(window)
    frame2.grid(row=0, column=0)

    # Create the plot of the future return
    
    fig2 = Figure(figsize=(6, 6), dpi=100)
    ax2 = fig2.add_subplot()
    
    ax2.plot(PF.returns_targ_future, label='Target PF')
    ax2.plot(PF.market_ret_future, label='Market PF')
    ax2.set_xlabel('Time')  
    ax2.set_ylabel('Return')  
    ax2.set_title('Returns for the future')
    ax2.legend()

    canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
    canvas2.draw()
    toolbar2 = NavigationToolbar2Tk(canvas2, frame2)
    toolbar2.update()
    canvas2.get_tk_widget().pack()
    toolbar2.pack()

    # Create a frame for the third plot
    frame3 = tk.Frame(window)
    frame3.grid(row=0, column=1)

    # Create the cumulative return plot for the future
    fig3 = Figure(figsize=(6, 6), dpi=100)
    ax3 = fig3.add_subplot()
    
    ax3.plot(PF.cumret_targ_future, label='Target PF')
    ax3.plot(PF.cumret_mkt_future, label='Market PF')
    ax3.set_xlabel('Time')  
    ax3.set_ylabel('Return') 
    ax3.set_title('Cumulative Returns for the future')
    ax3.legend()


    canvas3 = FigureCanvasTkAgg(fig3, master=frame3)
    canvas3.draw()
    toolbar3 = NavigationToolbar2Tk(canvas3, frame3)
    toolbar3.update()
    canvas3.get_tk_widget().pack()
    toolbar3.pack()

    # Run the main window loop
    window.mainloop()


def call_bab(ext_stock_ret, ext_stock_ret_total, ext_stock_ret_future, ext_stock_info, Rf, Rf_returns, curr_date, market_ret, market_ret_future):
    """create a BAB portfolio and display it"""
    
    #"""Parameters : see the constructor of the associated class for the portfolio"""
    PF = core.BAB_Portfolio(ext_stock_ret_total, ext_stock_info, Rf, Rf_returns, market_ret, market_ticker='SPY',percentage_high=0.3, percentage_low=0.3)
    

    # Create the main window
    window = tk.Tk()
    window.title("Betting Against Beta Results")


    # Create a frame for the second plot
    frame2 = tk.Frame(window)
    frame2.grid(row=0, column=0)

    # Create the past return plot
    
    fig2 = Figure(figsize=(7, 6), dpi=100)
    ax2 = fig2.add_subplot()
    
    ax2.plot(PF.df_mkt_pf[0], label='Target PF')
    ax2.plot(PF.df_mkt_pf[PF.market_ticker], label='Market PF')
    ax2.set_xlabel('Time')  
    ax2.set_ylabel('Return')  
    ax2.set_title('Returns for the period')
    ax2.legend()

    canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
    canvas2.draw()
    toolbar2 = NavigationToolbar2Tk(canvas2, frame2)
    toolbar2.update()
    canvas2.get_tk_widget().pack()
    toolbar2.pack()

    # Create a frame for the third plot
    frame3 = tk.Frame(window)
    frame3.grid(row=0, column=1)

    # Create the cumulative return plot
    fig3 = Figure(figsize=(6, 6), dpi=100)
    ax3 = fig3.add_subplot()
    
    ax3.plot(PF.cumret_targ_total, label='Target PF')
    ax3.plot(PF.cumret_mkt_total, label='Market PF')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Return')
    ax3.set_title('Cumulative Returns for the period')
    ax3.legend()
    

    canvas3 = FigureCanvasTkAgg(fig3, master=frame3)
    canvas3.draw()
    toolbar3 = NavigationToolbar2Tk(canvas3, frame3)
    toolbar3.update()
    canvas3.get_tk_widget().pack()
    toolbar3.pack()

    # Run the main window loop
    window.mainloop()


def call_info(ext_stock_ret, ext_stock_ret_total, ext_stock_ret_future, ext_stock_info, Rf, Rf_returns, curr_date, market_ret, market_ret_future):
    """create a the 3 implemented portfolios and display their weight"""
    
    #"""Parameters : see the constructor of the associated class for the portfolio"""
    babPF = core.BAB_Portfolio(ext_stock_ret_total, ext_stock_info, Rf, Rf_returns, market_ret, market_ticker='SPY',percentage_high=0.3, percentage_low=0.3)
    momPF = core.Momentum_Portfolio(ext_stock_ret, ext_stock_ret_total, ext_stock_ret_future, ext_stock_info, Rf, Rf_returns, datetime.datetime.strptime(curr_date, '%Y-%m-%d'), market_ret, market_ret_future, market_ticker='SPY',percentage_winners=0.3, percentage_losers=0.3)
    
    #choose either risk or return
    decision = define_mu_or_sigma()
    if decision:
        mu_targ = define_return()
        std_targ = None
    else:
        mu_targ = None
        std_targ = define_risk()

    mvePF = core.MVE_Portfolio(ext_stock_ret, ext_stock_ret_total, ext_stock_ret_future, ext_stock_info, Rf, Rf_returns, datetime.datetime.strptime(curr_date, '%Y-%m-%d'), market_ret, market_ret_future, market_ticker='SPY', mu_targ=mu_targ, std_targ=std_targ)
    
    
    df_bab=babPF.portfolio_specs()
    df_bab_weight=babPF.get_weights()
    df_bab_weight.reset_index(inplace=True)
    
    df_mom=momPF.portfolio_specs()
    df_mom_weight=momPF.get_weights()
    df_mom_weight.reset_index(inplace=True)
    
    df_mve=mvePF.portfolio_specs()
    df_mve_weight=mvePF.get_weights()
    df_mve_weight.reset_index(inplace=True)
    
    names=list(df_bab_weight['index'])

    
    bab_weights=list(df_bab_weight['weight'])
    
    mom_weights=list(df_mom_weight['weight'])
    
    mve_weights=list(df_mve_weight['weight'])
    
    
    # Create the main window
    window = tk.Tk()
    window.title("Mean-Variance Efficient Portfolio")


    # Create a frame for the first plot
    frame1 = tk.Frame(window)
    frame1.grid(row=0, column=0)

    # Create the 3 plots for weights allocation
    
    fig1 = Figure(figsize=(6, 7), dpi=100)
    ax1 = fig1.add_subplot()
    
    
    ax1.bar(range(len(names)),mve_weights)
    ax1.set_xlabel('Stocks')
    ax1.set_ylabel('Wealth allocation')
    ax1.set_title('Wealth allocation for MVE portfolio')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=90,fontsize=7)

    canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
    canvas1.draw()
    toolbar1 = NavigationToolbar2Tk(canvas1, frame1)
    toolbar1.update()
    canvas1.get_tk_widget().pack()
    toolbar1.pack()

    # Create a frame for the second plot
    frame2 = tk.Frame(window)
    frame2.grid(row=0, column=1)
    
    fig2 = Figure(figsize=(6, 7), dpi=100)
    ax2 = fig2.add_subplot()
    
    ax2.bar(range(len(names)), mom_weights)
    ax2.set_xlabel('Stocks')

    ax2.set_title('Wealth allocation for Momentum portfolio')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=90,fontsize=7)
    
    
    canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
    canvas2.draw()
    toolbar2 = NavigationToolbar2Tk(canvas2, frame2)
    toolbar2.update()
    canvas2.get_tk_widget().pack()
    toolbar2.pack()

    # Create a frame for the third plot
    frame3 = tk.Frame(window)
    frame3.grid(row=0, column=2)


    fig3 = Figure(figsize=(6, 7), dpi=100)
    ax3 = fig3.add_subplot()
    
    ax3.bar(range(len(names)), bab_weights)
    ax3.set_xlabel('Stocks')
 
    ax3.set_title('Wealth allocation for BAB portfolio')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=90,fontsize=7)


    canvas3 = FigureCanvasTkAgg(fig3, master=frame3)
    canvas3.draw()
    toolbar3 = NavigationToolbar2Tk(canvas3, frame3)
    toolbar3.update()
    canvas3.get_tk_widget().pack()
    toolbar3.pack()

    # Run the main window loop
    window.mainloop()

    # Create the main window
    window = tk.Tk()
   
    # Create a Label widget for each value
    label = tk.Label(window, text="Stocks : ")

    # Set the position of each Label in the grid
    label.grid(row=0, column=0)
    
    # Create a Label widget for each value
    label = tk.Label(window, text="Wealth allocation MVE : ")

    # Set the position of each Label in the grid
    label.grid(row=0, column=1)
    
     # Create a Label widget for each value
    label = tk.Label(window, text="Wealth allocation Momentum : ")

    # Set the position of each Label in the grid
    label.grid(row=0, column=2)
    
     # Create a Label widget for each value
    label = tk.Label(window, text="Wealth allocation BAB : ")

    # Set the position of each Label in the grid
    label.grid(row=0, column=3)
    
    for i,name in enumerate(names):
        # Create a Label widget for each value
        label = tk.Label(window, text=names[i]+"  :  ")

        # Set the position of each Label in the grid
        label.grid(row=i+1, column=0)
        
        # Create a Label widget for each value
        label = tk.Label(window, text=np.round(mve_weights[i],3))

        # Set the position of each Label in the grid
        label.grid(row=i+1, column=1)
        
        # Create a Label widget for each value
        label = tk.Label(window, text=np.round(mom_weights[i],3))

        # Set the position of each Label in the grid
        label.grid(row=i+1, column=2)
        
         # Create a Label widget for each value
        label = tk.Label(window, text=np.round(bab_weights[i],3))

        # Set the position of each Label in the grid
        label.grid(row=i+1, column=3)

    # Start the main event loop
    window.mainloop()
    
    
    # Create the main window
    window = tk.Tk()
   
    grid=np.array([["","Expected Return" , "Standard Deviation" , "Sharpe Ratio"],
                   ["MVE : ",np.round(float(df_mve["Expected Return"]),2),np.round(float(df_mve["Standard Deviation"]),2) ,np.round(float(df_mve["Sharpe Ratio"]),2)],
                   ["MOM : ",np.round(float(df_mom["Expected Return"]),2),np.round(float(df_mom["Standard Deviation"]),2) ,np.round(float(df_mom["Sharpe Ratio"]),2)],
                   ["BAB : ",np.round(float(df_bab["Expected Return"]),2),np.round(float(df_bab["Standard Deviation"]),2) ,np.round(float(df_bab["Sharpe Ratio"]),2)]])
   
    for i in range(4):
        for j in range(4):
            
            # Create a Label widget for each value
            label = tk.Label(window, text=grid[i][j])

            # Set the position of each Label in the grid
            label.grid(row=i, column=j)

    # Start the main event loop
    window.mainloop()
