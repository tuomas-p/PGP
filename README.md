# PGP: The Portfolio Generatio Program

The aim of this project is the development of an investment platform for personal use. The software is geared towards the construction of optimized financial portfolios from a pre-determined asset universe. The user is given the freedom to choose among a variety of standard, well-established, investment strategies, and, together with its personal risk-return preferences, the program will generate an optimal asset allocation. The overall infrastructure is user-friendly with a dedicated graphical interface that facilitates investment analytics and decisions. 

# User Guide
To effectively assist our users in navigating the project, we have prepared this user manual, which outlines the necessary steps for utilizing the PGP platform.

### PGP Preparations
If you are new to the PGP or have not used it for an extended period, please ensure that you have all the necessary dependencies installed in your Python environment. Take a moment to consult the pip/Conda documentation to learn how to install any missing dependencies.

### PGP Launch
After successfully installing the dependencies, you can now launch the PGP. Navigate to the main folder where you downloaded the .py files and execute the **main.py** file using the terminal command
```console
python main.py
```
If you are using the program for the first time or wish to update your return and information .csv files to ensure the latest data, please choose the "YES" option when prompted by the program. It is recommended to update the data whenever in doubt, as having the most recent information justifies the brief waiting period.

### PGP Usage
Now that you have reached the stock selection menu, you may select your desired stocks, countries, industries, and sectors where you want to invest in by clicking on the corresponding buttons. Do not forget to select at least one, or you will be met by a system error! Once done, you may click on the "OK" button.

You may then enter the date on which you want to decide to invest. For a new investment, proceed with today (or some close-by date). In order to check the status of a previous investment, you may put the actual investment date. The program will notice you if there are any issues with your date selection!

Now, the system will display the portfolio selection menu. From this point forward, the choices are in your hands. You can select one of the first three buttons to view graphs of different portfolios or choose "Portfolio Information" to access weight allocations and portfolio details for the three options. Remember to close the windows by using the X button, just like you would with any other window in your operating system, to proceed to the next view. If you accidentally closed a view prematurely and wish to return to the previous one, no worries! Once you have completed the views at least once, you can simply select the same option again from the portfolio menu to revisit it. (Note: For the mean-variance efficient portfolio and portfolio information, you will be prompted to input a target return or a target risk level. Feel free to choose your own values!)

Once you are done using the PGP for your current stock and date selection, please exit the program using the "Exit" button in the portfolio selection menu.

Thank you for reading through this guide, we wish you great success with your future investments!
