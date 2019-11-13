# Rational Passive Investing

================

An asset allocation app useful to study how to diversify a passive portfolio.

Through this app you select two sets of weights for two different buy & hold portfolios and the app
runs a **stationary bootstrap** method for both portfolios; the algorithm
locks the same random state generator of stochastic path for both sets of weights.
In this way you can evaluate two passive portfolios _ceteris paribus_.
Each time you select the two different sets of weights the algorith will refresh the random state generator,
always by locking the same stochastic generator for both asset allocations.

    You can choose weights among 3 assets:

1. An ETF that tracks the S&P500 index
2. An ETF that tracks the US 10-year T-bonds
3. An ETF/ETC on Gold

The model takes into account costs (except taxes) and severe scenarios.

You can find more informations in the sections *The explanation of the model* and
*Hypothesis and results*.


Demo
----


![RationalPassiveInvesting Demo](demo/samplePassiveInvestingApp.gif)

 


**Requirements**

Python 3.6 version or superior

 

**How to run this demo**

1.  cd to the directory where *requirements.txt* is located

2.  activate your virualenv

3.  run: `pip install -r requirements.txt` in your shell

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
git clone https://github.com/antonio-catalano/PassivePortfolioStrategy.git
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



cd into the project folder


'path'\>cd PassivePortfolioStrategy

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

streamlit run app.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
