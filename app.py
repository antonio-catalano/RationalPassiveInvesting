import pandas as pd, numpy as np, matplotlib.pyplot as plt
from empyrical import max_drawdown, annual_return, cagr, omega_ratio
from matplotlib.ticker import FixedLocator
from scipy.stats import geom
import math
import streamlit as st
from matplotlib.pyplot import rc
from PIL import Image

df = pd.read_excel('stocks_bonds.xls', sheet_name = 'Returns by year', header = 17, usecols = 'A,B,D')
gold = pd.read_csv('gold_1950.csv')
gold.set_index('Date', inplace = True)
gold.index = pd.to_datetime(gold.index).to_period()

df1 = df[:91]
df1.set_index('Year', inplace = True)

gold_ret = gold.pct_change()[1:]
gold_ret.rename(columns = {'Price': 'Gold Pct change'}, inplace= True)

df1.index = pd.to_datetime(df1.index.map(str)).to_period()

common_index = gold_ret.index.intersection(df1.index)

overall = pd.concat([df1.loc[common_index], gold_ret], axis = 1)



def set_pub():
    font = {'family' : 'normal', 'weight':'bold'}
    rc('font', **font)    # bold fonts are easier to see
    #rc('grid', c='0.5', ls='-', lw=0.5)
    rc('figure', figsize = (18,14))
    plt.style.use('fivethirtyeight') #'fivethirtyeight'
    rc('lines', linewidth=3.2, color='b')

class portfolio_stationaryBootstrap:

    set_pub()
    TER = 0.006

    def __init__(self, data, w1_stock, w2_bond, w3_gold,
                 capital = 20000, holding_period = 10,
                 mean_block_length = 3, stress_test_frequency = False, stress_test_intensity = 1):
        self.data = data
        self.w1_stock = w1_stock
        self.w2_bond = w2_bond
        self.w3_gold = w3_gold
        self.capital = int(capital)
        self.holding_period = holding_period
        self.mean_block_length = mean_block_length
        self.stress_freq = stress_test_frequency
        self.stress_intensity = stress_test_intensity

    def single_sim(self):
        bootret_by_year = []
        cumulative_lengths = []
        data_panel = []
        for i in range(self.holding_period):
            index_start = self.data.sample().index[0]
            index_start = self.data.index.get_loc(index_start)
            L_i = self.holding_period + 1
            while L_i > self.holding_period:
                L_i = geom.rvs(p = 1/self.mean_block_length)
            cumulative_lengths.append(L_i)
            if sum(cumulative_lengths) > self.holding_period:
                L_final = self.holding_period - sum(cumulative_lengths[:-1])
                if L_final > len(self.data) - index_start:
                    diff = L_final - (len(self.data) - index_start)
                    subsample_generated = self.data.iloc[index_start-diff: (index_start-diff + L_final), :]
                else:
                    subsample_generated = self.data.iloc[index_start: index_start + L_final, :]
                data_panel.append(subsample_generated)
                break
            else:
                subsample_generated = self.data.iloc[index_start: index_start + L_i, :]
                if L_i > len(self.data) - index_start :
                    L_i = len(self.data) - index_start
                data_panel.append(subsample_generated)
                cumulative_lengths[-1] = L_i

        bootstrapSample = pd.concat([subsample for subsample in data_panel], axis = 0, ignore_index = True)

        if self.stress_freq:
            historical_ret_by_year = self.data @ np.array([self.w1_stock, self.w2_bond, self.w3_gold]).T
            year_min_ret = historical_ret_by_year.idxmin()
            for i in range(self.holding_period):
                extreme_event_dummy = True if np.random.rand() < 0.05 else False
                if extreme_event_dummy:
                    if self.stress_intensity == 1:
                        bootstrapSample.iloc[i,:] = self.data.loc[year_min_ret,:]
                    else:
                        bootstrapSample.iloc[i,:] = self.data.loc[year_min_ret,:]
                        bootstrapSample.iloc[i,:] *= 1.5

        total_ret_by_year = bootstrapSample @ np.array([self.w1_stock, self.w2_bond, self.w3_gold]).T
        total_ret_by_year -= self.TER

        portfolio_path = self.capital * np.cumprod(total_ret_by_year + 1)

        cagr = (portfolio_path.values[-1] / self.capital) ** (1/self.holding_period) - 1
        annual_volatility = total_ret_by_year.std()
        maxDrawdown = max_drawdown(pd.Series(total_ret_by_year))
        omega_ratio2 = omega_ratio(pd.Series(total_ret_by_year), required_return = 0.02, annualization = 1)
        omega_ratio4 = omega_ratio(pd.Series(total_ret_by_year), required_return = 0.04, annualization = 1)
        omega_ratio8 = omega_ratio(pd.Series(total_ret_by_year), required_return = 0.08, annualization = 1)
        return (np.insert(portfolio_path.values, 0, self.capital), cagr, annual_volatility, maxDrawdown,
                omega_ratio2, omega_ratio4, omega_ratio8)

    def graph_N_simulation(self, N_sim):

        fig, ax = plt.subplots()

        cagr_s = []
        vol_s = []
        portfolio_s = []
        max_drawdown_s = []
        OR_2 = []
        OR_4 = []
        OR_8 = []
        for i in range(N_sim):
            path, cagr, vol, md, omega_ratio2, omega_ratio4, omega_ratio8 = self.single_sim()
            portfolio_s.append(path)
            cagr_s.append(cagr)
            vol_s.append(vol)
            OR_2.append(omega_ratio2)
            OR_4.append(omega_ratio4)
            OR_8.append(omega_ratio8)
            max_drawdown_s.append(md)
            ax.plot(path, '--')
        worst_portfolios = sorted(np.array(portfolio_s), key = lambda array: array[-1])
        worst_5percent = np.array(worst_portfolios)[:, -1]
        max_drawdown_s = np.array(max_drawdown_s)

        OR_2 = np.array(OR_2)
        OR_4 = np.array(OR_4)
        OR_8 = np.array(OR_8)

        cagr_s = np.array(cagr_s)
        mean_cagr = cagr_s.mean()
        vol_cagr = cagr_s.std()
        best_cagr = cagr_s.max()
        worst_cagr = cagr_s.min()
        mean_vol = np.array(vol_s).mean()

        VaR95 = np.percentile(cagr_s, 5)
        CVaR95 = cagr_s[ cagr_s <= VaR95].mean()

        if self.capital - np.array(worst_portfolios)[0,-1] > 0:
            max_loss = self.capital - np.array(worst_portfolios)[0,-1]
        else:
            max_loss = 0

        loss_probability = (cagr_s < 0).sum() / len(cagr_s)

        fig.suptitle('Portfolio: {}% stocks - {}% bonds - {}% gold'.format(int(100*self.w1_stock), int(100*self.w2_bond), int(100*self.w3_gold)), size = 40)
        ax.set_title('%d stationary bootstrap runs' % N_sim, fontsize = 32)
        mean_sharpeRatio= round(mean_cagr / mean_vol, 3)
        stress_severity = None if self.stress_freq == False else ('High' if self.stress_intensity == 1.5 else 'Normal')
        ax.legend(['Initial Capital in t_0 = {} $ \
                   \nHolding period = {} years \
                   \nStress Test = {} \
                   \nStress Severity = {}'.format(self.capital, self.holding_period, self.stress_freq, stress_severity)],
                    handletextpad=2.0, handlelength=0, ncol = 1, prop={'size': 22});

        ax.set_xlabel('Year', fontsize = 24, labelpad = 12)
        ax.set_ylabel('Portfolio $', fontsize = 24, labelpad = 1)
        ax.axhline(self.capital, color = 'k', linewidth=1)
        ax.tick_params(axis='both', which='major', labelsize=16)
        yt = ax.get_yticks()
        yt = np.append(yt, self.capital)
        ax.set_yticks(yt)
        st.pyplot()

        values = [self.capital, N_sim, self.holding_period, self.TER, round(mean_cagr,3)*100, round(vol_cagr,3)*100, round(worst_cagr,3)*100, round(best_cagr,3)*100, round(mean_sharpeRatio,3)
                    ,round(loss_probability, 3)*100, round(max_loss, 3)]
        df = pd.DataFrame(values)

        df.index = ['Initial Capital', 'N° simulations', 'Holding period', 'Total Expense Ratio',
                      'Average CAGR', 'Volatility CAGR', 'Worst CAGR', 'Best CAGR',
                      'Average Geometric Sharpe Ratio', 'Probability CAGR < 0', 'Worst Portfolio Loss' ]
        df.columns = [['Stocks-Bonds-Gold'],
                    ['[  {}%  -  {}%  -  {}%  ]'.format(int(self.w1_stock*100), int(self.w2_bond*100), int(self.w3_gold*100))]]
        df.iloc[:3,:] = df.iloc[:3,:].applymap(lambda x: '{:.0f}'.format(x))
        df.iloc[3:8,:] = df.iloc[3:8,:].applymap(lambda x: str(round(x,2)) + ' %')
        df.loc['Total Expense Ratio'] = str(self.TER * 100) + ' %'

        df.loc[['Initial Capital', 'Worst Portfolio Loss']] = df.loc[['Initial Capital', 'Worst Portfolio Loss']].applymap(
                                                                    lambda x: str(x)+ ' $')
        df.loc['Holding period'] = df.loc['Holding period'].map(lambda x: str(x) + ' years')
        if df.loc['Probability CAGR < 0'].iloc[0] < 0.000001:
            df.loc['Probability CAGR < 0'].iloc[0] = 'Low: CAGR < 0 never verified in %d simulations'%N_sim
        else:
            df.loc['Probability CAGR < 0'] = df.loc['Probability CAGR < 0'].map(lambda x: str(round(x,2)) + ' %')

        df_MD = pd.DataFrame(100* np.array([max_drawdown_s.mean(), max_drawdown_s.std(),max_drawdown_s.min()]))
        df_MD = df_MD.applymap(lambda x: str(round(x,2))+' %')
        df_MD.index = ['Max Drawdown Average', 'Max Drawdown Volatility', 'Max Drawdown Min']
        df_MD.columns = [['Stocks-Bonds-Gold'],
                    ['[  {}%  -  {}%  -  {}%  ]'.format(int(self.w1_stock*100), int(self.w2_bond*100), int(self.w3_gold*100))]]

        df_OR_2 = pd.DataFrame(np.array([round(np.nanmean(OR_2),2), round(np.nanstd(OR_2),2)]))
        df_OR_2.index = ['Average Omega Ratio(2%)', 'Volatility Omega Ratio(2%)']
        df_OR_2.columns = [['Stocks-Bonds-Gold'],
                           ['[  {}%  -  {}%  -  {}%  ]'.format(int(self.w1_stock*100), int(self.w2_bond*100), int(self.w3_gold*100))]]
        df_OR_4 = pd.DataFrame(np.array([round(np.nanmean(OR_4),2), round(np.nanstd(OR_4),2)]))
        df_OR_4.index = ['Average Omega Ratio(4%)', 'Volatility Omega Ratio(4%)']
        df_OR_4.columns = [['Stocks-Bonds-Gold'],
                           ['[  {}%  -  {}%  -  {}%  ]'.format(int(self.w1_stock*100), int(self.w2_bond*100), int(self.w3_gold*100))]]
        df_OR_8 = pd.DataFrame(np.array([round(np.nanmean(OR_8),2), round(np.nanstd(OR_8),2)]))
        df_OR_8.index = ['Average Omega Ratio(8%)', 'Volatility Omega Ratio(8%)']
        df_OR_8.columns = [['Stocks-Bonds-Gold'],
                           ['[  {}%  -  {}%  -  {}%  ]'.format(int(self.w1_stock*100), int(self.w2_bond*100), int(self.w3_gold*100))]]

        df_var = pd.DataFrame(100*np.array([VaR95, CVaR95]))
        df_var.index = ['VaR(95%) {} years'.format(self.holding_period), 'CVaR(95%) {} years'.format(self.holding_period)]
        df_var = df_var.applymap(lambda x: str(round(x,2))+' %')
        df_var.columns = [['Stocks-Bonds-Gold'],
                           ['[  {}%  -  {}%  -  {}%  ]'.format(int(self.w1_stock*100), int(self.w2_bond*100), int(self.w3_gold*100))]]
        total_stats = pd.concat([df, df_MD, df_OR_2, df_OR_4, df_OR_8, df_var], axis = 0)

        return total_stats

def joint_sim(w1,w2,w3, w11, w22, w33, capital, hp, stress_freq, stress_int):
    s1 = portfolio_stationaryBootstrap(overall, w1/100, w2/100, w3/100, capital, holding_period = hp, stress_test_frequency = stress_freq, stress_test_intensity = stress_int)
    s2 = portfolio_stationaryBootstrap(overall, w11/100, w22/100, w33/100, capital, holding_period = hp, stress_test_frequency = stress_freq, stress_test_intensity = stress_int)
    def s():
        gen = np.random.choice(1_000_000)
        np.random.seed(gen)
        a = s1.graph_N_simulation(N_sim = 100)
        np.random.seed(gen)
        b = s2.graph_N_simulation(N_sim = 100)
        return a,b
    return s()

'''  # How much is the benefit of diversification for a passive portfolio?
## We try to quantify it through historical data from 1950 by implementing a stationary bootstrap method.

You are going to select 2 portfolios with different weights and the algorithm
locks the same random state generator of stochastic path for both sets of weights.
In this way you can evaluate two sets of weights _ceteris paribus_.
Each time you select the two different sets of weights the algorith will refresh the random state generator,
always by locking the same stochastic generator for both asset allocations.

    You can choose weights among 3 assets:

1. An ETF that tracks the S&P500 index
2. An ETF that tracks the US 10-year T-bonds
3. An ETF/ETC on Gold

---
'''


image = Image.open('wallstreet.jpg')
st.sidebar.image(image, caption='', use_column_width=True)

st.sidebar.header('Build a robust passive portfolio')
st.sidebar.subheader('Choose the option to visualize')

sim = st.sidebar.checkbox('Simulation', value = True )
modelrational = st.sidebar.checkbox('The explanation of the method (unselect to come back to simulation)', value = False)
considerations = st.sidebar.checkbox('Hypothesis and results (unselect to come back to simulation)', value = False)

if modelrational or considerations:
    sim = False

if sim:
    capital = st.selectbox('Select the initial capital of both portfolios ', np.array(['20,000$','100,000$','500,000$']))
    capital = int(capital.replace(',','').replace('$',''))

    hp = st.slider('Select the Holding Period (years)', min_value = 5, max_value = 50, value = 10)

    st.markdown('''### Portfolio 1''')

    w1 = st.slider('Select Stocks Weight (%)', min_value = 0, max_value = 100, value = 60)
    w2 = st.slider('Select Bonds Weight (%)',min_value = 0, max_value = 100, value = 25)
    w3 = st.slider('Select Gold Weight (%)',min_value = 0, max_value = 100, value = 15)

    st.markdown('''### Portfolio 2''')

    w11 = st.slider('Stocks Weight (%)', min_value = 0, max_value = 100, value = 30)
    w22 = st.slider('Bonds Weight (%)',min_value = 0, max_value = 100, value = 60)
    w33 = st.slider('Gold Weight (%)',min_value = 0, max_value = 100, value = 10)


    if w1 + w2 + w3 != 100:
        st.error('The sum of weights must be 100!')
    else:
        stressTest = st.selectbox('Do you want to add a stress test procedure? (Advisable)', ('Yes', 'No'))

        stressTest = True if stressTest == 'Yes' else False
        if stressTest == True:
            stressCoef = st.selectbox('Select the grade of stress test severity', ('Normal', 'High'))
            stressCoef = 1 if stressCoef == 'Normal' else 1.5
        else:
            stressCoef = 1

        sim1, sim2 = joint_sim(w1, w2, w3, w11, w22, w33, capital, hp, stressTest, stressCoef)

        '''### Portfolio 1'''
        st.dataframe(sim1)
        '''### Portfolio 2'''
        st.dataframe(sim2)
if modelrational:
    '''# The explanation of the method'''

    st.markdown('''I used two datasets:
* http://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html
for S&P 500  and T-bonds historical total returns.
* https://www.macrotrends.net/ for gold historical returns.

We have a matrix of 69 rows (from 1950 to 2018 included) and 3 columns (S&P 500 index total return, T-bonds total return,
gold return) containing the total annual returns of the assets.
Each row represents a potential extraction in the bootstrap procedure.

In order to take into account the dependence structure of asset returns I've used a special moving block length bootstrap procedure, named *stationary block bootstrap*.
This procedure consists of an overlapping moving block length bootstrap where the length of each block is not fixed, instead it follows a geometric distribution with $p = 1/m$ , where $m$
is the average block length (I chose $m = 3$).

The method runs 100 simulations for each *holding period* path.

The results of the methods take into account the costs of passive investing.
I estimated these costs by deducting a 0.6% yearly.
It's higher then the current average Total Expense Ratio (TER) of about 0.4% , but in this way
we take into account also trading costs (very low given the hold and buy strategy).

In order to stress the strategy I used an historical stress test approach.
If you choose **Yes** in the corresponding box, the model first generates
the bootstrap sample, then, for each year (so for each 3-multivariate returns), it replaces with a probability of 5%
that year-returns of the bootstrap sample with the year-returns where the historical returns of the 3 assets give
the worst overall return for that strategy allocation:
in other words, for each set of weights that you choose, the algorithm finds the year of
the original historical dataset (that has 69 rows)
where your portfolio had the biggest loss, it stores that year, and then for each year of
the new bootstrap dataset (that has only _holding period_ rows) it applies a uniform
distribution (0,1) generator to decide whether or not to proceed to the substitution (proceed if the generated random number
< 0.05).

If you choose the *severe* test the algorithm applies the same rule described above but it multiplies the
returns of the worst year with a coefficient of 1.5, not 1.
 ---
''')

if considerations:
    '''# Hypothesis and results'''
    st.markdown('''Before we reach conclusions let's clarify which are the hypothesis.
Too often the hypothesis of a model are overlooked, but if hypothesis aren't realistic, results won't be realistic neither.
### Hypothesis of the model:
1. Buy & hold strategy for the whole holding period
2. The empirical future annual returns distribution of the 3 assets won't be dramatically different compared to that of the past 69 years.
This means that in the future the stock market can loose even 50% one year, but the model doesn't take into account the fact
that the stock market can loose 100% or that both stocks and bonds loose 30% in one year.
3. Historical stock and bonds data only concern the US market. Hence the bootstrap portfolio of our model is only a benchmark portfolio.
It is useful to compare different weights among the 3 principal assets (except cash) but it doesn't represent a fully diversified portfolio.

Having said that, let's proceed to some considerations:
* Gold is an insurance asset: its average CAGR is only slightly positive with high annual return volatility (indeed geometric sharpe ratio is almost null: that means
that it is NOT an insurance asset if held individually).
But you can try by yourself that a portfolio without a certain per cent
of gold (I guess at least 8-10%) is not pareto efficient.
* If you increase stocks weight the average CAGR increases (and also the average Omega ratio at 8%), but also volatility CAGR, VaR and CVaR increase.
* Above a certain stock weight threshold (I estimate about 50%), the risk increases more than the increase of the expected return: geometric sharpe ratio begins to drop.
Moreover max drawdown metrics begin to increase (in absolute value) in a clear manner.
* If you increase stock weight beyond a certain threshold (I estimate 85-90%), the portfolio is not anymore pareto efficient (i.e. you can build another
portfolio with the same expected CAGR but less risky).
* Based on my simulations, robust portfolios are weigth sets like these: (35% - 50% - 15%), (40% - 40%- 20%), (45% - 30% - 25%).
 Aggressive and pareto efficient portfolios are sets like these: (70% - 15% - 15%), (60% - 30% - 10%).
  ''')
