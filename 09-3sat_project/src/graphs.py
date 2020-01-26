"""
2020/01/25
Stanislaw Grams <sjg@fmdx.pl>
09-3sat_project/src/graphs.py
"""
import matplotlib.pyplot as plt
import pandas as pd

## M300_L50
dataframe_dpll = pd.read_csv('csvs/M300_L50/DPLL_M300_L50-20200125214008-results.csv', header=None)
dataframe_sga = pd.read_csv('csvs/M300_L50/SGA_M300_L50-20200125214008-results.csv', header=None)
dataframe_iaga = pd.read_csv('csvs/M300_L50/IAGA_M300_L50-20200125214008-results.csv', header=None)
m300_l50 = pd.DataFrame([dataframe_dpll, dataframe_sga, dataframe_iaga])

m300_l50_dpll_time = m300_l50[0][0][0].mean()
m300_l50_sga_time = m300_l50[0][1][0].mean()
m300_l50_sga_avg_fit = m300_l50[0][1][1].mean()
m300_l50_sga_max_fit = m300_l50[0][1][1].max()
m300_l50_iaga_time = m300_l50[0][2][0].mean()
m300_l50_iaga_avg_fit = m300_l50[0][2][1].mean()
m300_l50_iaga_max_fit = m300_l50[0][2][1].max()

## M170_L50
dataframe_dpll = pd.read_csv('csvs/M170_L50/DPLL_M170_L50-20200125221629-results.csv', header=None)
dataframe_sga = pd.read_csv('csvs/M170_L50/SGA_M170_L50-20200125221629-results.csv', header=None)
dataframe_iaga = pd.read_csv('csvs/M170_L50/IAGA_M170_L50-20200125221629-results.csv', header=None)
m170_l50 = pd.DataFrame([dataframe_dpll, dataframe_sga, dataframe_iaga])

m170_l50_dpll_time = m170_l50[0][0][0].mean()
m170_l50_sga_time = m170_l50[0][1][0].mean()
m170_l50_sga_avg_fit = m170_l50[0][1][1].mean()
m170_l50_sga_max_fit = m170_l50[0][1][1].max()
m170_l50_iaga_time = m170_l50[0][2][0].mean()
m170_l50_iaga_avg_fit = m170_l50[0][2][1].mean()
m170_l50_iaga_max_fit = m170_l50[0][2][1].max()

## M100_L50
dataframe_dpll = pd.read_csv('csvs/M100_L50/DPLL_M100_L50-20200125230325-results.csv', header=None)
dataframe_sga = pd.read_csv('csvs/M100_L50/SGA_M100_L50-20200125230325-results.csv', header=None)
dataframe_iaga = pd.read_csv('csvs/M100_L50/IAGA_M100_L50-20200125230325-results.csv', header=None)
m100_l50 = pd.DataFrame([dataframe_dpll, dataframe_sga, dataframe_iaga])

m100_l50_dpll_time = m100_l50[0][0][0].mean()
m100_l50_sga_time = m100_l50[0][1][0].mean()
m100_l50_sga_avg_fit = m100_l50[0][1][1].mean()
m100_l50_sga_max_fit = m100_l50[0][1][1].max()
m100_l50_iaga_time = m100_l50[0][2][0].mean()
m100_l50_iaga_avg_fit = m100_l50[0][2][1].mean()
m100_l50_iaga_max_fit = m100_l50[0][2][1].max()

## M80_L50
dataframe_dpll = pd.read_csv('csvs/M80_L50/DPLL_M80_L50-20200125232838-results.csv', header=None)
dataframe_sga = pd.read_csv('csvs/M80_L50/SGA_M80_L50-20200125232838-results.csv', header=None)
dataframe_iaga = pd.read_csv('csvs/M80_L50/IAGA_M80_L50-20200125232838-results.csv', header=None)
m80_l50 = pd.DataFrame([dataframe_dpll, dataframe_sga, dataframe_iaga])

m80_l50_dpll_time = m80_l50[0][0][0].mean()
m80_l50_sga_time = m80_l50[0][1][0].mean()
m80_l50_sga_avg_fit = m80_l50[0][1][1].mean()
m80_l50_sga_max_fit = m80_l50[0][1][1].max()
m80_l50_iaga_time = m80_l50[0][2][0].mean()
m80_l50_iaga_avg_fit = m80_l50[0][2][1].mean()
m80_l50_iaga_max_fit = m80_l50[0][2][1].max()

## M10_L22
dataframe_dpll = pd.read_csv('csvs/M10_L22/DPLL_M10_L22-20200125235936-results.csv', header=None)
dataframe_sga = pd.read_csv('csvs/M10_L22/SGA_M10_L22-20200125235936-results.csv', header=None)
dataframe_iaga = pd.read_csv('csvs/M10_L22/IAGA_M10_L22-20200125235936-results.csv', header=None)
m10_l22 = pd.DataFrame([dataframe_dpll, dataframe_sga, dataframe_iaga])

m10_l22_dpll_time = m10_l22[0][0][0].mean()
m10_l22_sga_time = m10_l22[0][1][0].mean()
m10_l22_sga_avg_fit = m10_l22[0][1][1].mean()
m10_l22_sga_max_fit = m10_l22[0][1][1].max()
m10_l22_iaga_time = m10_l22[0][2][0].mean()
m10_l22_iaga_avg_fit = m10_l22[0][2][1].mean()
m10_l22_iaga_max_fit = m10_l22[0][2][1].max()

## M91_L20
dataframe_dpll = pd.read_csv('csvs/M91_L20/DPLL_M91_L20-20200125235432-results.csv', header=None)
dataframe_sga = pd.read_csv('csvs/M91_L20/SGA_M91_L20-20200125235432-results.csv', header=None)
dataframe_iaga = pd.read_csv('csvs/M91_L20/IAGA_M91_L20-20200125235432-results.csv', header=None)
m91_l20 = pd.DataFrame([dataframe_dpll, dataframe_sga, dataframe_iaga])

m91_l20_dpll_time = m91_l20[0][0][0].mean()
m91_l20_sga_time = m91_l20[0][1][0].mean()
m91_l20_sga_avg_fit = m91_l20[0][1][1].mean()
m91_l20_sga_max_fit = m91_l20[0][1][1].max()
m91_l20_iaga_time = m91_l20[0][2][0].mean()
m91_l20_iaga_avg_fit = m91_l20[0][2][1].mean()
m91_l20_iaga_max_fit = m91_l20[0][2][1].max()

## graph 1: all algorithms, all times per number of variables
## prepare data
x = ['m91_l20', 'm10_l22', 'm80_l50', 'm100_l50', 'm170_l50', 'm300_l50']

dpll_times = [m91_l20_dpll_time, m10_l22_dpll_time, m80_l50_dpll_time, m100_l50_dpll_time, m170_l50_dpll_time, m300_l50_dpll_time]
sga_times = [m91_l20_sga_time, m10_l22_sga_time, m80_l50_sga_time, m100_l50_sga_time, m170_l50_sga_time, m300_l50_sga_time]
iaga_times = [m91_l20_iaga_time, m10_l22_iaga_time, m80_l50_iaga_time, m100_l50_iaga_time, m170_l50_iaga_time, m300_l50_iaga_time]

sga_max_fitnesses = [m91_l20_sga_max_fit, m10_l22_sga_max_fit, m80_l50_sga_max_fit, m100_l50_sga_max_fit, m170_l50_sga_max_fit, m300_l50_sga_max_fit]
iaga_max_fitnesses = [m91_l20_iaga_max_fit, m10_l22_iaga_max_fit, m80_l50_iaga_max_fit, m100_l50_iaga_max_fit, m170_l50_iaga_max_fit, m300_l50_iaga_max_fit]

sga_avg_fitnesses = [m91_l20_sga_avg_fit, m10_l22_sga_avg_fit, m80_l50_sga_avg_fit, m100_l50_sga_avg_fit, m170_l50_sga_avg_fit, m300_l50_sga_avg_fit]
iaga_avg_fitnesses = [m91_l20_iaga_avg_fit, m10_l22_iaga_avg_fit, m80_l50_iaga_avg_fit, m100_l50_iaga_avg_fit, m170_l50_iaga_avg_fit, m300_l50_iaga_avg_fit]

fig, ax = plt.subplots()
ax.semilogy(x, dpll_times, label='DPLL')
ax.semilogy(x, sga_times, label='SGA')
ax.semilogy(x, iaga_times, label='IAGA')

ax.set_title("All algorithms, all times per variables, n_iterations=50")
#ax.xaxis.set_label_text("")

plt.legend()
plt.savefig('../report/img/plot1.png')
fig.clf()

fig, ax = plt.subplots()
ax.plot(x, sga_max_fitnesses, label='SGA max_fit')
ax.plot(x, iaga_max_fitnesses, label='IAGA max_fit')
ax.set_title("SGA and IAGA max fitnesses according to dataset, n_iterations=50")
plt.legend()
plt.savefig('../report/img/plot2.png')
fig.clf()

fig, ax = plt.subplots()
ax.plot(x, sga_avg_fitnesses, label='SGA avg_fit')
ax.plot(x, iaga_avg_fitnesses, label='IAGA avg_fit')
ax.set_title("SGA and IAGA mean fitnesses according to dataset, n_iterations=50")
plt.legend()
plt.savefig('../report/img/plot3.png')
fig.clf()
