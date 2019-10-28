import math
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

## import csv
dataset = pd.read_csv("miasta.csv")
print (dataset)

## new row
new_row = [2010, 460, 555, 405]
dataset = dataset.to_numpy ()
dataset = np.vstack([dataset, new_row])
print (dataset)

x = dataset[:, [0]]
y = dataset[:, [1]]
y_gda = dataset[:, [1]]
y_poz = dataset[:, [2]]
y_szc = dataset[:, [3]]

figa, ax = plt.subplots()
ax.plot (x, y)
ax.scatter (x, y, color='r')
ax.set_xlabel ("Lata")
ax.set_ylabel ("Liczba ludnosci [w tys.]")
ax.set_title ("Ludnosc w miastach Polski")

figb, bx = plt.subplots();
bx.plot (x, y_gda, label='Gdansk')
bx.plot (x, y_poz, label='Poznan')
bx.plot (x, y_szc, label='Szczecin')

bx.set_xlabel ("Lata")
bx.set_ylabel ("Liczba ludnosci [w tys.]")
bx.set_title ("Ludnosc w miastach Polski")
bx.legend ()

plt.show ()
