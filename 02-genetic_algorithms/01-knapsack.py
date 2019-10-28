# 2019/10/27
# 1-dimensional knapsack problem
# Stanislaw Grams <sjg@fmdx.pl>
# 02-genetic_algorithms/01-knapsack.py
import random
import numpy
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

data = [
  { 'name': 'zegar', 'value': 100, 'weight': 7 },
  { 'name': 'obraz-pejzaz', 'value': 300, 'weight': 7 },
  { 'name': 'obraz-portret', 'value': 200, 'weight': 6 },
  { 'name': 'radio', 'value': 40, 'weight': 2 },
  { 'name': 'laptop', 'value': 500, 'weight': 5 },
  { 'name': 'lapka nocna', 'value': 70, 'weight': 6 },
  { 'name': 'srebrne sztucce', 'value': 100, 'weight': 1 },
  { 'name': 'porcelana', 'value': 250, 'weight': 3 },
  { 'name': 'figura z brazu', 'value': 300, 'weight': 10 },
  { 'name': 'skorzana torebka', 'value': 280, 'weight': 3 },
  { 'name': 'odkurzacz', 'value': 300, 'weight': 15 }
]

population_size      = 200
generations          = 100
mutation_probability = 0.05
mate_rate            = 0.5
elitism              = True
weight_max           = 25
chromosome_length    = len (data)

creator.create ("Fitness", base.Fitness, weights=(1.0,))
creator.create ("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register ("attr_bool", random.randint, 0, 1)
toolbox.register ("individual", tools.initRepeat, creator.Individual,
        toolbox.attr_bool, chromosome_length)
toolbox.register ("population", tools.initRepeat, list, toolbox.individual)

def fitness(chromosome):
    value, weight = 0.0, 0.0
    for i in range (chromosome_length):
        if chromosome[i]:
            value  += data[i]['value']
            weight += data[i]['weight']
    if weight > weight_max:
        value = 0.0
    return [value]

toolbox.register ("evaluate", fitness)
toolbox.register ("mate", tools.cxTwoPoint)
toolbox.register ("mutate", tools.mutFlipBit, indpb=mutation_probability)
toolbox.register ("select", tools.selTournament, tournsize=3)

population      = toolbox.population (n=population_size)
best_individual = tools.HallOfFame (1)
statistics      = tools.Statistics (lambda ind: ind.fitness.values)

statistics.register ("avg", numpy.mean)
statistics.register ("max", numpy.max)

population, log = algorithms.eaSimple(population, toolbox, cxpb=mate_rate, mutpb=mutation_probability, ngen=generations, halloffame=best_individual, stats=statistics, verbose=False)
print ("max = " + str(log[99]['max']))

x = [row['gen'] for row in log]
y_max = [row['max'] for row in log]
y_avg = [row['avg'] for row in log]

figa, ax = plt.subplots()
ax.plot (x, y_avg, color='b', label='avg')
ax.plot (x, y_max, color='r', label='max')
ax.set_xlabel ("pokolenie")
ax.set_ylabel ("fitness (ocena)")
ax.legend ()
plt.show ()
