"""
2020/01/23
Stanislaw Grams <sjg@fmdx.pl>
09-3sat_project/src/main.py
"""
from dimacs import Dimacs
from algorithms import Genetic
from basic_types import Population

def main():
    """ main function """
    dimacs = Dimacs(filepath="../data/AIM/aim-50-1_6-yes1-4.cnf")
    equation = dimacs.equation

    ## test genetic algorithm
    generations = 250
    crossover_rate = 0.8
    mutation_rate = 0.01
    population_size = 500

    genetic_algorithm = Genetic(crossover_rate, mutation_rate)
    population = Population(population_size, equation)
    population.initialize()

    for _ in range(generations):
        population = genetic_algorithm.evolution(population)

    print(population.best.valid)
    print(population.best.fitness)

if __name__ == "__main__":
    main()
