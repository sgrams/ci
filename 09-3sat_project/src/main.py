"""
2020/01/23
Stanislaw Grams <sjg@fmdx.pl>
09-3sat_project/src/main.py
"""
from dimacs import Dimacs
from algorithms import StandardGenetic

def main():
    """ main function """
    dimacs = Dimacs(filepath="../data/AIM/aim-100-6_0-yes1-2.cnf")
    equation = dimacs.equation

    ## test genetic algorithm
    generations = 1000
    crossover_rate = 0.9
    mutation_rate = 0.001
    population_size = 200
    elitism = True

    genetic_algorithm = StandardGenetic([crossover_rate, mutation_rate], elitism,
                                        population_size, generations)
    best_population = genetic_algorithm.run(equation)

    print(elitism)
    print(equation.variables)
    print(equation.clauses)
    print(best_population.best.valid)
    print(best_population.best.fitness)

if __name__ == "__main__":
    main()
