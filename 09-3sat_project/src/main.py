"""
2020/01/23
Stanislaw Grams <sjg@fmdx.pl>
09-3sat_project/src/main.py
"""
from timeit import default_timer as timer
from dimacs import Dimacs
from algorithms import StandardGenetic
from algorithms import AdaptiveGenetic
from algorithms import DPLL

def main():
    """ main function """
    dimacs = Dimacs(filepath="../data/uf20/uf20-091.cnf")
    equation = dimacs.parse()
    is_file_valid = dimacs.validate()
    print(is_file_valid)

    ## test genetic algorithm
    #crossover_rate = 0.9
    #mutation_rate = 0.08
    population_size = 80
    generations = 10000
    #elitism = True

    adaptive_algorithm = AdaptiveGenetic(10, population_size, generations)
    best_population = adaptive_algorithm.run(equation, verbose=True)

    #genetic_algorithm = StandardGenetic([crossover_rate, mutation_rate], elitism,
    #                                    population_size, generations)
    #start_time = timer()
    #best_population = genetic_algorithm.run(equation, verbose=True)
    #end_time = timer()

    #print(elitism)
    #print(equation.variables)
    #print(equation.clauses)
    print(best_population.best.valid)
    print(best_population.best.fitness)
    #print("time = " + str(end_time - start_time))

    #dpll = DPLL()
    #solution = dpll.run(equation)

    #print(solution)
    #print(equation.validate(solution))

if __name__ == "__main__":
    main()
