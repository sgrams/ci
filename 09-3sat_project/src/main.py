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
    dimacs = Dimacs(filepath="../data/test/uf20-0900.cnf")
    equation = dimacs.parse()
    is_file_valid = dimacs.validate()
    print(is_file_valid)

    ## test genetic algorithm
    crossover_rate = 0.9
    mutation_rate = 0.08
    population_size = 50
    generations = 1000
    elitism = True

    ## genetic
    genetic_algorithm = StandardGenetic([crossover_rate, mutation_rate], elitism,
                                        population_size, generations)
    start_time = timer()
    best_population = genetic_algorithm.run(equation, verbose=False)
    end_time = timer()
    print(best_population.best.valid)
    print(best_population.best.fitness)
    print("SGA time = " + str(end_time - start_time))

    ## adaptive
    adaptive_algorithm = AdaptiveGenetic(population_size, generations, 20, 0.75)
    start_time = timer()
    best_population = adaptive_algorithm.run(equation, verbose=False)
    end_time = timer()
    print(best_population.best.valid)
    print(best_population.best.fitness)
    print("IAGA time = " + str(end_time - start_time))
    print("")

    #dpll = DPLL()
    #solution = dpll.run(equation)

    #print(solution)
    #print(equation.validate(solution))

if __name__ == "__main__":
    main()
