"""
2020/01/23
Stanislaw Grams <sjg@fmdx.pl>
09-3sat_project/src/main.py
"""
import sys
import getopt
from timeit import default_timer as timer
from datetime import datetime
import pandas as pd
from dimacs import Dimacs
from algorithms import StandardGenetic
from algorithms import AdaptiveGenetic
from algorithms import DPLL

def usage():
    """ prints program usage """
    print("Usage: python main.py -f /path/to/DIMACS_file.cnf " +
          "-o /path/to/output_dir/ -i iterations")

# pylint: disable-msg=too-many-locals
# pylint: disable-msg=too-many-statements
# pylint: disable-msg=too-many-branches
def main():
    """ main function """
    if len(sys.argv) == 1:
        usage()
        sys.exit(0)
    try:
        opts, _ = getopt.getopt(sys.argv[1:],
                                'f:o:i:h', ['file=', 'output=', 'iterations=', 'help'])
    except getopt.GetoptError:
        usage()
        sys.exit(1)

    ## iterations
    iterations = 50

    ## output_dir (standard)
    output_dir = "."

    ## parse input
    for opt, arg in opts:
        if opt in('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-f', '--file'):
            filepath = arg
        elif opt in ('-o', '--output'):
            output_dir = arg
        elif opt in ('-i', '--iterations'):
            iterations = int(arg)
        else:
            usage()
            sys.exit(1)

    ## create parser object
    try:
        dimacs = Dimacs(filepath=str(filepath))
        equation = dimacs.parse()
    except UnboundLocalError:
        print("Unable to parse given file!")
        sys.exit(1)

    print("isFileValid == " + str(dimacs.validate()))

    ####################
    ## general settings
    only_true = False
    cur_datetime = datetime.now().strftime("%Y%m%d%H%M%S")

    ## DPLL algorithm
    ## tests
    dpll_results = list()
    dpll_it = 0
    dpll = DPLL()
    print("testing DPLL...")
    while dpll_it < iterations:
        start_time = timer()
        solution = dpll.run(equation)
        time = timer() - start_time
        if only_true is True:
            if equation.validate(solution)[1] is True:
                dpll_it += 1
                dpll_results.append(time)
        else:
            dpll_it += 1
            dpll_results.append([time, equation.validate(solution)[1] is True])
    ## save results
    output_filepath = output_dir + "/DPLL_M" + str(equation.clauses) + "_L" + \
                      str(equation.variables) + "-" + cur_datetime + "-results.csv"
    try:
        dataframe = pd.DataFrame(data=dpll_results)
        dataframe.to_csv(output_filepath, sep=',', index=False, header=None)
    except UnboundLocalError:
        print("Unable to save data!")

    #####################
    ## genetic algorithms statically set variables
    population_size = 50
    generations = 1000

    ## SGA - standard genetic algorithm
    ## statically set rates
    crossover_rate = 0.9
    mutation_rate = 0.08
    elitism = True

    ## tests
    sga_results = list()
    sga_it = 0
    sga = StandardGenetic([crossover_rate, mutation_rate], elitism,
                          population_size, generations)
    print("testing SGA...")
    while sga_it < iterations:
        start_time = timer()
        bestpop = sga.run(equation, verbose=False)
        time = timer() - start_time
        if only_true is True:
            if bestpop.best.valid is True:
                sga_it += 1
                sga_results.append(time)
        else:
            sga_it += 1
            sga_results.append([time, bestpop.best.fitness])
    ## save results
    output_filepath = output_dir + "/SGA_M" + str(equation.clauses) + "_L" + \
                      str(equation.variables) + "-" + cur_datetime + "-results.csv"
    try:
        dataframe = pd.DataFrame(data=sga_results)
        dataframe.to_csv(output_filepath, sep=',', index=False, header=None)
    except UnboundLocalError:
        print("Unable to save data!")

    ## IAGA - improved adaptive genetic algorithm
    ## statically set rates
    restart_ratio = 20
    fzero = 0.75

    ## tests
    iaga_results = list()
    iaga_it = 0
    iaga = AdaptiveGenetic(population_size, generations, restart_ratio, fzero)
    print("testing IAGA...")
    while iaga_it < iterations:
        start_time = timer()
        bestpop = iaga.run(equation, verbose=False)
        time = timer() - start_time
        if only_true is True:
            if bestpop.best.valid is True:
                iaga_it += 1
                iaga_results.append(time)
        else:
            iaga_it += 1
            iaga_results.append([time, bestpop.best.fitness])
    ## save results
    output_filepath = output_dir + "/IAGA_M" + str(equation.clauses) + "_L" + \
                      str(equation.variables) + "-" + cur_datetime + "-results.csv"
    try:
        dataframe = pd.DataFrame(data=iaga_results)
        dataframe.to_csv(output_filepath, sep=',', index=False, header=None)
    except UnboundLocalError:
        print("Unable to save data!")

if __name__ == "__main__":
    main()
