"""
2020/01/22
Stanislaw Grams <sjg@fmdx.pl>
09-3sat_project/src/algorithms.py
"""
import os
import random
from copy import copy
from timeit import default_timer as timer
from basic_types import Chromosome
from basic_types import Population

class DPLL():
    """ implements DPLL algorithm """
    def __init__(self):
        self._solved = False
        self._solution = None

    @property
    def solved(self):
        """ returns status of DPLL"""
        return self._solved

    @property
    def solution(self) -> list:
        """returns solution for given equation """
        return self._solution

    @staticmethod
    def bcp(literals: list, pure_unit: int):
        """ implements boolean constraint propagation """
        new = []
        for triplet in literals:
            if pure_unit in triplet:
                continue
            if -pure_unit in triplet:
                tmp_triplet = [x for x in triplet if x != -pure_unit]
                if len(tmp_triplet) == 0:
                    return -1
                new.append(tmp_triplet)
            else:
                new.append(triplet)
        return new

    @staticmethod
    def get_quantity_map(literals: list):
        """ returns an array of variables quantity """
        qty = {}
        for clause in literals:
            for literal in clause:
                if literal in qty:
                    qty[literal] += 1
                else:
                    qty[literal] = 1
        return qty

    def find_pure_literal(self, literals: list):
        """ performs pure literal assignment """
        qty_map = self.get_quantity_map(literals)
        assignment = []
        pures = []

        for literal, _ in qty_map.items():
            if -literal not in qty_map:
                pures.append(literal)

        for pure_unit in pures:
            literals = self.bcp(literals, pure_unit)
        assignment += pures
        return literals, assignment

    def propagate_unit(self, literals: list):
        """ performs unit propagation """
        assignment = []
        unit_clauses = [tmp for tmp in literals if len(tmp) == 1]
        while len(unit_clauses) > 0:
            unit = unit_clauses[0]
            literals = self.bcp(literals, unit[0])
            assignment += [unit[0]]
            if literals == -1:
                return -1, []
            if not literals:
                return literals, assignment
            unit_clauses = [tmp for tmp in literals if len(tmp) == 1]
        return literals, assignment

    def select_variable(self, literals: list):
        """ randomly select variable from literals """
        qty_map = self.get_quantity_map(literals)
        return random.choice(list(qty_map.keys()))

    def backtrack(self, literals: list, assignment: list) -> list:
        """ recurses over list of literals """
        literals, pure_assignment = self.find_pure_literal(literals)
        literals, unit_assignment = self.propagate_unit(literals)
        assignment = assignment + pure_assignment + unit_assignment

        if literals == -1:
            return []
        if not literals:
            return assignment

        variable = self.select_variable(literals)
        solution = self.backtrack(self.bcp(literals, variable), assignment + [variable])
        if not solution:
            solution = self.backtrack(self.bcp(literals, -variable), assignment + [-variable])

        return solution

    def run(self, equation) -> list:
        """ executes DPLL against given equation """
        # pylint: disable-msg=unnecessary-lambda
        solution = self.backtrack(equation.literals, [])
        if solution:
            solution += [x for x in range(1, equation.variables + 1) \
                         if x not in solution and -x not in solution]
            solution.sort(key=lambda x: abs(x))

        output = list()
        for val in solution:
            if val > 0:
                output.append(1)
            if val < 0:
                output.append(0)
        return output

class StandardGenetic():
    """ implements standard genetic algorithm """

    def __init__(self, rates: [float, float], elitism: bool,
                 population_size: int, generations: int):
        random.seed(a=os.urandom)
        self._crossover_rate = rates[0]
        self._mutation_rate = rates[1]
        self._elitism = elitism
        self._population_size = population_size
        self._generations = generations

    @staticmethod
    def selection(fitnesses: list) -> int:
        """ selects a parent for crossover """
        fitness = random.random() * sum(fitnesses)
        index = 0

        while fitness > 0:
            fitness -= fitnesses[index]
            index += 1

        if index < 0:
            index = 0
        elif index > 0:
            index -= 1

        return index

    @staticmethod
    def crossover(parent_a: Chromosome, parent_b: Chromosome) -> Chromosome:
        """ randomly returns a child from crossover of parent_a and parent_b """

        if len(parent_a) == 2:
            index = 1
        else:
            index = random.randint(0, len(parent_a) - 2)

        genes = parent_a[:index] + parent_b[index:]
        return Chromosome(parent_a.equation, genes)

    @staticmethod
    def mutation(chromosome: Chromosome) -> Chromosome:
        """ randomly mutates a given chromosome"""
        genes = chromosome.genes[:]
        avail_genes = set(range(len(chromosome)))
        indexes = []

        # mutate all genes
        range_begin = 0
        for _ in range(random.randint(range_begin, len(chromosome) - 1)):
            choice = random.choice(list(avail_genes))
            indexes.append(choice)
            avail_genes.remove(choice)

        ## flip genes for chosen indexes
        for index in indexes:
            genes[index] = int(not genes[index])

        return Chromosome(chromosome.equation, genes)

    def pick_parents(self, population, fitnesses):
        """ randomly picks parents A and B from given population """
        parent_a = population[self.selection(fitnesses)]
        parent_b = population[self.selection(fitnesses)]

        return parent_a, parent_b

    def run(self, equation, time_limit=200.0, verbose=False, stop_if_satisfied=True) -> Population:
        """ executes SGA against given equation """
        # pylint: disable-msg=too-many-locals
        # pylint: disable-msg=too-many-branches

        ## create initial population
        population = Population(self._population_size, equation)
        population.initialize() ## initial population

        ## save start time
        time_start = timer()
        ## iterate over generations but end if timelimit is hit
        for generation in range(self._generations):
            ## introduce time limit
            if timer() - time_start >= time_limit:
                break

            ## evaluation: check for fitness (1.0)
            if stop_if_satisfied is True:
                previous_chromosomes = list(population.chromosomes[:])
                previous_chromosomes.sort(key=lambda x: x.fitness, reverse=True)
                if previous_chromosomes[0].fitness == 1.00:
                    break

            ## create new population
            new_population = Population(self._population_size, equation)

            ## proper evolution
            if self._elitism is True:
                range_offset = 2
            else:
                range_offset = 0

            ## fitness calculation
            fitnesses = [chromosome.fitness for chromosome in population]

            ## selection operation
            for _ in range(len(population) - range_offset):
                parent_a, parent_b = self.pick_parents(population, fitnesses)

                if random.random() <= self._crossover_rate:
                    ## crossover
                    child = self.crossover(parent_a, parent_b)
                    if random.random() <= self._mutation_rate:
                        ## mutate
                        child = self.mutation(child)

                    ## add result child to population
                    new_population.push(child)

            ## elitism: keep the best individual from previous generation
            if self._elitism is True:
                new_population.push(previous_chromosomes[0])
                new_population.push(previous_chromosomes[1])

            population_diff = len(population) - len(new_population)
            if population_diff > 0:
                ## select best genes
                previous_chromosomes = list(population.chromosomes[:])
                previous_chromosomes.sort(key=lambda x: x.fitness, reverse=True)

                population_supplements = previous_chromosomes[:population_diff]

                for supplement in population_supplements:
                    new_population.push(supplement)

            ## new_population is the new population
            population = new_population

            ## print result for generation if verbose issued
            if verbose is True:
                if generation % 10 == 0:
                    print("generation %i: fitness=%f" % (generation, population.best.fitness))

        ## return evolved population
        return copy(population)

class AdaptiveGenetic(StandardGenetic):
    """ implements improved adaptive genetic algorithm """
    def __init__(self, population_size: int, generations: int,
                 restart_rate: int, fzero=1.0):
        StandardGenetic.__init__(self, [0.00, 0.00], False, population_size, \
                                 generations)
        self._restart_rate = restart_rate
        self._population_size = population_size
        self._generations = generations
        self._fzero = fzero
        self._infsim = 0.00000000000000001

    @staticmethod
    def calc_avg_fitness(population: Population) -> float:
        """ calculates avg fitness """
        fitness_sum = 0
        for chromosome in population.chromosomes:
            fitness_sum += chromosome.fitness
        return fitness_sum / len(population.chromosomes)

    @staticmethod
    def calc_min_fitness(population: Population) -> float:
        """ finds min fitness """
        fitness_min = 1.0
        for chromosome in population.chromosomes:
            if chromosome.fitness < fitness_min:
                fitness_min = chromosome.fitness
        return fitness_min

    @staticmethod
    def calc_max_fitness(population: Population) -> float:
        """ finds max fitness """
        fitness_max = 0.0
        for chromosome in population.chromosomes:
            if chromosome.fitness >= fitness_max:
                fitness_max = chromosome.fitness
        return fitness_max

    def calc_m1(self, population: Population) -> int:
        """ calculates M1 """
        counter = 0
        avg_fitness = self.calc_avg_fitness(population)
        for chromosome in population.chromosomes:
            if chromosome.fitness > avg_fitness:
                counter += 1
        return counter

    def calc_m2(self, population: Population) -> int:
        """ calculates M2 """
        return len(population.chromosomes) - self.calc_m1(population)

    def find_best_population(self, population: Population) -> Population:
        """ finds best population (1/2 best * 2) """
        best_chromosomes = list(population.chromosomes[:])
        best_chromosomes.sort(key=lambda x: x.fitness, reverse=True)

        best_chromosomes = best_chromosomes[:int(self._population_size / 2)] + \
                           best_chromosomes[:int(self._population_size / 2)]

        best_population = Population(population.equation, self._population_size)
        best_population.chromosomes = best_chromosomes

        return best_population

    @staticmethod
    def fill_population(population: Population, source: Population) -> Population:
        """ fills population to initialized population sized """
        population_diff = len(source) - len(population)
        if population_diff > 0:
            previous_chromosomes = list(source.chromosomes[:])
            previous_chromosomes.sort(key=lambda x: x.fitness, reverse=True)
            population_supplements = previous_chromosomes[:population_diff]
            for supplement in population_supplements:
                population.push(supplement)
        return population

    def crossover_and_mutation(self, population: Population, generation: int):
        # pylint: disable-msg=too-many-locals
        # pylint: disable-msg=too-many-branches
        """ crossover and mutation """
        new_population = Population(population.equation, self._population_size)
        best_population = self.find_best_population(population)

        ## fitness calculation
        fitnesses = [chromosome.fitness for chromosome in best_population]

        for _ in range(len(population)):
            ## crossover and mutation operation (based on g ≤ 0.75 * G)
            avg_fit = self.calc_avg_fitness(population)
            min_fit = self.calc_min_fitness(population)
            max_fit = self.calc_max_fitness(population)
            var_m1 = self.calc_m1(population)
            var_m2 = self.calc_m2(population)

            ## torunament
            parent_a, parent_b = self.pick_parents(best_population, fitnesses)

            if parent_a.fitness > parent_b.fitness:
                fprim = parent_a.fitness
            else:
                fprim = parent_b.fitness

            ## adaptively calculate crossover_rate
            if generation <= 0.75 * self._generations:
                if ((max_fit - avg_fit) / (avg_fit - min_fit + self._infsim) < 1.0) \
                    and (var_m1 > var_m2):
                    self._crossover_rate = \
                    0.8 * ((max_fit - avg_fit) / (avg_fit - min_fit + self._infsim))
                else:
                    self._crossover_rate = 0.9 - \
                    ((0.3 * (fprim - min_fit)) / (max_fit - min_fit + self._infsim))
            else:
                if ((max_fit - avg_fit) / (avg_fit - min_fit + self._infsim) < 1.0) \
                    and (var_m1 > var_m2):
                    self._crossover_rate = \
                    0.8 * (self._fzero - max_fit) / (self._fzero - avg_fit + self._infsim)
                else:
                    self._crossover_rate = 0.9 - \
                    (0.3 * (self._fzero - max_fit) / (self._fzero - fprim + self._infsim))

            ## perform crossover at calculated rate
            if random.random() <= self._crossover_rate:
                child = self.crossover(parent_a, parent_b)

                ## mutate but decision will be taken later
                mutated_child = self.mutation(child)
                feps = mutated_child.fitness

                ## adaptively calculate mutation rate
                if generation <= 0.75 * self._generations:
                    if ((max_fit - avg_fit) / (avg_fit - min_fit + self._infsim) < 1.0) \
                        and (var_m1 > var_m2):
                        self._mutation_rate = \
                        0.1 * (max_fit - avg_fit) / (avg_fit - min_fit + self._infsim)
                    else:
                        self._mutation_rate = 0.1 - \
                        (0.09 * (feps - min_fit) / (max_fit - min_fit + self._infsim))
                else:
                    if ((max_fit - avg_fit) / (avg_fit - min_fit + self._infsim) < 1.0) \
                        and (var_m1 > var_m2):
                        self._mutation_rate = \
                        0.1 * (self._fzero - max_fit) / (self._fzero - avg_fit + self._infsim)
                    else:
                        self._mutation_rate = 0.1 - \
                        (0.09 * (self._fzero - max_fit) / (self._fzero - feps + self._infsim))

                ## if rate lower than expected then restore unmutated child
                if random.random() <= self._mutation_rate:
                    new_population.push(mutated_child)
                else:
                    new_population.push(child)

            ## return filled population
            new_population = self.fill_population(new_population, population)
            return new_population

    def greedy(self, population, old_population):
        """ performs greedy operation """
        ## randomly select chromosome
        rand_chr_index = random.randint(0, self._population_size - 1)
        chromosome = population.chromosomes[rand_chr_index]

        ## randomly select variable out of genes
        rand_gen_index = random.randint(0, len(chromosome.genes) - 1)

        ## find fitness if variable flipped
        new_chromosome = Chromosome(chromosome.equation, chromosome.genes)
        new_chromosome.genes[rand_gen_index] = -new_chromosome.genes[rand_gen_index]
        max_fit_after_turning = new_chromosome.fitness

        ## perform flip on population if applicable
        if population.best.fitness >= old_population.best.fitness:
            if max_fit_after_turning >= population.best.fitness:
                population.chromosomes[rand_chr_index] = new_chromosome
        else:
            sorted_chromosomes = list(population.chromosomes)
            sorted_chromosomes.sort(key=lambda x: x.fitness, reverse=True)

            ## replace the best chromosome of current generation with
            ## the best chromosome of previous generation
            sorted_chromosomes[0].genes = old_population.best.genes
            population.chromosomes = sorted_chromosomes

        return population

    # pylint: disable-msg=arguments-differ
    def run(self, equation, time_limit=200.0, verbose=False) -> Population:
        """ executes I_AGA against given equation """
        ## save start time
        time_start = timer()

        ## iterate over generations but end if timelimit is hit
        for restart in range(self._restart_rate):
            ## create initial population
            population = Population(self._population_size, equation)
            population.initialize() ## initial population

            for generation in range(self._generations):
                ## introduce time limit (explicit end condition)
                if timer() - time_start >= time_limit:
                    break

                ## save old generation
                old_population = population

                ## evaluation: check for fitness (1.0)
                previous_chromosomes = list(population.chromosomes[:])
                previous_chromosomes.sort(key=lambda x: x.fitness, reverse=True)
                if previous_chromosomes[0].fitness == 1.00:
                    return population ## 5: while (terminate condition)

                ## crossover and mutation
                population = self.crossover_and_mutation(population, generation)

                ## greedy
                if population.best.fitness == 1.0:
                    ## terminate if max fitness already reached
                    return population
                population = self.greedy(population, old_population)

                ## print result for generation if verbose issued
                if verbose is True:
                    print("generation %i: fitness=%f: restart %i" %
                          (generation, population.best.fitness, restart))

        ## return evolved population
        return population
