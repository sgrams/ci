"""
2020/01/22
Stanislaw Grams <sjg@fmdx.pl>
09-3sat_project/src/algorithms.py
"""
import random
from copy import deepcopy
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
        solution = self.backtrack(equation.literals, [])
        if solution:
            solution += [x for x in range(1, equation.variables) \
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

    def run(self, equation, time_limit=200.0) -> Population:
        """ executes SGA against given equation """
        # pylint: disable-msg=too-many-locals

        ## create initial population
        population = Population(self._population_size, equation)
        population.initialize()

        ## save start time
        time_start = timer()
        ## iterate over generations but end if timelimit is hit
        for _ in range(self._generations):
            if timer() - time_start >= time_limit:
                break

            ## create new population
            new_population = Population(self._population_size, equation)

            ## fitness calculation
            fitnesses = [chromosome.fitness for chromosome in population]

            ## proper evolution
            if self._elitism is True:
                range_offset = 2
            else:
                range_offset = 0

            for _ in range(len(population) - range_offset):
                parent_a = population[self.selection(fitnesses)]
                parent_b = population[self.selection(fitnesses)]

                ## perform crossover and mutate the result
                if random.random() <= self._crossover_rate:
                    child = self.crossover(parent_a, parent_b)
                    if random.random() <= self._mutation_rate:
                        child = self.mutation(child)

                    ## add result child to population
                    new_population.push(child)

            ## elitism: keep the best individual from previous generation
            if self._elitism is True:
                previous_chromosomes = list(population.chromosomes[:])
                previous_chromosomes.sort(key=lambda x: x.fitness, reverse=True)
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
            print(population.best.fitness)
            population = deepcopy(new_population)

        ## return evolved population
        return deepcopy(population)
