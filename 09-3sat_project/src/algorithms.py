"""
2020/01/22
Stanislaw Grams <sjg@fmdx.pl>
09-3sat_project/src/algorithms.py
"""
import random
from copy import deepcopy
from basic_types import Chromosome
from basic_types import Population

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

    def mutation(self, chromosome: Chromosome) -> Chromosome:
        """ randomly mutates a given chromosome"""
        genes = chromosome.genes[:]
        avail_genes = set(range(len(chromosome)))
        indexes = []

        # mutate all genes
        if self._elitism is True:
            range_begin = 1
        else:
            range_begin = 0
        for _ in range(random.randint(range_begin, len(chromosome) - 1)):
            choice = random.choice(list(avail_genes))
            indexes.append(choice)
            avail_genes.remove(choice)

        ## flip genes for chosen indexes
        for index in indexes:
            genes[index] = int(not genes[index])

        return Chromosome(chromosome.equation, genes)

    def run(self, equation) -> Population:
        """ executes SGA against given equation """

        ## create initial population
        population = Population(self._population_size, equation)
        population.initialize()

        ## iterate over generations
        for _ in range(self._generations):
            ## create new population
            new_population = Population(self._population_size, equation)

            ## fitness calculation
            fitnesses = [chromosome.fitness for chromosome in population]

            ## elitism: keep the best individual from previous generation
            if self._elitism is True:
                previous_chromosomes = list(population.chromosomes[:])
                previous_chromosomes.sort(key=lambda x: x.fitness, reverse=True)
                new_population.push(previous_chromosomes[0])
                fitnesses = fitnesses[1:]


            ## proper evolution
            for _ in range(len(population)):
                parent_a = population[self.selection(fitnesses)]
                parent_b = population[self.selection(fitnesses)]

                ## perform crossover and mutate the result
                if random.random() <= self._crossover_rate:
                    child = self.crossover(parent_a, parent_b)
                    if random.random() <= self._mutation_rate:
                        child = self.mutation(child)

                    ## add result child to population
                    new_population.push(child)

                population_diff = len(population) - len(new_population)

            if population_diff > 0:
                ## select best genes
                previous_chromosomes = list(population.chromosomes[:])
                previous_chromosomes.sort(key=lambda x: x.fitness, reverse=True)

                population_supplements = previous_chromosomes[:population_diff]

                for supplement in population_supplements:
                    new_population.push(supplement)
            population = deepcopy(new_population)

        ## return evolved population
        return population
