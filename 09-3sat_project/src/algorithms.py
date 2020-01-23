"""
2020/01/22
Stanislaw Grams <sjg@fmdx.pl>
09-3sat_project/src/algorithms.py
"""
import random
from basic_types import Chromosome
from basic_types import Population

class Genetic():
    """ implements standard genetic algorithm """

    def __init__(self, crossover_rate: float, mutation_rate: float):
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate

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
        """ randombly returns a child from crossover of parent_a and parent_b """

        if len(parent_a) == 2:
            index = 1
        else:
            index = random.randint(0, len(parent_a) - 2)

        genes = parent_a[0:index] + parent_b[index:]
        return Chromosome(parent_a.equation, genes)

    @staticmethod
    def mutation(chromosome: Chromosome) -> Chromosome:
        """ randomly mutates a given chromosome"""
        genes = chromosome.genes[:]
        avail_genes = list(range(len(chromosome)))
        indexes = []

        # mutate all genes
        for _ in range(random.randint(1, len(chromosome - 1))):
            choice = random.choice(avail_genes)
            indexes.append(choice)
            avail_genes.remove(choice)
        for index in indexes:
            genes[index] = int(not genes[index])

        return Chromosome(chromosome.equation, genes)

    def evolution(self, population: Population) -> Population:
        """ a single evolution of given population """

        ## create object for evolved population
        evolved_population = Population(population.size, population.equation)

        ## fitness calculation
        fitnesses = [chromosome.fitness for chromosome in population]

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
            evolved_population.push(child)

        ## add the best gene from previous population to evolved one
        population_diff = len(population) - len(evolved_population)

        if population_diff > 0:
            ## select best gene
            previous_chromosomes = list(population.chromosomes[:])
            previous_chromosomes.sort(key=lambda x: x.fitness, reverse=True)

            population_supplements = previous_chromosomes[:population_diff]

            for supplement in population_supplements:
                evolved_population.push(supplement)

        ## return evolved population
        return evolved_population
