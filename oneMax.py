from deap import base, creator, tools, algorithms
import numpy as np

import random
import matplotlib.pyplot as plt


def oneMaxFitness(individual):
    """Takes an individual as an argument and returns its fitness
        (sum of its bits in this case) [1,1,1,0] ==> (3,) 
        PS : Fitness values in DEAP are tuples"""
    return sum(individual),

ONE_MAX_LENGTH = 100 #individual bit length

#Genetic parameters
POPULATION_SIZE = 200
P_CROOSOVER = 0.9
P_MUTATION = 0.1
GENERATIONS = 50

def prepare_genetic_alg():

    global toolbox 
    toolbox = base.Toolbox()
    toolbox.register("zeroOrOne", random.randint, 0, 1)
    #Registering the fitness strategy, maximizing the equation in our case
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    #Declaring the Individual class
    creator.create("Individual", list, fitness=creator.FitnessMax)
    #create an instance of the Individual class composed of 100 integers (0 or 1)
    toolbox.register("individualCreator",tools.initRepeat, creator.Individual,toolbox.zeroOrOne, ONE_MAX_LENGTH)
    #create a population of individuals
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
    #Definition of evaluate opreator for the fitness function
    toolbox.register("evaluate", oneMaxFitness)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1/ONE_MAX_LENGTH)

def generate_solution_man():
    #creating the initial population
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    generationCounter = 0 #variable to count generations
    fitnessValues = list(map(toolbox.evaluate, population))
    #Combine the fitness values with individuals
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue
    #Extract first value out of each fitness for statistics
    fitnessValues = [individual.fitness.values[0] for individual in population]
    #Declaring statistics variables
    maxFitnessValues = []
    meanFitnessValues = []
    while max(fitnessValues) < ONE_MAX_LENGTH and generationCounter < GENERATIONS:
        generationCounter += 1
        #Applying the selection operation
        offspring = toolbox.select(population, len(population))
        #offspring are cloned to apply the corssover operation without affecting the original pop
        offspring = list(map(toolbox.clone, offspring))
        #Applying the crossover operation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROOSOVER:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        #Applying the crossover operation
        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        #Update fitness values for fresh individuals
        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
        for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            individual.fitness.values = fitnessValue
        #Replace old population with new one
        population[:] = offspring
        #Statistics gathering and plotting
        fitnessValues = [ind.fitness.values[0] for ind in population]
        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)
        print("- Generation {}: Max Fitness = {}, Avg Fitness = {}".format(generationCounter, maxFitness, meanFitness))
        best_index = fitnessValues.index(max(fitnessValues))
        print("Best Individual = ", *population[best_index], "\n")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()

def generate_sol_auto():
    """Same previous process using DEAP builtin functions"""
    population = toolbox.populationCreator(n=POPULATION_SIZE) #register population
    stats = tools.Statistics(lambda ind: ind.fitness.values) #register statistics
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    _, logbook = algorithms.eaSimple(population, toolbox, P_CROOSOVER, P_MUTATION, GENERATIONS, stats, verbose=True) #start algorithm
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()

if __name__ == "__main__":
    prepare_genetic_alg()
    # generate_solution_man()
    generate_sol_auto()