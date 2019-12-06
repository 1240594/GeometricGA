import random
import time
from deap import base
from deap import creator
from deap import tools
import numpy as np
from secrets import randbelow

start_time = time.time() # Calculates runtime
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", randbelow, (10))
toolbox.register("individual", tools.initRepeat, creator.Individual, 
                 toolbox.attr_int, 20) 

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Archetype
Arch = np.array([0,0,0,0,1,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

def evalComp(individual):
    compare = ((np.absolute(Arch[3] - individual[3])+np.absolute(Arch[4] - individual[4])+np.absolute(Arch[5] - individual[5])+np.absolute(Arch[6] - individual[6])+np.absolute(Arch[7] - individual[7])+np.absolute(Arch[8] - individual[8])+np.absolute(Arch[9] - individual[9])+np.absolute(Arch[10] - individual[10])+np.absolute(Arch[11] - individual[11])+np.absolute(Arch[12] - individual[12])+np.absolute(Arch[13] - individual[13])+np.absolute(Arch[14] - individual[14])+np.absolute(Arch[15] - individual[15])+np.absolute(Arch[16] - individual[16])+np.absolute(Arch[17] - individual[17])+np.absolute(Arch[18] - individual[18])+np.absolute(Arch[19] - individual[19]))/17)
    return compare,

toolbox.register("evaluate", evalComp)
toolbox.register("mate", tools.cxUniform)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=4)

def main():
    random.seed(43)
    pop = toolbox.population(n=100)

    print("Start of evolution")

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print(" Evaluated %i individuals" % len(pop))

    CXPB, MUTPB = 0.5, 0.3
    fits = [ind.fitness.values[0] for ind in pop]
    g = 0
   
    while min(fits) > 0 and g < 50:
        g = g + 1

        print("-- Generation %i --" % g)
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2, 0.3)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print(" Evaluated %i individuals" % len(invalid_ind))

        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print(" Min %s" % min(fits))
        print(" Max %s" % max(fits))
        print(" Avg %s" % mean)
        print(" Std %s" % std)
        print("Archetype",  Arch) # print Archetype for current generation
        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values)) # print best individual of generation
        print("\n")
        
    print("-- End of (successful) evolution --")
    print("\n")
    print("\n")
    print("Archetype",  Arch)
    print("\n")
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()


print("--- %s seconds ---" % (np.around((time.time() - start_time),4)))