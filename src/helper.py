import random
from deap import tools
from deap import base, creator
from functools import partial
import math

def print_routes(routes):
    for i in range(len(routes)):
        print(f'Route {i}: {routes[i]}')

def get_updated_route_cost(route, new_city_i, inst=None):
    retVal = 0
    distance_matrix = inst['distance_matrix']
    prev = 0
    for city_i in route:
        retVal = retVal + distance_matrix[prev][city_i]
        prev = city_i
    retVal = retVal + distance_matrix[prev][0]
    return retVal

def get_updated_route_payload(route, new_city, inst=None):
    route.append(new_city)
    return get_route_payload(route, inst=inst)

def get_route_payload(route, inst=None):
    retVal = 0
    for city_i in route:
        customer_key = f"customer_{city_i}"
        retVal = retVal + inst[customer_key]['demand']
    return retVal

def evaluate_cost(routes, inst):
    retVal = 0
    distance_matrix = inst['distance_matrix']

    for route in routes:
        if len(route) > 0 : 
            # calculating current route cost
            curr_route_cost = 0
            prev = 0
            for i in range(len(route)):
                curr_city = route[i]
                curr_route_cost = curr_route_cost + distance_matrix[prev][curr_city]
                prev = curr_city
            curr_route_cost = curr_route_cost + distance_matrix[prev][0]
            # calculating current route payload
            curr_route_payload = get_route_payload(route, inst)
            # vehicle capacity check
            if curr_route_payload < inst['vehicle_capacity'] :
                retVal = retVal + curr_route_cost                
            else:
                retVal = math.inf
                break
        
    return retVal

def decode(individual, inst):
    routes = [[]]
    order_master = []
    decisions = individual[::2]
    vehicle_capacity = inst['vehicle_capacity']

    # Prepare order master
    for i, float_value in enumerate(individual[1::2]):
        order_master.append([float_value, i + 1])
    order_master.sort()
    
    # Prepare routes based on the order master
    for pair in order_master:
        city_index = pair[1]
        city_decision_index = city_index - 1
        if decisions[city_decision_index] < 0.5 :
            # insert in curr route
            new_curr_route_payload = get_updated_route_payload(routes[-1][:], city_index, inst=inst)
            if new_curr_route_payload < vehicle_capacity :
                routes[-1].append(city_index)
            else:
                routes.append([city_index])
        else: 
            # insert greedily in one of the previous routes
            min_dist = math.inf
            target_route = None

            for i in range(len(routes) - 1):
                new_target_route_cost = get_updated_route_cost(routes[i][:], city_index, inst)
                new_target_route_payload = get_updated_route_payload(routes[i][:], city_index, inst)
                if new_target_route_cost < min_dist and new_target_route_payload < vehicle_capacity: 
                    min_dist = new_target_route_cost
                    target_route = routes[i]

            if target_route is not None:
                target_route.append(city_index)
            else :
                routes.append([city_index])

    return routes

def decode_evaluate(individual, inst=None):
    routes = decode(individual, inst)
    return evaluate_cost(routes, inst)

def algo(inst = None, toolbox = None):
    POP_SIZE = 20  # Population Size
    CXPB, MUTPB, NGEN, TERM = 0.2, 0.8, 1000, 1000

    # Initialize Population 
    pop = toolbox.population(n=POP_SIZE)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop)) # Doubt: How's map different from dict. Reference of Map object in documentation not found
    print(fitnesses)
    for ind, fit in zip(pop, fitnesses): # here ind is reference to the creator.Individual in the pop
        ind.fitness.values = (fit,) # As per creator.FitnessMin definition in helper

    # Store the fittest initial individual
    fittestInd = toolbox.clone(pop[0])
    for ind in pop:
        if ind.fitness.values[0] < fittestInd.fitness.values[0] :
            fittestInd = toolbox.clone(ind)

    # last improvement
    last_improvement = 0

    for GEN in range(NGEN):
        # termination on stagnation
        if GEN - last_improvement < TERM:    
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop)) # returns references
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring)) # returns value (both clone() and list())

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]): # child1, child 2 are references
                if random.random() < CXPB:
                    toolbox.mate(child1, child2) # modifies in place
                    del child1.fitness.values # also invalidates it
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant) # modifies in place
                    del mutant.fitness.values

            # Convert negative to positive
            for ind in offspring:
                if ind.fitness.valid == False:
                    for i in range(len(ind)):
                        if ind[i] < 0 :
                            ind[i] = abs(ind[i])

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]        
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit,) # also sets .fitness.valid = True

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Update fittestInd
            for ind in pop:
                if ind.fitness.values[0] < fittestInd.fitness.values[0] :
                    last_improvement = GEN
                    fittestInd = toolbox.clone(ind)

            print(f"Generation {GEN}, Best Fitness : {fittestInd.fitness.values[0]}")
            # print(fittestInd)

    return fittestInd

def helper(inst = None):
    # Configure Creator
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Configure Toolbox 
    IND_SIZE = 2 * (len(inst['distance_matrix']) - 1)  # Individual Size
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                    toolbox.attribute, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # evaluation function
    def evaluate(individual, inst=None):
        return decode_evaluate(individual, inst)

    # Register Operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, inst=inst)

    algo(inst=inst, toolbox=toolbox)