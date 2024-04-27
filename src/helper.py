'''
Individual:
    index 0 represents customer 1
    index 1 represents customer 2
    ...

distance_matrix:
    [0][1] : means distance betwen depot (depart) and customer 1

'''

import random
from deap import tools
from deap import base, creator
from functools import partial
import math
from src import util
import numpy as np

def print_routes(routes):
    for i in range(len(routes)):
        print(f'Route {i}: {routes[i]}')

def tsp(cluster_points, distance_matrix):
    if len(cluster_points) == 0:
        return 0, []
    elif len(cluster_points) == 1:
        return distance_matrix[0][cluster_points[0]] * 2, [(0, cluster_points[0]), (cluster_points[0], 0)]
    
    total_cost = 0
    shortest_travel_route = []
    edges = []

    def getClosestToDepartNodeIndex():
        best_cluster_point_i = 0
        best_cluster_point = cluster_points[best_cluster_point_i]
        closestDistance = distance_matrix[0][best_cluster_point]
        for curr_cluster_point_i in range(len(cluster_points)):
            curr_cluster_point = cluster_points[curr_cluster_point_i]
            curr_distance_from_depart = distance_matrix[0][curr_cluster_point]
            if curr_distance_from_depart < closestDistance :
                closestDistance = curr_distance_from_depart
                best_cluster_point_i = curr_cluster_point_i
                best_cluster_point = cluster_points[best_cluster_point_i]
        return best_cluster_point_i

    curr_node_index = curr_target_index = start_node_index = getClosestToDepartNodeIndex() # math.floor(random.random() * len(cluster_points))
    unvisited = [1 for _ in range(len(cluster_points))]
    unvisited[start_node_index] = 0
    shortest_travel_route.append(start_node_index)

    #depot to first customer
    total_cost = distance_matrix[0][start_node_index]

    for _ in range(len(cluster_points) - 1) :
        curr_min_distance = math.inf

        for i in range(len(cluster_points)):
            a = cluster_points[curr_node_index]
            b = cluster_points[i]
            if a != b and unvisited[i] == 1 :
                curr_dist = distance_matrix[a][b]
                
                if curr_dist < curr_min_distance :
                    curr_target_index = i
                    curr_min_distance = curr_dist

        total_cost = total_cost + curr_min_distance
        shortest_travel_route.append(curr_target_index)
        unvisited[curr_target_index] = 0
        curr_node_index = curr_target_index
     
    # back to depot
    b = cluster_points[curr_target_index]
    total_cost = total_cost + distance_matrix[b][0]

    for i in range(len(shortest_travel_route) - 1):
        a = cluster_points[shortest_travel_route[i]]
        b = cluster_points[shortest_travel_route[i + 1]]
        edge = (a, b)
        edges.append(edge)

    edges.append((edges[-1][1], 0))
    edges.append((0, edges[0][0]))

    # #debug
    # nodes = cluster_points
    # visited = dict()
    # for node in nodes:
    #     visited[node] = False
    # for edge in edges:
    #     visited[edge[0]] = True
    #     visited[edge[1]] = True
    # ok = True
    # for a, b in visited.items():
    #     if b is False:
    #         ok = False
    # print(ok)
    # #debug

    return total_cost, edges

def evaluate_graph_cost(edges, distance_matrix):
    retVal = 0 

    for edge in edges:
        # TODO: avoid duplication of edge here
        a = edge[0]
        b = edge[1]
        retVal = retVal + distance_matrix[a][b]

    return retVal

# evaluation function
def evaluate(individual, inst=None):
    fitness = 0
    coordinates = util.transform_inst_to_networkx_coord(inst)
    clusters = dict()

    for i in range(len(individual)):
        cluster_id = individual[i]
        customer_id = i + 1
        try:
            clusters[cluster_id].append(customer_id)
        except KeyError:
            clusters[cluster_id] = [customer_id]

    for cluster_id, nodes in clusters.items():
        distance_matrix = inst["distance_matrix"]
        if len(nodes) > 0 :
            cost, edges = tsp(nodes, distance_matrix)
            fitness = fitness + cost
        # util.plot_graph(nodes, edges, coordinates) # debug

    return fitness

def algo(inst = None, toolbox = None):
    POP_SIZE = 30  # Population Size
    CXPB, MUTPB, NGEN, TERM = 0.2, 0.8, 1000, 1000

    # Initialize Population 
    pop = toolbox.population(n=POP_SIZE)

    # #debug
    # ok = True
    # for ind in pop:
    #     if util.isFeasibleSolution(ind, inst) is False:
    #         ok = False
    #         break
    # print("ok?: ", ok)
    # #debug

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop)) # Doubt: How's map different from dict. Reference of Map object in documentation not found
    # print(fitnesses) #debug
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
            # Select two parents via Binary Tournament each
            parent1, parent2 = toolbox.select(pop, 2)
            # Produce two oFspring from the parents
            child1, child2 = toolbox.clone(parent1), toolbox.clone(parent2)
            toolbox.mate(child1, child2); del child1.fitness.values; del child2.fitness.values
            # Evaluate 2tness and un2tness of oFspring
            child1.fitness.values = (toolbox.evaluate(child1),)
            child2.fitness.values = (toolbox.evaluate(child2),)

            if util.isFeasibleSolution(child1, inst):
                toolbox.replace(pop, child1)
            else:
                child1 = None

            if util.isFeasibleSolution(child2, inst):
                toolbox.replace(pop, child2)
            else:
                child2 = None


            # Choose favoured oFspring
            # for each offspring remove if not feasible
            # If entry criteria are satis2ed by chosen oFspring
                # Choose population member to be replaced (worst fitness)
                # OFspring enters population and the chosen member is removed

            # Update fittestInd
            for ind in pop:
                if ind.fitness.values[0] < fittestInd.fitness.values[0] :
                    last_improvement = GEN
                    fittestInd = toolbox.clone(ind)

            print(f"Generation {GEN}, Best Fitness : {fittestInd.fitness.values[0]}")
            # print(fittestInd)

    return fittestInd

def sweepGeneration(ind_size, inst):
    individual = [0 for _ in range(ind_size)]

    # [[<polar_angle>, <customer_id>]]
    root = []
    coordinates = util.transform_inst_to_networkx_coord(inst)
    for ind_i in range(ind_size):
        customer_id = ind_i + 1 # changing individual index to customer_id it signifies
        polar_angle =  util.getPolarAngle(customer_id, 0, coordinates)
        root.append([polar_angle, customer_id])
    
    # sort root by polar angle
    root.sort()
    
    # sweep
    curr_vehicle_id = 0
    cluster_demand = dict()
    vehicle_capacity = inst["vehicle_capacity"]
    offset = math.floor(random.random() * ind_size) # ind_size also represents number of customers

    for i in range(len(root)):
        root_i = (i + offset) % len(root)
        customer_id = root[root_i][1]
        curr_customer_demand = util.getCustomerDemand(customer_id, inst)
        ind_i = customer_id - 1

        if curr_vehicle_id in cluster_demand:
            if cluster_demand[curr_vehicle_id] + curr_customer_demand > vehicle_capacity:
                curr_vehicle_id = curr_vehicle_id + 1
                cluster_demand[curr_vehicle_id] = curr_customer_demand
            else:
                cluster_demand[curr_vehicle_id] = cluster_demand[curr_vehicle_id] + curr_customer_demand
        else:
            # assuming customer demand never exceeds vehicle capacity
            cluster_demand[curr_vehicle_id] = curr_customer_demand 

        individual[ind_i] = curr_vehicle_id

    return individual


def helper(inst = None):
    # Configure Creator
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Configure Toolbox 
    IND_SIZE = inst["max_vehicle_number"] - 1  # Individual Size
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.randint, 0, 5)
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE) #TODO: remove this line
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: sweepGeneration(IND_SIZE, inst) )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    # Register Operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", evaluate, inst=inst)

    # Register Replacement Scheme    
    def custom_steady_state(population, offspring):
        parents = toolbox.select(population, 3) # TODO: change to worst fitness
        for parent in parents:
            if offspring.fitness.values[0] < parent.fitness.values[0] :
                population.remove(parent)
                population.append(offspring)
                break

    toolbox.register("replace", custom_steady_state)

    fittestInd = algo(inst=inst, toolbox=toolbox)

    # debug
    fitness = 0
    coordinates = util.transform_inst_to_networkx_coord(inst)
    clusters = dict()

    for i in range(len(fittestInd)):
        cluster_id = fittestInd[i]
        customer_id = i + 1
        try:
            clusters[cluster_id].append(customer_id)
        except KeyError:
            clusters[cluster_id] = [customer_id]

    for cluster_id, nodes in clusters.items():
        distance_matrix = inst["distance_matrix"]
        if len(nodes) > 0 :
            cost, edges = tsp(nodes, distance_matrix)
            fitness = fitness + cost
        util.plot_graph(nodes, edges, coordinates) # debug

    print("Best Fitness: ", fitness)
    print("Best Fitness: ", toolbox.evaluate(fittestInd))
    print(fittestInd.fitness.values[0])
    print("No of Vehicles: ", max(fittestInd))
    # debug