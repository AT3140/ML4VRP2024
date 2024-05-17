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
from sklearn.cluster import KMeans

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
    POP_SIZE = 40  # Population Size
    CXPB, MUTPB, NGEN, TERM = 0.9, 0.1, 1000, 50

    # Initialize Population 
    pop = toolbox.population(n=POP_SIZE)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop)) # Doubt: How's map different from dict. Reference of Map object in documentation not found
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
            decision_variable = random.random()

            if decision_variable < CXPB :
                # Select two parents via Binary Tournament each
                parent1, parent2 = toolbox.select(pop, 2)
                # Produce two oFspring from the parents
                child1, child2 = toolbox.clone(parent1), toolbox.clone(parent2)
                child1, child2 = util.getEquivalentCompatibleParents(child1, child2)
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

            if decision_variable < MUTPB :
                mutation_targets = []
                for _ in range(POP_SIZE):
                    ind_decision_variable = random.random()
                    if ind_decision_variable < MUTPB :
                        candidates = toolbox.select(pop, 3)
                        target_candidate = candidates[0]
                        for candidate in candidates :
                            if candidate.fitness.values[0] < target_candidate.fitness.values[0] :
                                target_candidate = candidate
                        mutation_targets.append(target_candidate)
                
                for target_individual in mutation_targets:
                    original_individual = toolbox.clone(target_individual)
                    toolbox.mutate(target_individual)
                    if not util.isFeasibleSolution(target_individual, inst) :
                        target_individual = original_individual

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

            # print(f"Generation {GEN}, Best Fitness : {fittestInd.fitness.values[0]}")
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

def kmeansGeneration(num_clusters, ind_size, inst):
    cust_coordinates = util.transform_inst_to_networkx_coord(inst)
    coordinates = []
    for cust_key in range(1, inst['max_vehicle_number']):
        coordinates.append(cust_coordinates[cust_key])

    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(coordinates)
    cluster_centers = list(kmeans.cluster_centers_)
    individual = [-1 for _ in range(len(coordinates))]
    random.shuffle(cluster_centers)
    vehicle_counter = 0
    curr_load = 0
    curr_cluster_center_i = 0

    while min(individual) == -1 :
        closest_customer_i = -1
        closest_customer_id = closest_customer_i + 1 
        min_dist = math.inf
        for cust_i, vehicle_i in enumerate(individual):
            cust_id = cust_i + 1
            if individual[cust_i] == -1:
                dist_with_curr_cluster_center = util.calc_distance(cluster_centers[curr_cluster_center_i % len(cluster_centers)], cust_coordinates[closest_customer_id])
                if curr_load + util.getCustomerDemand(cust_id, inst) <= inst['vehicle_capacity'] and dist_with_curr_cluster_center < min_dist :
                    min_dist = dist_with_curr_cluster_center
                    closest_customer_i = cust_i
                    closest_customer_id = cust_id
        if closest_customer_i == -1 :           
            vehicle_counter = vehicle_counter + 1
            curr_load = 0
            curr_cluster_center_i = curr_cluster_center_i + 1
        else:
            individual[closest_customer_i] = vehicle_counter
            curr_load = curr_load + util.getCustomerDemand(closest_customer_id, inst)

    cluster_loads = [0 for _ in range(num_clusters)]
    for i, cluster_id in enumerate(individual):
        cust_id = i + 1
        cust_demand = util.getCustomerDemand(cust_id, inst)
        cluster_loads[cluster_id] = cluster_loads[cluster_id] + cust_demand

    return individual

def helperGeneration(ind_size, inst):
    individual = sweepGeneration(ind_size, inst)
    MLGENPB = 0.5
    decision_variable = random.random()
    if decision_variable < MLGENPB :
        num_routes = max(individual)
        temp_individual = kmeansGeneration(num_routes, ind_size, inst)
        if util.isFeasibleSolution(temp_individual, inst):
            individual = temp_individual
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
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: helperGeneration(IND_SIZE, inst) )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    # Register Operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", evaluate, inst=inst)

    # Register Mutation
    def custom_mutation(individual):
        no_of_vehicles = max(individual) + 1
        selected_vehicle = math.floor(random.random() * no_of_vehicles)
        ngbr_vehicle = (selected_vehicle + 1) %  no_of_vehicles
        find_ngbr = False
        for i in range(2 * len(individual)):
            i = i % len(individual)
            if find_ngbr:
                if individual[i] == ngbr_vehicle:
                    b = i
                    break
            else:
                if individual[i] == selected_vehicle:
                    a = i
                    find_ngbr = True
        
        individual[a], individual[b] = individual[b], individual[a]

    toolbox.register("mutate", custom_mutation)

    # Register Replacement Scheme    
    def custom_steady_state(population, offspring):
        parents = toolbox.select(population, 3) 
        target_parent = parents[0]

        for i in range(len(parents)):
            if parents[i].fitness.values[0] < target_parent.fitness.values[0] :
                target_parent = parents[i]

        population.remove(target_parent)
        population.append(offspring)

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

    route_counter = 0
    for cluster_id, nodes in clusters.items():
        distance_matrix = inst["distance_matrix"]
        if len(nodes) > 0 :
            cost, edges = tsp(nodes, distance_matrix)
            fitness = fitness + cost

            route_counter = route_counter + 1
            print(f"Route #{route_counter}:", end=" ")
            start_i = 0
            for i in range(len(edges)):
                if edges[i][0] == 0 :
                    start_i = i
                    break
            curr_i = start_i
            while edges[curr_i][1] != 0 :
                print(edges[curr_i][1], end=" ")
                curr_i = (curr_i + 1) % len(edges)
            print()

        util.plot_graph(nodes, edges, coordinates) 

    # print("Cost ", fitness)
    # print("No of Vehicles: ", max(fittestInd) + 1)