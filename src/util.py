import math
import matplotlib.pyplot as plt
from datetime import datetime
import networkx as nx

def check_unvisited_node(unvisited):
    for u in unvisited:
        if u == 1:
            return True
    return False

def get_unvisited_node(unvisited):
    for index, node in enumerate(unvisited):
        if node == 1:
            return index
    return -1

def find_best_route(node_no, travel_route, min_distance):
    shortest_travel_route = travel_route[0]
    shortest_min_distance = min_distance.item(0)
    for start_node in range(0, node_no):
        if min_distance[start_node] < shortest_min_distance:
            shortest_min_distance = min_distance.item(start_node)
            shortest_travel_route = travel_route[start_node]

    print("min distance is: " + str(shortest_min_distance))
    print("travel route is: ")
    print(shortest_travel_route)

    return shortest_min_distance, shortest_travel_route


def in_travel_route(node, travel_route):
    for t in travel_route:
        if t == node:
            return True
    return False


def calc_distance(city1, city2):
    x1,y1 = city1
    x2,y2 = city2
    dist = math.sqrt(pow((x1-x2),2) + pow((y1-y2),2))
    return dist

def transform_inst_to_networkx_coord(inst):
    # Depart as 0
    coordinates = {0: (inst["depart"]["coordinates"]["x"], inst["depart"]["coordinates"]["y"])}

    # Customers as >=1
    for key in inst.keys():
        if key.find("customer_") != -1:
            customer_id = int(key[key.find("_") + 1:])
            position = (inst[key]["coordinates"]["x"], inst[key]["coordinates"]["y"])
            coordinates[customer_id] = position

    # coordinates = {<int_id>: (<float_x>, <float_y>)}
    return coordinates

def plot_graph(nodes, edges, coordinates): # nodes is list of nodes used as keys in coordinates, edges is list of tuples [(from, to)]
    # Create a directed graph
    # G = nx.DiGraph()
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Draw the graph
    nx.draw(G, pos=coordinates, with_labels=True, node_size=5, node_color='skyblue', font_size=6, font_weight='normal', arrowsize=6, width=0.2)
    plt.savefig('./visualizations/' + 'plot' + '_' +str(f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S %Z %z')}.png"))

def getPolarAngle(target, origin, coordinates):
    pt = coordinates[target]
    po = coordinates[origin] 
    x_diff = pt[0] - po[0]
    y_diff = pt[1] - po[1]
    return math.atan2(y_diff, x_diff)

def getCustomerDemand(customer_id, inst):
    customer_key = f"customer_{customer_id}"
    return inst[customer_key]["demand"]

def isFeasibleSolution(individual, inst):
    ok = True
    vehicle_capacity = inst["vehicle_capacity"]
    cluster_demand = dict()
    
    for i in range(len(individual)):
        customer_id = i + 1
        cust_demand = getCustomerDemand(customer_id, inst)
        cluster_id = individual[i]
        if cluster_id in cluster_demand:
            cluster_demand[cluster_id] = cluster_demand[cluster_id] + cust_demand
        else:
            cluster_demand[cluster_id] = cust_demand

    for _, total_demand in cluster_demand.items():
        if total_demand > vehicle_capacity:
            ok = False
            break

    return ok