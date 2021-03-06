import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gzip
import json
import time
from datetime import datetime
import networkx as nx
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import Counter
import copy
###################################################
#						  #
#		  Functions Ex1			  #
#						  #
###################################################

def get_user(user, dict_users):
    """
	Returns the class User from ID and update the dictionary

            Parameters:
                    user (int): user ID
                    dict_users (dict): as a key we must have the user ID and as a value we must have the class User 

            Returns:
                    user (User): class User
                    dict_users (dict) return the dict_users updated if the user not exists

	"""
	# create user if it wasn't created previously
    if user not in dict_users:
        user_obj = User(user)
        dict_users[user] = user_obj
    return dict_users[user], dict_users


def create_graph(df, type_node, weight, graph):
    """
    Returns the updated graph, the first call of this function, the graph must be empty
    
            Parameters:
                    df (dataframe): the dataset
    				type_node(string): the type of dataset (c2a, c2q, a2q)
    				weight(int): the weight to assign for each edge
    				graph(dict): the graph obtained until now, so the first one must be empty
    
            Returns:
                    graph(dict): return the updated graph
    
    """
    # for each row in the dataset
    for index, row in df.iterrows():
		# we only take the first 6 characters, so year and month for example 201808 -> August 2008
        year_month = int(str(row[2])[:-2])
        source_ = int(row[1])
        target_ = int(row[0])
        source, graph = get_user(source_, graph)
        target, graph = get_user(target_, graph)
		# create Relation(edge) between source(v) and target(u)
        rel = Relation(type_node, year_month, source, target, weight)
		# add relation in user
        graph[target_].add_in_relation(rel)   # add to u the incoming edge from v
        graph[source_].add_out_relation(rel) #  # add to v the oucoming edge to v   
    return graph

###################################################
#						  #
#		  Functions Ex2			  #
#						  #
###################################################

# Ex 2.1
def get_features(name_graph, graph):
    """
    plot all information below
    	- Whether the graph is directed or not
    	- Number of users
    	- Number of answers/comments
    	- Average number of links per user
    	- Density degree of the graph
    	- Whether the graph is sparse or dense
    
            Parameters:
                    name_graph (string): the name of the graph for which we want to obtain this information
    """
    type_graph = "directed"
    users = set()
    number_rel = 0 
    for user in graph:
        for year in graph[user].get_out_relation:
            if name_graph in graph[user].get_out_relation[year]:
                for relation in graph[user].get_out_relation[year][name_graph]:
                    users.add(relation.target)
                    users.add(relation.source)
                    number_rel += 1

    average_link_user = round(number_rel / len(users), 1)
    density_degree = round(number_rel / (len(users) * (len(users) - 1)), 4)
    type_graph = "DENSE" if density_degree >= 0.5 else "SPARSE"

    #################################################################################
    #visualization
    row = [["Directed", "True"], [ "Number of users", str(len(users))], ["Number of answers/comments", str(number_rel) ], ["Average number of links per user",str(average_link_user) ],\
           ["Density degree of the graph", str(density_degree)], ["Type of graph", type_graph]]
    rowEvenColor = 'grey'
    rowOddColor = 'white'
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    fig.set_figheight(4)
    fig.set_figwidth(10)
    fig.patch.set_visible(False)
    t = ax.table(cellText=row,
             cellLoc = "center",
                      cellColours=[[rowOddColor,rowOddColor],[rowEvenColor, rowEvenColor]]*3,
                      loc='center')
    t.set_fontsize(14)
    t.scale(1.5, 1.5)
    fig.tight_layout()
	
########################################################################################################################################

#Ex 2.2
def dijkstraNumShortesPath(graph, source, start, end):
    """
    Implements Dijkstra's single source shortest path algorithm
    for a directed graph 
    
            Parameters:
                    graph: the graph we are working on 
                    source (int): the vertex choose as source
                    start (string): beginning date in format "MM/YYYY"
                    end (string): ending date in format "MM/YYYY"
            Returns:
                    prev (dict): a dictionary with vertex as key and as value the list of the previous nodes in the path
                                -1 if it is not reachable
                    dist (dict): a dictionary with vertex as key and the total distance from the source as value
                    path (dict): as key the  node and as value the number of the shortest paths   
    """
    start = convertDate(start)
    end = convertDate(end)
    visited = set() # set of visited node
    unvisited = set(graph) # set of univised node
    dist = dict() # as key the node and as a value the distance between all node
    prev = dict() # as key the node and as a value list of the previous node with min distance
    path = dict() # as key the node and as value the number of the shortest paths
    #initialization
    for u in unvisited:
        dist[u] = float('inf')
        prev[u] = -1
    
    dist[source] = 0    
    visited.add(source)
    # while there node to visited and visited not contain neighbor
    while len(unvisited) > 0 or not set(neighbor).issubset(visited):     
        current_node = getMinUnvisited(unvisited, dist) # get the node with the minimum cost of weight
        unvisited.remove(current_node) 
        visited.add(current_node)
        neighbor = getNeighbors(current_node, graph, start, end)
        # visited neighbors not visited yet 
        for u in unvisited.intersection(set(neighbor)):
            new_dist = dist[current_node] + neighbor[u]
            if new_dist < dist[u]:
                dist[u] = new_dist
                prev[u] = [current_node]
                path[u] = 1
            # if find distance that equals of the shortest, so we take into account all the shortest path
            elif dist[u] == new_dist and dist[u] != float("inf"):
                    path[u] = path.get(u) + 1    # number of the shorted path for the node u
                    prev[u].append(current_node)   
    return prev, dist, path 

def freq_(matrix):
    """
    Function that return the numbers of frequency for each element in the matrix
    
            Parameters:
                    matrix
            Returns:
                    result(dictionary): as the key the element and as value the numbers of frequency for each element
    """
    result = dict()
    for array in matrix:
        for el in array:
            result[el] = result.get(el,0) + 1
    return result


def allShortPath(prev, current, source, path, all_path):
    """
    Function that return all path from the target(current) to the source 
    
            Parameters:
                    prev (dict): a dictionary with vertex as key and as value the list of the previous nodes in the path
                    current(int): the started node
                    source(int): the desiderable node to find the shortest path
                    path(list): empty list
                    all_path(list): empty list
                    
            Returns:
                    result(dictionary): as the key the element and as value the numbers of frequency for each element
    """
    for node in prev[current]:
        if node == source:
            all_path.append(path + [node])
            path = path + [node]
            return all_path
        allShortPath(prev, node, source, path + [node], all_path)
    return all_path

def getDegree(relation, start, end, target_bool=True):
    """
    Update the residual in the path given in input
    
            Parameters:
                    relation(dict): as key the year and as a value a dict that have as a value the type of relation 
                        and as a key the list of all relation
                    start (int): timestamp
                    end (int): timestamp
                    target_bool(boolean): if True out_relation otherwise in_relation                
            Returns:
                    out(int): the degree of the node taken in input
                    node(set): the neighbor of the source
    """
    out = 0
    nodes = set()
    for year in relation:
        for rel in relation[year]:
            for edge in relation[year][rel]:
                if(start <= edge.time <= end):
                    out += edge.weight
                    if target_bool:
                        nodes.add(edge.target)
                    else:
                        nodes.add(edge.source)
    return out, nodes
###########################################################################################################################################

# Ex 2.3

def getMinUnvisited(unvisited, dist):
    """
    return the minimum distance vertex from
    the set of vertices not yet processed.
            
            Parameters:
                    unvisited (set): the set containing all the vertex not yet processed
                    dist (dict): a dictionary with vertex as key and the total distance from the source as value
    """         
    aux = {key: dist[key] for key in unvisited}
    minimum = min(aux.values())
    for key in unvisited:
        if dist[key] == minimum:
            return key

def convertDate(time):
    """
    Returns the converted time, accept only this format DD/MM/YYYY
    
            Parameters:
                    time (string) 
            Returns:
                    time (int): return converted time as this format YYYYMM, so year and month
    """
    tmp = time.split("/")
    return int(tmp[1] + tmp[0])
	
def getShortestPath(source, target, prev, dist):
    """
    Rebuild the shortest path from source to target as a list and its cost
    
            Parameters:
                    source (int): the vertex choose as source
                    target (int): the vertex choose as target
                    prev (dict): a dictionary with vertex as key and last previous vertex in the path from the source as value 
                                -1 if it is not reachable
                    dist (dict): a dictionary with vertex as key and the total distance from the source as value
            Returns:
                    path (list): the sequence of nodes covered
                    cost (int): cost of the path
    """
    path = [target]
    cost = dist[target]
    # go back from target to source using prev dictionary
    while target != source:
        path.append(prev[target])
        target = prev[target]
    path.reverse()
    return path, cost

def getNeighbors(node, graph, start, end):
    """
    Find all the vertex reachable from the current node and the respective cost of the paths updated
    
            Parameters:
                    node (int): the current node
                    graph: the graph we are working on 
                    start (string): beginning date in format "MM/YYYY"
                    end (string): ending date in format "MM/YYYY"
           Returns:
                    neighbors (dict): a dictionary with vertexs nearby node as key and the total distance from the source as value
    """
    neighbors = dict()
    x = graph[node].get_out_relation
    # examinating all the edges getting out from node
    for date in x.keys():
        if start <= date <= end:
            for rel in x[date].keys():
                for edge in x[date][rel]:
                    target = edge.target
                    weight = edge.weight
                    # updating the weight to get from the source to this specific vertex (target)
                    neighbors[target] = neighbors.get(target, weight) + weight
    return neighbors

def myDijkstra(graph, source, start, end):
    """
    Implements Dijkstra's single source shortest path algorithm
    for a directed graph 
    
            Parameters:
                    graph: the graph we are working on 
                    source (int): the vertex choose as source
                    start (string): beginning date in format "MM/YYYY"
                    end (string): ending date in format "MM/YYYY"
            Returns:
                    prev (dict): a dictionary with vertex as key and last previous vertex in the path from the source as value 
                                -1 if it is not reachable
                    dist (dict): a dictionary with vertex as key and the total distance from the source as value    
    """
    # date from string to int
    start = convertDate(start)
    end = convertDate(end)
    
    visited = set()
    unvisited = set(graph.keys())
    dist = dict()
    prev = dict()
    
    # set all the initial values:
    #         - infinity to distances
    #         - -1 to previous node    
    for u in unvisited:
        dist[u] = float('inf')
        prev[u] = -1  
    dist[source] = 0  
    
    # mark source as visited
    visited.add(source)
    
    # iterate until:
    #         - there is something to visit 
    #      or
    #         - all the neighbors of the current node are not marked as visited
    while len(unvisited) > 0 or not set(neighbor.keys()).issubset(visited):
        # choose the correct node 
        current_node = getMinUnvisited(unvisited, dist)
        unvisited.remove(current_node)
        visited.add(current_node)
        
        neighbor = getNeighbors(current_node,graph, start, end)
        
        for u in unvisited.intersection(set(neighbor.keys())):
            # updating the cost if necessary
            new_dist = dist[current_node] + neighbor[u]
            if new_dist < dist[u]:
                dist[u] = new_dist
                prev[u] = current_node      

    return prev, dist

def shortestOrderedRoute(graph, start, end, seq_users, p_1, p_n): 
    """
    Find the shortest ordered route starting from p_1, passing by seq_users
    and ending in p_n
    
            Parameters:
                    graph: the graph we are working on 
                    start (string): beginning date in format "MM/YYYY"
                    end (string): ending date in format "MM/YYYY"
                    seq_users (list): list with all intermediate nodes
                    p_1 (int): starting node
                    p_n (int): ending node
            Returns: 
                    path (list): list containing all the node of the the shortest ordered route
                    weight (int): hom much cost the ordered route

    """
    nodes = [p_1] + seq_users + [p_n]
    # starting the path and its weight
    path = [p_1]
    weight = 0
    
    for i in range(len(nodes)-1):
        # compute the shortest path from nodes[i] to all reachable nodes
        prev, dist = myDijkstra(graph, nodes[i], start, end)
        # find a list of the shortest path from nodes[i] to nodes[i+1] and its weigth
        seq, w = getShortestPath(nodes[i], nodes[i+1], prev, dist)
        # if nodes[i+1] is reachable from nodes[i] update the path and the total weigth
        if w < float('inf'):
            path.extend(seq[1:])
            weight += w
        else:
            print("It is not possible to find the shortest ordered route because node", nodes[i], "and node", nodes[i+1], "are not connected!")
            return 
    return path, weight
	
######################################################################################################################################################

#function 2.4
def graph_by_interval(graph, start, end):
    """
    Return a graph with the desiderable interval
    
            Parameters:
                    graph: the graph we are working on 
                    start (string): beginning date in format "MM/YYYY"
                    end (string): ending date in format "MM/YYYY"
            Returns: 
                    result (class): sub graph with the desiderable interval
                    

    """
    start, end = convertDate(start), convertDate(end)

    result = copy.deepcopy(graph) # deep copy
    users = list(graph)
    
    for user in users:
        out_ = result[user].get_out_relation 
        years = set(out_)
        for year in years:
            if not (start <= year <= end):
                out_.pop(year)
        
        in_ = result[user].get_in_relation
        years = set(in_)
        for year in years:
            if not (start <= year <= end):
                in_.pop(year)
        if len(out_) == 0 and len(in_) == 0:
            result.pop(user)
    return result

def merge(graph1, graph2):
    """
    Return the merge with graph1 and graph2
    
            Parameters:
                    graph1(dictionary): as key node and as value the class of User in the graph  
                    graph2(dictionary): as key node and as value the class of User in the graph  
            Returns: 
                    result (dictionary): the merge between graph1 and graph2
    """
    result = copy.deepcopy(graph1)
    for user in graph2:
        if user not in result:
            result[user] = graph2[user]
        else:
            for year in graph2[user].get_out_relation:
                for type_relation in graph2[user].get_out_relation[year]:
                    for relation in graph2[user].get_out_relation[year][type_relation]:
                        result[user].add_out_relation(relation)
            for year in graph2[user].get_in_relation:
                for type_relation in graph2[user].get_in_relation[year]:
                    for relation in graph2[user].get_in_relation[year][type_relation]:
                        result[user].add_in_relation(relation)
    return result


def createResidualG(graph):
    return copy.deepcopy(graph)

def getNeighborsMinWeight(node, graph, start, end):
    """
    Return a graph with the desiderable interval
    
            Parameters:
                    node: the node that we would to find all neighbor
                    graph: the graph we are working on 
                    start (string): beginning date in format "MM/YYYY"
                    end (string): ending date in format "MM/YYYY"
            Returns: 
                    result (dict): contains as key all neighbors and as value for each key the weight 
                    (if exists 2 or more edges between source and neighbor we take the edge with the minimum cost)"
                    
    """
    neighbors = dict()
    x = graph[node].get_out_relation
    for date in x.keys():
        if start <= date <= end:
            for rel in x[date].keys():
                for edge in x[date][rel]:
                    neighbors[edge.target] = min(neighbors.get(edge.target, float('inf')), edge.weight)
    x = graph[node].get_in_relation
    # we take into account also the the edges in relation
    for date in x.keys():
        if start <= date <= end:
            for rel in x[date].keys():
                for edge in x[date][rel]:
                    neighbors[edge.source] = min(neighbors.get(edge.source, float('inf')), edge.weight_in)
    return neighbors

def getBottleneck(path, source, target):
    """
    Return the bottleneck, so the edges with the minimum cost in the path
    
            Parameters:
                    graph: the graph we are working on 
                    start (string): beginning date in format "MM/YYYY"
                    end (string): ending date in format "MM/YYYY"
            Returns: 
                    bottleneck (int): minimum cost in the path
                    

    """
    current_value = target
    bottleneck = float("inf")
    while current_value != source:
        bottleneck = min(bottleneck, path[current_value][1])
        current_value = path[current_value][0]
    return bottleneck

def BFS(graph, s, t, path, start=200808, end=201603):
    """
    return two values, the first one if exists a path between s and t, and the second one is the path among s and t 
    
            Parameters:
                    graph: the graph we are working on 
                    path(dictionary): as key all node of the graph and as value -1
                    flow(int): value used to update the residual in the path
                    s(int): source -> starting point
                    t(int): target -> end point
            
            Returns:
                    boolean: if exists a path between s and t
                    path(dictionary): as key the node and as value the parent node if exists, otherwise -1

    """
    visited = set()
    queue = [s]
    visited.add(s)
    if s not in path: path[s] = ()
    while len(queue) > 0:
        source = queue.pop(0)
        neighbors = getNeighborsMinWeight(source, graph, start, end)
        for target in neighbors:
            if target not in visited and neighbors[target] > 0:
                queue.append(target)
                visited.add(target)
                path[target] = (source, neighbors[target])
    if t in visited:  return True, path       
    return False, path

def updateResGraph(graph, path, flow, s, t):
    """
    Update the residual in the path given in input
    
            Parameters:
                    graph: the graph we are working on 
                    path(dictionary): as key the node and as value the parent node
                    flow(int): value used to update the residual in the path
                    s(int): source -> starting point
                    t(int): target -> end point
    """
    walk = [t]
    current_value = t
    while current_value != s:
        walk.append(path[current_value][0])
        current_value = path[current_value][0]
    walk.reverse() 
    for i in range(len(walk)-1):
        el = walk[i]
        out_ = graph[el].get_out_relation
        for year in out_:
            for rel in out_[year]:
                for ind in range(len(out_[year][rel])):
                    if out_[year][rel][ind].target == walk[i+1]:
                        w = out_[year][rel][ind].weight - flow
                        out_[year][rel][ind].set_weight(w)
                        w_in = out_[year][rel][ind].weight_in + flow
                        out_[year][rel][ind].set_weight_in(w_in)
                        
def reachFromS(graph, source, start, end):
    """
    Return all node that reach from s
    
            Parameters:
                    graph: the graph we are working on
                    source: the starting node that we would to find all node reach from source
                    start (string): beginning date in format "MM/YYYY"
                    end (string): ending date in format "MM/YYYY"
            Returns: 
                    result (class): sub graph with the desiderable interval
                    
    """
    path = dict()
    path = BFS(graph, source, 0, path, start, end)[1]
    return path.keys()

#################################################################################################################
#views ex 2.4

def getEdges(G, path):
    """
    Return a graph with the desiderable interval
    
            Parameters:
                    G: the graph we are working on 
                    path(dictionary): as key the node and as value the parent node
            Returns: 
                    e1(list): all edges that weight is equal to 1
                    e2(list): all edges that weight is equal to 2
                    e3(list): all edges that weight is equal to 3
                    edge_path1: all edges of the shortes path that weight is equal to 1
                    edge_path2: all edges of the shortes path that weight is equal to 2
                    edge_path3: all edges of the shortes path that weight is equal to 3                 
    """
    e1 = []
    e2 = []
    e3 = []   
    edge_path1 = []
    edge_path2 = []
    edge_path3 = []
    
    edge_path = []
    
    for i in range(len(path)-1):
        edge_path.append((path[i],path[i+1]))

    for (x, y, w) in G.edges(data = True):
        if x != y and w["weight"] == 1:
            e1.append((x,y))
            if (x,y) in edge_path:
                edge_path1.append((x,y))             
        elif x != y and w["weight"] == 2:
            e2.append((x,y))
            if (x,y) in edge_path:
                edge_path2.append((x,y))                     
        elif x != y and w["weight"] == 3:
            e3.append((x,y))       
            if (x,y) in edge_path:
                edge_path3.append((x,y))  
                
    return e1, e2, e3, edge_path1, edge_path2, edge_path3
##################################################################################################################

def graph_to_networkx(graph, type_graph): 
    """
    Trasform the our graph in Networkx object
    
            Parameters:
                    graph (dict): as a key we have the user ID and as a value we have the class associate of the User 
                    type_graph(string): the type of graph that we want to obtain (c2a, c2q, a2q), 
                        NOTE: if type_graph is all, this mean we must to add all type of node
    
            Returns:
                    G(networkx): the graph
    
    """
	
    G = nx.DiGraph()
    for user in graph:
        for year in graph[user].get_out_relation:
            for type_relation in graph[user].get_out_relation[year]:
                if type_relation == type_graph or type_graph.lower() == 'all':
                    for relation in graph[user].get_out_relation[year][type_relation]:
					    #create node
                        G.add_nodes_from([user, relation.target])
						# create edge
                        G.add_edge(user, relation.target, weight=relation.weight, time=year)
    return G


def graph_by_user(graph, source, target, start, end):
    """
    Return a graph with the desiderable interval
    
            Parameters:
                    graph: the graph we are working on 
                    source(int): source -> starting point
                    target(int): target -> end point
            Returns: 
                    result (dict): sub graph with the desiderable node
    """
    result = copy.deepcopy(graph)
    users = list(graph)
    s = set(reachFromS(graph, source, start, end))
    for user in users:
        if user not in s:
            result.pop(user)
    return result