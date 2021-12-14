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
def get_features(name_graph):
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
    users = set() # varible used to have all users and in the final count them
    number_rel = 0 # variable to use to count the number of edge
    for user in graph:
        for year in graph[user].get_out_relation:
            if name_graph in graph[user].get_out_relation[year]:
                for relation in graph[user].get_out_relation[year][name_graph]:
                    users.add(relation.target)
                    users.add(relation.source)
                    number_rel += 1
	
    average_link_user = round(number_rel / len(users), 1)
    density_degree = round(number_rel / (len(users) * (len(users) - 1)), 4)
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'
    type_graph = "DENSE" if density_degree >= 0.5 else "SPARSE"
	
	##################################################################################################
	### create Table for user 
    first =  ["Directed", "Number of users", "Number of answers/comments", "Average number of links per user", "Density degree of the graph", "Type of graph"]
    second = ["True", str(len(users)), str(number_rel), str(average_link_user), str(density_degree), type_graph]

    fig = go.Figure(data=[go.Table(
                header=dict(values= ["Request", "Response"], line_color='darkslategray', align=['left','center'],
                    fill_color='grey', font=dict(color='black', size=11)
                    ), 
                cells = dict(values=[first, second], line_color='darkslategray',
                        fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor, rowEvenColor]*2],font=dict(color='darkslategray', size=11),
                         align=['left', 'center'])
                                  )    
                         ])
    
    fig.update_layout(width=400, height=450)
    fig.show(width=300, height=200)
	
########################################################################################################################################

#Ex 2.2
def GetCloseness(graph, source, start, end):
    distances = 0
    keys = set(graph.keys())
    keys.remove((source))
    _, cost = myDijkstra(graph, source, start,end)
    cost = np.array(list(cost.values()))
    position = np.where(cost != float('inf'))[0]
    if sum(cost[position]) == 0: return 0
    return round((len(position) - 1) / sum(cost[position]),3)
	
	
def degreeCentrality(graph, source, start, end):
    start = convertDate(start)
    end = convertDate(end)
    weight = 0
    g = graph[source].get_out_relation
    for year in g:
        for rel in g[year]:
            for edge in g[year][rel]:
                weight += edge.weight
    return weight/(len(graph)-1)

###########################################################################################################################################

# Ex 2.3
def getMinUnvisited(unvisited, dist):
    result, dist_min = -1 , float('inf')
    
    aux = {key: dist[key] for key in unvisited}
    return min(aux, key=aux.get)
	
def getMinUnvisited(unvisited, dist):
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
    path = [target]
    cost = dist[target]
    while target != source:
        path.append(prev[target])
        target = prev[target]
    path.reverse()
    return path, cost

def getNeighbors(node, graph, start, end):
    neighbors = dict()
    x = graph[node].get_out_relation
    for date in x.keys():
        if start <= date <= end:
            for rel in x[date].keys():
                for edge in x[date][rel]:
                    target = edge.target
                    weight = edge.weight
                    neighbors[target] = neighbors.get(target, weight) + weight
    return neighbors

def myDijkstra(graph, source, start, end):
    start = convertDate(start)
    end = convertDate(end)
    visited = set()
    unvisited = set(graph.keys())
    dist = dict()
    prev = dict()
    for u in unvisited:
        dist[u] = float('inf')
        prev[u] = -1  
    dist[source] = 0    
    visited.add(source)
    while len(unvisited) > 0 or not set(neighbor.keys()).issubset(visited):     
        current_node = getMinUnvisited(unvisited, dist)
        unvisited.remove(current_node)
        visited.add(current_node)
        neighbor = getNeighbors(current_node,graph, start, end)
        for u in unvisited.intersection(set(neighbor.keys())):
            new_dist = dist[current_node] + neighbor[u]
            if new_dist < dist[u]:
                dist[u] = new_dist
                prev[u] = current_node      

    return prev, dist

def shortestOrderedRoute(graph, start, end, seq_users, p_1, p_n):
    nodes = [p_1] + seq_users + [p_n]
    path = [p_1]
    weight = 0
    
    for i in range(len(nodes)-1):
        prev, dist = myDijkstra(graph, nodes[i], start, end)
        seq, w = getShortestPath(nodes[i], nodes[i+1], prev, dist)
        if w < float('inf'):
            path.extend(seq[1:])
            weight += w
        else:
            print("It is not possible to find the shortest ordered route because node", nodes[i], "and node", nodes[i+1], "are not connected!")
            return 
    return path, weight
	
######################################################################################################################################################
	
def graph_to_networkx(graph, type_graph): 
    """
    Trasform the our graph in Networkx object
    
            Parameters:
                    graph (dict): as a key we have the user ID and as a value we have the class associate of the User 
                    type_graph(string): the type of graph that we want to obtain (c2a, c2q, a2q), NOTE: if type_graph is all, this mean we must to add all type of node
    
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


