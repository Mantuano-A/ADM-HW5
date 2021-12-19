import random
import math
from utils_function import *

def networkx_to_subgraph(graph,path, start, end):
    
    # transforming our graph in netwrokx graph
    G = graph_to_networkx(graph, "all")
    perc = 0.3
    nodes = [*set(path)]
    
    for user in path:
        neighbor = list(getNeighbors(user,graph, start, end))
        nodes.extend(random.sample(neighbor,math.ceil(perc * len(neighbor))))
        
    return G.subgraph(nodes)

def visualizationShortestPath(G,path):
    G = graph_to_networkx(G, "all")
    plt.figure(figsize = (10,8))
    walk = [*range(0,len(path))]
    newG = nx.DiGraph()
    edges = []
    e1 = []
    e2 = []
    e3 = []   
    
    for i in range(len(path[:-1])):
        el1 = path[i]
        el2 = path[i+1]
        for (x, y, w) in G.edges(data = True):
            if x == el1 and y == el2:
                edges.append((i,i+1,w))
                
    newG.add_edges_from(edges)
    
    for (x, y, w) in newG.edges(data = True):
        if x != y and w["weight"] == 1:
            e1.append((x,y))
        elif x != y and w["weight"] == 2:
            e2.append((x,y))             
        elif x != y and w["weight"] == 3:
            e3.append((x,y)) 
            
    walk.reverse()
    pos = {walk[i] : [i%4,i//4] for i in walk}
    walk.reverse()
    nx.draw_networkx_nodes(newG, pos, node_size = 1800, nodelist = walk[1:-1], alpha = 0.6, node_color = "orange", label = "path nodes",edgecolors = "darkred")
    
    nx.draw_networkx_nodes(newG, pos, node_size = 2000, nodelist = [0],alpha = 0.8, node_color = "gold", label = "start node",edgecolors = "darkorange")
    
    nx.draw_networkx_nodes(newG, pos, node_size = 2000, nodelist = walk[-1:], alpha = 0.5, node_color = "red", label = "path nodes",edgecolors = "darkred")
    
    labels = nx.draw_networkx_labels(newG,pos,{i: path[i] for i in walk},font_size = 20,font_color='black')

    labels1 = nx.draw_networkx_edge_labels(newG,pos,edge_labels = {i: 1 for i in e1},font_size = 18,font_color='black')
    
    labels2 = nx.draw_networkx_edge_labels(newG,pos,edge_labels = {i: 2 for i in e2},font_size = 18,font_color='black')
    
    labels3 = nx.draw_networkx_edge_labels(newG,pos,edge_labels = {i: 3 for i in e3},font_size = 18,font_color='black')
    
    nx.draw_networkx_edges(G,pos, edgelist = edges, width = 3, arrowstyle = "-|>", arrowsize = 15, edge_color = "black", alpha = 1 , node_size = 2100, label = "ciao" )
    
    plt.axis("off")
    plt.show()
    

def visualizationGeneralShortestPath(G,path, start, end, k ,seed):
    
    G = networkx_to_subgraph(G,path, start, end)
    plt.figure(figsize = (20,15))

    pos = nx.spring_layout(G, k = k, iterations=20, seed = seed)
    
    nodes = G.nodes
    e1, e2, e3, edge_path1, edge_path2, edge_path3 = getEdges(G, path)
    start_node = path[:1]
    end_node = path[-1:]
    rem = {start_node[0],end_node[0]}
    path = set(path) - rem
    
    nx.draw_networkx_nodes(G, pos, node_size = math.ceil(800*40/len(nodes)), nodelist = set(nodes) - set(path) - rem, alpha = 1, node_color = "white", label = "neighbour nods", edgecolors = "black")
    
    nx.draw_networkx_nodes(G, pos, node_size = math.ceil(1000*50/len(nodes)), nodelist = path, alpha = 0.6, node_color = "orange", label = "path nodes",edgecolors = "darkred")
    
    nx.draw_networkx_nodes(G, pos, node_size = math.ceil(1300*50/len(nodes)), nodelist = start_node,alpha = 0.9, node_color = "gold", label = "start node",edgecolors = "darkorange")
    
    nx.draw_networkx_nodes(G, pos, node_size = math.ceil(1300*50/len(nodes)), nodelist = end_node, alpha = 0.5, node_color = "red", label = "end node",edgecolors = "darkred")
    
    font = 12
    labels = nx.draw_networkx_labels(G,pos,{i: i for i in (set(path).union(rem))},font_size = font,font_color='black')

    nx.draw_networkx_edges(G,pos, edgelist = e1, width = round(0.4*30/len(nodes),1), arrowstyle = "-|>", arrowsize = 8, edge_color = "gray", alpha = 1,connectionstyle = "Arc3, rad=0.6" , node_size = math.ceil(1000*50/len(nodes)), label = "ciao" )
    
    nx.draw_networkx_edges(G,pos, edgelist = e2, width = round(0.8*30/len(nodes),1), arrowstyle = "-|>", arrowsize = 8, edge_color = "gray", alpha = 1,connectionstyle = "Arc3, rad=-0.2", node_size = math.ceil(1000*50/len(nodes)), label = "ciao" )
    
    nx.draw_networkx_edges(G,pos, edgelist = e3, width = round(1.2*30/len(nodes),1), arrowstyle = "-|>", arrowsize = 8, edge_color = "gray", alpha = 1,connectionstyle = "Arc3, rad=0.2" , node_size = math.ceil(1000*50/len(nodes)), label = "ciao" )
    
    nx.draw_networkx_edges(G,pos, edgelist = edge_path1, width = round(1.5*30/len(nodes),1), arrowstyle = "-|>", arrowsize = 15, edge_color = "black", alpha = 1,connectionstyle = "Arc3, rad=0.2" , node_size = math.ceil(1400*50/len(nodes)), label = "ciao" )
    
    nx.draw_networkx_edges(G,pos, edgelist = edge_path2, width = round(1.5*30/len(nodes),1), arrowstyle = "-|>", arrowsize = 15, edge_color = "black", alpha = 1,connectionstyle = "Arc3, rad=-0.2" , node_size = math.ceil(1400*50/len(nodes)), label = "ciao" )
    
    nx.draw_networkx_edges(G,pos, edgelist = edge_path3, width = round(1.5*30/len(nodes),1), arrowstyle = "-|>", arrowsize = 15, edge_color = "black", alpha = 1,connectionstyle = "Arc3, rad=0.2" , node_size = math.ceil(1400*50/len(nodes)), label = "ciao" )
    
    plt.axis("off")
    plt.legend(fontsize = 15, markerscale = 0.5)
    plt.show()
    
def visualization3(graph, path, start, end, k, seed):
    
    visualizationGeneralShortestPath(graph,path,start,end,k,seed)
    
    visualizationShortestPath(graph,path)
    
def visualization4(graph, source, target, k, seed,cut):
    G = graph_to_networkx(graph, "all")
    plt.figure(figsize = (20,15))

    pos = nx.spring_layout(G, k = k, iterations=20, seed = seed)
    
    nodes = G.nodes
    e1, e2, e3, _, _, _ = getEdges(G, [])
    
    nx.draw_networkx_nodes(G, pos, node_size = math.ceil(800*40/len(nodes)), nodelist = set(nodes) - set(source,target), alpha = 1, node_color = "white", label = "neighbour nods", edgecolors = "black")
        
    nx.draw_networkx_nodes(G, pos, node_size = math.ceil(1300*50/len(nodes)), nodelist = [source,target],alpha = 0.8, node_color = "gold", label = "start node",edgecolors = "darkorange")    
    
    font = 14
    labels = nx.draw_networkx_labels(G,pos,{i: i for i in [source,target]},font_size = font,font_color='black')

    nx.draw_networkx_edges(G,pos, edgelist = e1, width = round(0.4*30/len(nodes),1), arrowstyle = "-|>", arrowsize = 8, edge_color = "gray", alpha = 1,connectionstyle = "Arc3, rad=0.6" , node_size = math.ceil(1000*50/len(nodes)))
    
    nx.draw_networkx_edges(G,pos, edgelist = e2, width = round(0.8*30/len(nodes),1), arrowstyle = "-|>", arrowsize = 8, edge_color = "gray", alpha = 1,connectionstyle = "Arc3, rad=-0.2", node_size = math.ceil(1000*50/len(nodes)))
    
    nx.draw_networkx_edges(G,pos, edgelist = e3, width = round(1.2*30/len(nodes),1), arrowstyle = "-|>", arrowsize = 8, edge_color = "gray", alpha = 1,connectionstyle = "Arc3, rad=0.2" , node_size = math.ceil(1000*50/len(nodes)))
 
    nx.draw_networkx_edges(G,pos, edgelist = cut, width = round(1.5*30/len(nodes),1), arrowstyle = "-|>", arrowsize = 10, edge_color = "red", alpha = 1,connectionstyle = "Arc3, rad=0.2" , node_size = math.ceil(1400*50/len(nodes)))

    plt.axis("off")
    plt.show()