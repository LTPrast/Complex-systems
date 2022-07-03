import networkx as nx
import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
import operator
from collections import Counter
from math import ceil

# Initializes the network based on the data from the paper
def initialize_network():
    # using Table S-3 and S-4 from Appendix
    # Step 1: make seperate barabasi albert graphs for every role, 
    # with its known #nodes and average degree (Table S-4)
    # Step 2: make G_VC as disjoint union of these "role" graphs
    # Step 3: Add #edges specified in Table S-4 between random nodes in two roles
    # after filtering Table S-4 to remove duplicates

    df = pd.read_csv('Appendix_Table_S_4_corrected.csv', sep=';')  
    df['C_deg'] = df['C_deg'].astype(float)
    df['D_vc'] = df['D_vc'].astype(float)
    df['Role'] = df['Role'].astype(str).str.strip()
    table = df.to_dict('records')

    G_VC = nx.empty_graph()
    for row in table:
        # average degree is np
        G = nx.barabasi_albert_graph(row['N'], ceil(row['D_vc']/row['N']), seed=41)
        nx.set_node_attributes(G, row['Role'], "role")
        G_VC = nx.disjoint_union(G_VC, G)

    df2 = pd.read_csv('Appendix_Table_S_3_corrected.csv', sep=';')
    df2['Role_1'] = df2['Role_1'].astype(str).str.strip()
    df2['Role_2'] = df2['Role_2'].astype(str).str.strip()
    df2['Edges'] = df2['Edges'].astype(int)

    # dropping duplicates https://stackoverflow.com/a/55425400
    df2 = df2[~df2[['Role_1','Role_2']].apply(frozenset,axis=1).duplicated()].reset_index(drop=True)
    df2 = df2[df2['Role_1'] != df2['Role_2']].reset_index(drop=True)

    for index, row in df2.iterrows():
        role_1_nodes = [x for x,y in G_VC.nodes(data=True) if y['role']==row['Role_1']]
        role_2_nodes = [x for x,y in G_VC.nodes(data=True) if y['role']==row['Role_2']]
        # add #Edges between random nodes in role 1 & 2 that don't already share an edge
        i = 0
        while i < row['Edges']:
            u = rd.choice(role_1_nodes)
            v = rd.choice(role_2_nodes)
            # if not G_VC.has_edge(u, v):
            G_VC.add_edge(u, v)
            i += 1

    # dictionary of nodes list for every role
    roles = list(df['Role'].unique())
    roles_dict = {}
    for role in roles:
        nodes = []
        for node in G_VC.nodes(data=True):
            if node[1]['role'] == role:
                nodes.append(node[0])
        roles_dict[role] = nodes
    return G_VC, df, roles_dict

# add a somewhat random macro network
def add_macro_basic(G_VC):
    # set attribute VC = True for G_VC nodes 
    nx.set_node_attributes(G_VC, [True], "VC")

    G_macro = nx.barabasi_albert_graph(2000, 1)
    # set attribute VC = False for G_macro nodes 
    nx.set_node_attributes(G_macro, [False], "VC")
    # G_macro = nx.convert_node_labels_to_integers(G_macro, len(G_VC))

    # assign nodes a random role
    # random weighted choice according to fraction of role from whole in G_VC
    df = pd.read_csv('Appendix_Table_S_4_corrected.csv', sep=';')
    df['Role'] = df['Role'].astype(str).str.strip()
    roles = df['Role'].tolist()
    weights = (df['N'] / df['N'].sum()).tolist()
    
    for node in G_macro.nodes(data=True):
        role = np.random.choice(roles, 1, weights)
        node[1]['role'] = role[0]

    # combine G_VC and G_macro
    G_combined = nx.disjoint_union(G_VC, G_macro)

    # node in G_VC is more probable to have more connections to G_macro
    # if its role is heigher up in value chain
    for node in list(G_VC.nodes()):
        role = G_VC.nodes[node]['role']
        if role in ['Financing', 'Coordinator', 'Growshop owner']:
            random_edges = rd.randint(1,10)
        else:
            random_edges = rd.randint(1,2)
        
        for i in range(random_edges):
            G_combined.add_edge(node, rd.choice(list(G_macro.nodes())))
    return G_macro, G_combined

# add a macro network based on statistics
def add_macro_stats(G_VC):
    # set attribute VC = True for G_VC nodes 
    nx.set_node_attributes(G_VC, [True], "VC")

    factor = 3

    df = pd.read_csv('Appendix_Table_S_4_corrected.csv', sep=';')  
    df['C_deg'] = df['C_deg'].astype(float) * factor
    df['D_vc'] = df['D_vc'].astype(float) * factor
    df['Role'] = df['Role'].astype(str).str.strip()
    df['N'] = df['N'] * factor
    table = df.to_dict('records')

    G_macro = nx.empty_graph()
    for row in table:
        # average degree is np
        G = nx.barabasi_albert_graph(row['N'], ceil(row['D_vc']/row['N']), seed=41)
        nx.set_node_attributes(G, row['Role'], "role")

        G_macro = nx.disjoint_union(G_macro, G)

    df2 = pd.read_csv('Appendix_Table_S_3_corrected.csv', sep=';')
    df2['Role_1'] = df2['Role_1'].astype(str).str.strip()
    df2['Role_2'] = df2['Role_2'].astype(str).str.strip()
    df2['Edges'] = df2['Edges'].astype(int) * factor

    # dropping duplicates https://stackoverflow.com/a/55425400
    df2 = df2[~df2[['Role_1','Role_2']].apply(frozenset,axis=1).duplicated()].reset_index(drop=True)
    df2 = df2[df2['Role_1'] != df2['Role_2']].reset_index(drop=True)

    for index, row in df2.iterrows():
        role_1_nodes = [x for x,y in G_macro.nodes(data=True) if y['role']==row['Role_1']]
        role_2_nodes = [x for x,y in G_macro.nodes(data=True) if y['role']==row['Role_2']]
        # add #Edges between random nodes in role 1 & 2
        i = 0
        while i < row['Edges']:
            u = rd.choice(role_1_nodes)
            v = rd.choice(role_2_nodes)
            G_macro.add_edge(u, v)
            i += 1 
    # set attribute VC = False for G_macro nodes 
    nx.set_node_attributes(G_macro, [False], "VC")
    # G_macro = nx.convert_node_labels_to_integers(G_macro, len(G_VC))
    
    # combine G_macro and G_VC
    G_combined = nx.disjoint_union(G_VC, G_macro)

    # node in G_VC is more probable to have more connections to G_macro
    # if its role is heigher up in value chain
    for node in list(G_VC.nodes()):
        role = G_VC.nodes[node]['role']
        if role in ['Financing', 'Coordinator', 'Growshop owner']:
            random_edges = rd.randint(1,10)
        else:
            random_edges = rd.randint(0,2)
        for i in range(random_edges):
            G_combined.add_edge(node, rd.choice(list(G_macro.nodes())))

    return G_macro, G_combined

# prints & returns number of nodes, number of edges, average degree
# avgerage shortest path, Diameter, Largest component
# and given a list of roles, returns dataframe with 
# number of nodes, number of edges, average degree for each role
def network_stats(G, roles=[]):
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    average_degree = sum(n for _, n in G.degree())/nodes
    shortest_path = nx.average_shortest_path_length(G)
    diameter = nx.diameter(G)
    largest_component = len(max(nx.connected_components(G), key=len))

    print("Number of nodes: ", nodes)
    print("Number of edges: ", edges)
    print("Average degree: ", average_degree)
    print("Avgerage shortest path: ", shortest_path)
    print("Diameter: ", diameter)
    print("Largest component: ", largest_component)

    if roles == []:
        return [nodes, edges, average_degree, shortest_path, diameter, largest_component]
    else:
        df_roles_stats = pd.DataFrame(columns = ['Role', 'Number of nodes', 'Number of edges', 'Average degree'])
        for role in roles_dict:
            nodes = [node for node, data in G.nodes(data=True) if data.get("role") == role]
            subgraph = G.subgraph(nodes)
            # print(role)
            # subgraph = G.subgraph(roles_dict[role])
            # print(subgraph)
            nodes = subgraph.number_of_nodes()
            edges = subgraph.number_of_edges()
            average_degree = sum(n for _, n in subgraph.degree())/nodes
            df_roles_stats = df_roles_stats.append({'Role': role, 
            'Number of nodes': nodes, 'Number of edges': edges, 
            'Average degree': average_degree}, ignore_index = True)
        print(df_roles_stats)
        return [nodes, edges, average_degree, shortest_path, diameter, largest_component, df_roles_stats]


#dictionary of all nodes and their degree to remove node with highest degree
def get_nodes_degree(G):
  node_links = {}
  for i in G.degree():
    node_links[i[0]] = i[1]
  return node_links

# To get all the links of the removed node
def links_removed_node(G, node_most_links):
  links_of_removed_node = [ ]
  for i in G.edges(node_most_links):
    links_of_removed_node.append(i[1])
  return links_of_removed_node

# To calculate the part of the nodes that belong to the GC
def giant_component_perc(G):
  n_giant_component = len(sorted(nx.connected_components(G), key=len, reverse=True)[0])
  n_nodes = G.number_of_nodes()
  percentage_in_giant = n_giant_component/n_nodes * 100
  return percentage_in_giant 

# Calculates efficiency of value chain network
def calc_efficiency(G):
  efficiency_cycle = nx.global_efficiency(G)
  efficiency_cycle_perc = efficiency_cycle*100
  return efficiency_cycle_perc
  
# Calculates efficiency of macro network
def calc_efficiency_macro(G_VC, G):
    sum = 0
    for node_i in list(G_VC.nodes()):
        for node_j in list(G_VC.nodes()):
            if node_i != node_j:
                sum += nx.efficiency(G, node_i, node_j)
    n = G_VC.number_of_nodes()
    return sum / (n *(n-1)) * 100

""" structural(social capital) disruption strategies : 1. Random   2.Hub   3.Broker(between centrality)"""
""" removal of a random node/actor """
def removal_random(G):
  #remove a random node
  nodes = list(G.nodes())
  rd_node = rd.choice(nodes)

  #getting the links of the removed node so that we link the chosen node
  #to those nodes
  links_of_removed_node = links_removed_node(G, rd_node)  
  removed_node_role = G.nodes[rd_node]["role"]

  G.remove_node(rd_node)
  return removed_node_role,rd_node, links_of_removed_node

""" removal of Hub : Example - financer """
def removal_highest_degree(G):
  #dictionary of all nodes and their degree to remove node with highest degree
  node_links = get_nodes_degree(G)
  node_most_links = max(node_links, key=node_links.get)

  #getting the links of the removed node so that we link the chosen node
  #to those nodes
  links_of_removed_node = links_removed_node(G, node_most_links)

  """ Added on 23rd June : Harshita"""
  removed_node_role = G.nodes[node_most_links]["role"]

  G.remove_node(node_most_links)
  return removed_node_role,node_most_links, links_of_removed_node

""" removal of Broker : Example - Coordinator """
def removal_highest_betweeness(G):

  #dictionary of the betweenness_centrality and removal of the node with
  # highest betweenness centrality to remove the node with highest betweenness centrality
  betweenness_centrality = nx.betweenness_centrality(G)
  largest_betweenness_centrality = max(betweenness_centrality, key=betweenness_centrality.get)
  
  #getting the links of the removed node so that we can link the nodes with recovery algorithm
  links_of_removed_node = links_removed_node(G, largest_betweenness_centrality)

  """ Added on 23rd June : Harshita"""
  removed_node_role = G.nodes[largest_betweenness_centrality]["role"]

  G.remove_node(largest_betweenness_centrality)
  return removed_node_role,largest_betweenness_centrality, links_of_removed_node


## disrupting VC degree (2 versions) ##
# importance is assigned according to how high in the hierarchy the role is (paper figure 3)
VC_importance = {'Financing': 10, 'Coordinator': 9, 'Growshop owner': 7, 
    'Arranging location for plantation': 5, 'Supply of growth necessities': 5, 
    'Taking care of plants': 5, 'Cutting toppings': 5, 'Adding weight to the toppings': 5,
    'Transporting': 5, 'Arranging fake owners of property': 4, 'Diverting electricity': 4,
    'Protection of plantation': 4, 'Controlling cutters': 4, 'Drying toppings': 4, 
    'Selling to coffeeshops': 4, 'Fake owner of a company or house': 3, 'Building a plantation': 3,
    'Disposing waste and leftovers': 3, 'Arranging storage': 3, 'International trade': 3,
     'Supply of cuttings for plants': 2}
# why use unique neighbors:
# if a node is connected to only one Coordinator (9) and another connected to 4 Arranging storage (12)
# then it will overpower it even though that is not necessarily the case 
# that's why we'll only consider unique roles of neighbors
# version 1: without weights (only number of unique neighbors)
# version 2: assign weight for unique neighbors (based on figure 3)
# e.g. if a node is connected to only one Coordinator (9) and another connected to 
# Drying topping and Arranging toppins (4+3=7), so if we disrupt the node with Coordinator neighbor 
# then it increases the chances of exposing a crucial member of the value chain

# adds VC degree (without weights) as attribute to each node
def VC_degree_attribute(G):
    for node in G.nodes(data=True):
        neighbors = list(G.neighbors(node[0]))
        neighbors_roles = [G.nodes[neighbor]["role"] for neighbor in neighbors]
        unique_roles = list(dict.fromkeys(neighbors_roles))
        
        node[1]['VC_degree'] = len(unique_roles)


# adds VC degree with weights as attribute to each node
def VC_degree_attribute_weighted(G):
    for node in G.nodes(data=True):
        neighbors = list(G.neighbors(node[0]))
        neighbors_roles = [G.nodes[neighbor]["role"] for neighbor in neighbors]
        unique_roles = list(dict.fromkeys(neighbors_roles))

        VC_degree_weighted = 0
        for role in unique_roles:
            #print(role)
            #print(VC_importance[role])
            VC_degree_weighted  =  VC_degree_weighted + VC_importance[role]
        
        node[1]['VC_degree_weighted'] = VC_degree_weighted
      

# removes node at random from set of nodes with highest VC degree
def disrupt_VC_degree(G):
    # unfreeze graph first by creating a copy
    #G = nx.Graph(G)

    VC_degrees_dict = nx.get_node_attributes(G, "VC_degree")
    max_VC_degree = max(VC_degrees_dict.values())
    nodes_max_VC_degree = [key for key in VC_degrees_dict if VC_degrees_dict[key] == max_VC_degree]
    removed_node = rd.choice(nodes_max_VC_degree)
    # before removing node, find its neighbors (will be orphans)
    orphaned_nodes = list(G.neighbors(removed_node))

    #getting the links of the removed node so that we link the chosen node
    #to those nodes
    links_of_removed_node = links_removed_node(G, removed_node)
    # get role of removed node
    removed_node_role = G.nodes[removed_node]["role"]
    G.remove_node(removed_node)

    # update VC_degree and VC_degree_weighted of orphaned nodes
    for node in orphaned_nodes:
        neighbors = list(G.neighbors(node))
        neighbors_roles = [G.nodes[neighbor]["role"] for neighbor in neighbors]
        unique_roles = list(dict.fromkeys(neighbors_roles))
        G.nodes[node]['VC_degree'] = len(unique_roles)
        #VC_degree_weighted = 0
        #for role in unique_roles:
        #    VC_degree_weighted += VC_importance[role]
        #G.nodes[node]['VC_degree_weighted'] = VC_degree_weighted

    return removed_node_role,removed_node, links_of_removed_node

# removes node at random from set of nodes with highest weighted VC degree
def disrupt_VC_degree_weighted(G):
    # unfreeze graph first by creating a copy
    #G = nx.Graph(G)

    VC_degrees_dict = nx.get_node_attributes(G, "VC_degree_weighted")
    max_VC_degree = max(VC_degrees_dict.values())
    nodes_max_VC_degree = [key for key in VC_degrees_dict if VC_degrees_dict[key] == max_VC_degree]

    removed_node = rd.choice(nodes_max_VC_degree)

    # before removing node, find its neighbors (will be orphans)
    orphaned_nodes = list(G.neighbors(removed_node))

    #getting the links of the removed node so that we link the chosen node
    #to those nodes
    links_of_removed_node = links_removed_node(G, removed_node)
    # get role of removed node
    removed_node_role = G.nodes[removed_node]["role"]

    G.remove_node(removed_node)

    # update VC_degree and VC_degree_weighted of orphaned nodes
    for node in orphaned_nodes:
        neighbors = list(G.neighbors(node))
        neighbors_roles = [G.nodes[neighbor]["role"] for neighbor in neighbors]
        unique_roles = list(dict.fromkeys(neighbors_roles))
        G.nodes[node]['VC_degree'] = len(unique_roles)

        VC_degree_weighted = 0
        for role in unique_roles:
            VC_degree_weighted += VC_importance[role]
        
        G.nodes[node]['VC_degree_weighted'] = VC_degree_weighted
    
    return removed_node_role,removed_node, links_of_removed_node

# removes not add random with specific role (in paper 'Diverting electricity')

# removes not add random with specific role (in paper 'Diverting electricity')
def disrupt_VC_role(G):
    # roles_dict = nx.get_node_attributes(G, "role")
    role = 'Diverting electricity' # as in paper
    # nodes_with_role = roles_dict[role]
    nodes_with_role = [x for x,y in G.nodes(data=True) if y['role']== role]
    removed_node = rd.choice(nodes_with_role)
    #getting the links of the removed node so that we link the chosen node
    #to those nodes
    links_of_removed_node = links_removed_node(G, removed_node)
    # get role of removed node
    removed_node_role = G.nodes[removed_node]["role"]
    G.remove_node(removed_node)
    return removed_node_role,removed_node, links_of_removed_node


""" Recovery Algorithms """


def random_recovery(role_removed_node,removed_node,group, links_of_removed_node, G, p):
    
    role_macro_nodes = [x for x,y in G.nodes(data=True) if y['role']== role_removed_node]

    # Dictionary of all shortest paths
    sp = dict(nx.all_pairs_shortest_path(G))
    distances = []
    for link in links_of_removed_node:
        chosen_node = rd.choice(role_macro_nodes)
        
        if(nx.has_path(G, chosen_node, link)):
            #P that a link to the removed node is linked to the chosen node 
            if np.random.uniform() < p:
                    distances.append(len(sp[chosen_node][link]))
                    G.add_edge(chosen_node, link)
    if not distances:
        distances.append(0)
    
    return group,distances

#Degree recovery algorithm
def degree_recovery(role_removed_node,removed_node, group,links_of_removed_node, G, p):
    if not group:
        distances = []
        return group, distances
    
    distances = []
    if removed_node in group:
        group.remove(removed_node)

    # Dictionary of all shortest paths
    sp = dict(nx.all_pairs_shortest_path(G))
    # List of tuples of degrees of all nodes + node names
    group_degrees = G.degree(group)
    # List of degrees of all nodes
    weights = np.array([x[1] for x in group_degrees])

    for link in links_of_removed_node:
        # Pick node randomly based on degree
        probs = weights/np.sum(weights)
        index_largest_degree_prob = np.random.choice(np.arange(0,len(weights),1),1, p= probs)[0]
        chosen_node = list(group_degrees)[index_largest_degree_prob][0]
        #to avoid self loops and Does path exist
        if ((link != chosen_node) and (nx.has_path(G, chosen_node, link))):
        #P that a link to the removed node is linked to the chosen node 
            if np.random.uniform() < p:
                distances.append(len(sp[chosen_node][link]))
                G.add_edge(chosen_node, link)
    if not distances:
        distances.append(0)
    return group, distances

## Distance Recovery - Algorithm 2 : Breadth First Search
def distance_recovery(role_disrupted_node,removed_node,group,links_removed_node_list,G,prob_rewire,max_depth=20):
    
    replacement_list = []
    distance_list = []
    success_rewiring = []
    ## for each link look for a replacement at at shortest distance
    for node in links_removed_node_list:
        
        replaced = False
        #print("Linked node: ", node)
        for distance in range(1,max_depth+1):
            set_neighbours = nx.descendants_at_distance(G, node, distance)
            for neighbour in set_neighbours:
                #print(neighbour)
                #print(G.nodes[neighbour]['role'])
                if(G.nodes[neighbour]['role'] == role_disrupted_node):
                    replacement = neighbour
                    dist = distance
                    replaced = True
                    break
            if(replaced):
                distance_list.append(dist)
                replacement_list.append(replacement)
                random_num = np.random.uniform()
                # Accept it
                if(random_num < prob_rewire):
                    # add an edge 
                    G.add_edge(neighbour, node)     
                    success_rewiring.append(1)
                else:
                    success_rewiring.append(0)
                break
        
        if(replaced == False):
            distance_list.append(-1)
            replacement_list.append(-1)
            success_rewiring.append(0)
    
    return group,distance_list


# Simulates macro network
def simulate_macro_VC(n_nodes_removal, n_simulations, remove_strat, recover_strat):
    data_per_giant1_total = np.array([])
    total_efficiency1_total = np.array([])
    total_density1_total = np.array([])
    dist_total = []
    for i in range(n_simulations):
        distances = []
        print('Simulation', i+1)
        G_VC, df, roles_dict = initialize_network()
        G_macro, G = add_macro_basic(G_VC)

        VC_degree_attribute(G_VC)
        VC_degree_attribute_weighted(G_VC)

        data_per_giant1 = []
        total_efficiency1 = []
        total_density1 = []

        ## calculations for prob_rewire
        roles = nx.get_node_attributes(G, "role")
        count_roles = Counter(roles.values())

        for j in range(n_nodes_removal):
           
            print("Cycle ", j+1)
            #calculation of the percentage in giant component
            percentage_in_giant = giant_component_perc(G_VC)
            data_per_giant1.append(percentage_in_giant)

            #efficiency of the network  
            efficiency_cycle_perc = calc_efficiency_macro(G_VC, G)
            total_efficiency1.append(efficiency_cycle_perc)

            #density of the network
            density_cycle = nx.density(G_VC)
            total_density1.append(density_cycle)

            #to remove node with highest degree and recover the links to that node
            role_disrupted_node,node_most_links, links_of_removed_node = remove_strat(G_VC)

            # no more nodes of role_to_remove left in the network 
            if(count_roles[role_disrupted_node] == 0):
                break

            count_roles[role_disrupted_node] = count_roles[role_disrupted_node] - 1 

            prob_rewire = 1-1/(count_roles[role_disrupted_node]+1)
            #print(prob_rewire)

            #degree recovery algorithm/ random recovery
            roles_dict[role_disrupted_node], distance = recover_strat(role_disrupted_node,node_most_links , roles_dict[role_disrupted_node], links_of_removed_node, G, prob_rewire)
            distances.append(distance)

        dist_total.append(distances)
        data_per_giant1 = np.array(data_per_giant1)
        data_per_giant1_total = np.append(data_per_giant1_total,data_per_giant1)

        total_efficiency1 = np.array(total_efficiency1)
        total_efficiency1_total = np.append(total_efficiency1_total,total_efficiency1)

        total_density1 = np.array(total_density1)
        total_density1_total = np.append(total_density1_total,total_density1)
    
    return data_per_giant1_total, total_efficiency1_total, total_density1_total, dist_total

# dictionary with list object in values
def get_dict_sim_details(disrupt_strat,recovery_strat,efficiency,density,n_sim,n_nodes_removed,combined_network):
    details = {
        'disruption' : disrupt_strat,
        'recovery' : recovery_strat,
        'efficiency' : efficiency,
        'density': density,
        'n_sim': n_sim,
        'n_nodes_removed': n_nodes_removed,
        'combined_network': combined_network
    }
    return details

# Simulates value chain network
def simulate(n_nodes_removal, n_simulations, remove_strat, recover_strat):
    data_per_giant1_total = np.array([])
    total_efficiency1_total = np.array([])
    total_density1_total = np.array([])
    dist_total = []
    for i in range(n_simulations):
        distances = []
        print('Simulation', i+1)
        G, df, roles_dict = initialize_network()

        VC_degree_attribute(G)
        VC_degree_attribute_weighted(G)

        data_per_giant1 = []
        total_efficiency1 = []
        total_density1 = []

        ## calculations for prob_rewire
        roles = nx.get_node_attributes(G, "role")
        count_roles = Counter(roles.values())

        for j in range(n_nodes_removal):
            print("Cycle ", j+1)
            #calculation of the percentage in giant component
            percentage_in_giant = giant_component_perc(G)
            data_per_giant1.append(percentage_in_giant)

            #efficiency of the network  
            efficiency_cycle_perc = calc_efficiency(G)
            total_efficiency1.append(efficiency_cycle_perc)

            #density of the network
            density_cycle = nx.density(G)
            total_density1.append(density_cycle)

            #to remove node with highest degree and recover the links to that node
            role_disrupted_node,node_most_links, links_of_removed_node = remove_strat(G)

            # no more nodes of role_to_remove left in the network 
            if(count_roles[role_disrupted_node] == 0):
                break

            count_roles[role_disrupted_node] = count_roles[role_disrupted_node] - 1 

            prob_rewire = 1-1/(count_roles[role_disrupted_node]+1)

            #degree recovery algorithm/ random recovery
            roles_dict[role_disrupted_node], distance = recover_strat(role_disrupted_node,node_most_links , roles_dict[role_disrupted_node], links_of_removed_node, G, prob_rewire)
            distances.append(distance)

        dist_total.append(distances)
        data_per_giant1 = np.array(data_per_giant1)
        data_per_giant1_total = np.append(data_per_giant1_total,data_per_giant1)

        total_efficiency1 = np.array(total_efficiency1)
        total_efficiency1_total = np.append(total_efficiency1_total,total_efficiency1)

        total_density1 = np.array(total_density1)
        total_density1_total = np.append(total_density1_total,total_density1)
    
    return data_per_giant1_total, total_efficiency1_total, total_density1_total, dist_total
