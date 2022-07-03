# Complex-systems

## Group 3: Crime Hunters

### Description

Our project explores the resilience of criminal networks, specifically for 5 different types of disruption strategies where a node is removed from the network and for 3 types of recovery strategies where the network tries to replace the removed node with another having the same role. Our work closely follow the paper "The Relative Ineffectiveness of Criminal Network Disruption" by Duijn P. et al. 

The Jupyter notebook file makes use of functions imported from funcs.py and generates & simulates a criminal network which is disrupted at every time step, specifying certain disruption and recovery strategies, and plots how the effieceincy, density, and average distance for new connection changes over time (all explained more thoroughly below). 

The slides from our presentation can also be found as a pdf here.

### Functions

#### Network

The ***initialize_network*** function makes use of the two .csv files in /Data that were taken from Table S-3 and Table S-4 from the paper's Appendix. It tries to rebuilt the cannabis cultivation value chain network to match the data as much as possible. From Table S-4 it takes the number of nodes (N) and average degree (D) per role and build a barabasi albert graph (n = N, m = D/N) for every role, then takes the disjoint union of these graphs and for every pair of roles (graphs) in this network, it adds the number of edges specified in Table S-3. Returns the value chain network, called G_VC.

***add_macro_basic*** adds to G_VC a somewhat random outer network, using barabasi albert (n = 2000, m = 1). It loops over every node in G_VC and add some edges from it to the outer network. For the more connected roles Financing, Coordinato, and Growshop owner, it adds 1-10 edges, otherwise it adds 1-2 edges. Returns G_macro (only outer network) and G_combines (the new network with G_VC connected).

***add_macro_stats*** adds another version of the outer network, this time based on the network statistics in Appendix (didn't use this in simulations as it didn't produce expected results). It multiples the data from the two tables by a factor of 3 and uses the same process as in initialize_network to build an outer network. G_VC and this outer network are connected in a similar way as in add_macro_basic.

***network_stats prints*** and returns the number of nodes, number of edges, average degree, avgerage shortest path, diameter, and largest component of a given graph. If given a list of roles, it returns a dataframe with number of nodes, number of edges, and average degree for each role.

#### Measures and other useful functions

***get_nodes_degree*** returns a dictionary of all nodes of a graph and their degree (to use in removing node with highest degree).

***links_removed_node*** returns all the links of a removed node.

***giant_component_perc*** calculates the percentages of the nodes that belong to the giant component.

***calc_efficiency*** measure efficiency of the graph.

***calc_efficiency_macro*** measures efficiency of the macro graph, calculating the distances of the nodes in G_VC through the macro network.

***VC_degree_attribute*** adds VC degree as attribute to each node, calculated as the number of unique roles a node is connected to.

***VC_degree_attribute_weighted*** similar as VC_degree_attribute but assigns weights according to how high in the hierarchy the role is (Figure 3 in paper).

#### Disruption strategies

***removal_random*** removes a random node.

***removal_highest_degree*** removes node of highest degree.

***removal_highest_betweeness*** removes node with highest betweenness centrality.

***disrupt_VC_degree*** removes node at random from set of nodes with highest VC degree.

***disrupt_VC_degree_weighted*** removes node at random from set of nodes with highest weighted VC degree.

***disrupt_VC_role*** removes node at random from set of nodes with specific role (in paper 'Diverting electricity').

#### Recovery strategies

***random_recovery*** for every orphaned node, with probabilty of rewiring, finds possible replacements with removed node's roles, and chooses one of them randomly.

***degree_recovery*** for every orphaned node, with probabilty of rewiring, finds possible replacements with removed node's roles, and chooses one with highest degree.

***distance_recovery*** for every orphaned node, with probabilty of rewiring, finds possible replacements with removed node's roles, and chooses one with shortest distance.

#### Simulation

***simulate generates*** the G_VC graph then runs the specified disruption strategy n times, and at each time step with some probability of rewiring, uses the specified recovery algorithm. It calculates the percentage in giant component, efficiency and density at each time step.

***simulate_macro_VC*** generates the G_VC and macro graph then runs the specified disruption strategy n times, and at each time step with some probability of rewiring, uses the specified recovery algorithm. It calculates the percentage in giant component, efficiency (macro version) and density at each time step.
