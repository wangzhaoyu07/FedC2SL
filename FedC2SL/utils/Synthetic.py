#!/usr/bin/env python
#-*- coding: utf-8 -*-
import sys, os, random
import pandas as pd
sys.path.append(os.path.dirname(os.getcwd()))
import igraph as ig
import numpy as np
from scipy.stats import truncnorm
from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edges import Edges
from utils.DAG2PAG import dag2pag
from utils.logger import logging

def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def simulate_dag(d, s0, graph_type, is_iid=True):
    """Simulate random DAG with some expected number of edges.
       partly stolen from https://github.com/xunzheng/notears
    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF
        restrict_indegree (bool): set True to restrict nodes' indegree, specifically,
            for ER: from skeleton to DAG, randomly acyclic orient each edge.
                    if a node's degree (in+out) is large, we expect more of its degree is allocated for out, less for in.
                    so permute: the larger degree, the righter in adjmat, and
                                after the lower triangle, the lower upper bound of in-degree
            for SF: w.r.t. SF natrue that in-degree may be exponentially large, but out-degree is always low,
                    explicitly set the MAXTOLIND. transpose in/out when exceeding. refer to _transpose_in_out(B)
    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _acyclic_orientation(B_und):
        # pre-randomized here. to prevent i->j always with i>j
        return np.tril(B_und, k=-1)

    def _remove_isolating_node(B):
        non_iso_index = np.logical_or(B.any(axis=0), B.any(axis=1))
        return B[non_iso_index][:, non_iso_index]

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
    elif graph_type == 'SF':
        G_und = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=False, outpref=True, power=-3)
    else:
        raise ValueError('unknown graph type')

    B_und = _graph_to_adjmat(G_und)
    B_und = _random_permutation(B_und)
    B = _acyclic_orientation(B_und)
    B = _remove_isolating_node(B)
    B_perm = _random_permutation(B).astype(int)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    if not is_iid:
        B_perm = np.pad(B_perm,((0, 1), (0, 1)),'constant', constant_values=0)
        #B_perm[[0,1,2,3],-1]=1   
        B_perm[:5,-1]=1       
    return B_perm

def simulate_cards(B, card_param=None,is_iid=True,client_num=None):
    if card_param == None:
        card_param = {'lower': 2, 'upper': 6, 'mu': 3.5, 'basesigma': 1.5} # truncated normal distribution
    def _max_peers():
        '''
        why we need this: to calculate cpd of a node with k parents,
            the conditions to be enumerated is the production of these k
            parents' cardinalities which will be very exponentially slow w.r.t.
            k. so we want that, if a node has many parents (large k), these
            parents' cardinalities should be small
        i also tried to restrict each node's indegree at the graph sampling
        step,
            but i think that selection-bias on graph structure is worse than
            that on cardinalities
        an alternative you can try:
            use SEM to escape from slow forwards simulation, and then
            discretize.

        denote peers_num: peers_num[i, j] = k (where k>0),
            means that there are k parents pointing to node i, and j is among
            these k parents.
        max_peers = peers_num.max(axis=0): the larger max_peers[j], the smaller
        card[j] should be. :return:
        '''
        in_degrees = B.sum(axis=0)
        peers_num = in_degrees[:, None] * B.T
        return peers_num.max(axis=0)

    lower, upper, mu, basesigma = card_param['lower'], card_param['upper'], card_param['mu'], card_param['basesigma']
    sigma = basesigma / np.exp2(_max_peers()) ########## simply _max_peers() !
    cards = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).\
        rvs(size=B.shape[0]).round().astype(int)
    if not is_iid:
        cards[-1]=client_num
    return cards
    
def simulate_discrete(B, n, card_param=None, uniform_alpha_param=None, return_bn=False, is_iid=True,client_num=4):
    # try:
    if uniform_alpha_param == None:
        uniform_alpha_param = {'lower': 0.1, 'upper': 1.0}
    def _random_alpha():
        return np.random.uniform(uniform_alpha_param['lower'], uniform_alpha_param['upper'])
    cards = simulate_cards(B, card_param=card_param,is_iid=is_iid,client_num=client_num)

    diEdges = list(map(lambda x: (str(x[0]), str(x[1])), np.argwhere(B == 1))) # list of tuples
    bn = BayesianModel(diEdges) # so isolating nodes will echo error
    fd_num = 0#len(cards) * 0.15
    fd_edges = []
    for node in range(len(cards)):
        parents = np.where(B[:, node] == 1)[0].tolist()
        parents_card = [cards[prt] for prt in parents]

        if len(parents) == 1 and fd_num > 0:
            rand_ps = []
            for _ in range(int(np.prod(parents_card))):
                dist = np.zeros(cards[node])
                dist[np.random.randint(cards[node])] = 1.
                rand_ps.append(dist)
            rand_ps = np.array(rand_ps).T.tolist()
            fd_edges.append((parents[0], node))
            fd_num -= 1
        else:
            rand_ps = np.array([np.random.dirichlet(np.ones(cards[node]) * _random_alpha()) for _ in
                                range(int(np.prod(parents_card)))]).T.tolist()
        cpd = TabularCPD(str(node), cards[node], rand_ps,
                        evidence=list(map(str, parents)), evidence_card=parents_card)
        cpd.normalize()
        bn.add_cpds(cpd)
    inference = BayesianModelSampling(bn)
    df = inference.forward_sample(size=n, show_progress = False)
    df = df[[str(i) for i in range(len(cards))]]
    return df.to_numpy().astype(np.int), fd_edges

def simulate_dag_and_data(node_num:int, edge_num:int, graph_type:str, sample_num:int=10000, is_iid:bool=True, client_num = 4):
    logging.info("GENERATE SYNTHETIC DAG")
    B = simulate_dag(node_num, edge_num, graph_type, is_iid)
    logging.info("FORWARD SAMPLING")
    data, fd_edges = simulate_discrete(B, n=sample_num,client_num=client_num,is_iid=is_iid)
    nodes = []
    if is_iid:
        node_num = B.shape[0]
    else:
        node_num = B.shape[0]-1
    for i in range(node_num):
        node = GraphNode(f"X{i + 1}")
        node.add_attribute("id", i)
        node.add_attribute("col_name", f"col_{i}")
        nodes.append(node)
    dag = GeneralGraph(nodes)
    for i in range(node_num):
        for j in range(node_num):
            if B[i,j] == 1: dag.add_edge(Edges().directed_edge(nodes[i], nodes[j]))
    return dag, data

def simulate_pag_and_data(node_num:int, edge_num:int, graph_type:str,is_iid=True):
    logging.info("GENERATE SYNTHETIC DAG")
    B = simulate_dag(node_num, edge_num, graph_type,is_iid=is_iid)
    logging.info("FORWARD SAMPLING")
    data, fd_edges = simulate_discrete(B, n=10000,is_iid=is_iid)
    nodes = []
    if is_iid:
        node_num = B.shape[0]
    else:
        node_num = B.shape[0]-1
    for i in range(node_num):
        node = GraphNode(f"X{i + 1}")
        node.add_attribute("id", i)
        node.add_attribute("col_name", f"col_{i}")
        nodes.append(node)
    dag = GeneralGraph(nodes)
    
    for i in range(node_num):
        for j in range(node_num):
            if B[i,j] == 1: dag.add_edge(Edges().directed_edge(nodes[i], nodes[j]))
    logging.info("GENERATE PAG")

    pag = dag2pag(dag, random.choices(nodes, k=int(node_num * 0.05)))

    pag_fd_edge = []
    for fd_edge in fd_edges:
        x, y = fd_edge
        node_x, node_y = None, None
        for node in pag.get_nodes():
            node:GraphNode
            if node.get_attribute("id") == x: node_x = node
            if node.get_attribute("id") == y: node_y = node
        if node_x is not None and node_y is not None:
            pag_fd_edge.append((node_x.get_attribute("col_name"), node_y.get_attribute("col_name")))
    #         if pag.is_adjacent_to(node_x, node_y):
    #             edge = pag.get_edge(node_x, node_y)
    #             pag.remove_edge(edge)
    #         pag.add_edge(Edges().directed_edge(node_x, node_y))
    # pag = dag2pag(dag, [])
    observed_data = {}
    for node in pag.get_nodes():
        node: GraphNode
        node_id = node.get_attribute("id")
        node_name = node.get_attribute("col_name")
        observed_data[node_name] = data[:, node_id]
    if not is_iid:
        observed_data['col_last'] = data[:, -1]
    df = pd.DataFrame(observed_data)
    return pag, df


