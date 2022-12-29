from __future__ import annotations
from distutils.log import error

import time
import warnings
from itertools import combinations, permutations
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from numpy import ndarray
import copy
import sys
sys.path.append('..')
sys.path.append('../..')

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
import alg.cit_voting as cit_voting
import alg.fed_cit as fed_cit
import alg.SkeletonDiscovery as SkeletonDiscovery
from causallearn.utils.PCUtils import Helper, Meek, UCSepset
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
    orient_by_background_knowledge
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.search.ConstraintBased import PC
import utils
from causallearn.utils.cit import chisq


def pc(
        data: ndarray,
        alpha=0.05,
        indep_test="chisq",
        stable: bool = True,
        uc_rule: int = 0,
        uc_priority: int = 2,
        mvpc: bool = False,
        correction_name: str = 'MV_Crtn_Fisher_Z',
        background_knowledge: BackgroundKnowledge | None = None,
        verbose: bool = False,
        show_progress: bool = True,
        node_names: List[str] | None = None,
        client_num=10,
        dropout=0,
        is_iid =True
):
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    return pc_alg(data=data, node_names=node_names, alpha=alpha, indep_test=indep_test, stable=stable, uc_rule=uc_rule,
                  uc_priority=uc_priority, background_knowledge=background_knowledge, verbose=verbose,
                  show_progress=show_progress,client_num=client_num,dropout=dropout,is_iid=is_iid)


def pc_alg(
        data: ndarray,
        node_names: List[str] | None,
        alpha: float,
        indep_test: str,
        stable: bool,
        uc_rule: int,
        uc_priority: int,
        background_knowledge: BackgroundKnowledge | None = None,
        verbose: bool = False,
        show_progress: bool = True,
        client_num = 10,
        dropout=0,
        is_iid = True
) -> CausalGraph:
    """
    Perform Peter-Clark (PC) algorithm for causal discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)
    alpha : float, desired significance level of independence tests (p_value) in (0, 1)
    indep_test : str, the name of the independence test being used
            ["fisherz", "chisq", "gsq", "kci"]
           - "fisherz": Fisher's Z conditional independence test
           - "chisq": Chi-squared conditional independence test
           - "gsq": G-squared conditional independence test
           - "kci": Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)
    uc_rule : how unshielded colliders are oriented
           0: run uc_sepset
           1: run maxP
           2: run definiteMaxP
    uc_priority : rule of resolving conflicts between unshielded colliders
           -1: whatever is default in uc_rule
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns
    -------
    cg : a CausalGraph object, where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicates  i --> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """

    start = time.time()
    if indep_test=='cit_with_voting':
        indep_test = cit_voting.CIT(data, alpha=alpha, client_num=client_num,dropout=dropout,is_iid=is_iid)
    elif indep_test=='fed_cit':
        indep_test = fed_cit.CIT(data, alpha=alpha, client_num=client_num,dropout=dropout,is_iid=is_iid)
    else:
        raise NotImplementedError
    if not is_iid:
        data = data[:,:-1]
    cg_1 = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test, stable,
                                                background_knowledge=background_knowledge, verbose=verbose,
                                                show_progress=show_progress, node_names=node_names)

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_1, background_knowledge)

    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = UCSepset.uc_sepset(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.uc_sepset(cg_1, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = UCSepset.maxp(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.maxp(cg_1, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha, background_knowledge=background_knowledge)
        cg_before = Meek.definite_meek(cg_2, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_before, background_knowledge=background_knowledge)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")
    end = time.time()

    cg.PC_elapsed = end - start

    return cg

def pc_voting(fl_data:list[ndarray],client_num:int,truth_cpdag:GeneralGraph,num_nodes_in_truth:int,dropout:float=0):
    cg_ls=[]
    final_graph = copy.deepcopy(truth_cpdag)
    for loc_data in fl_data:
        cg_benchmark = PC.pc(loc_data, 0.05, chisq, True, 0, -1)
        cg_ls.append(np.expand_dims(cg_benchmark.G.graph, axis=2)+1)
    final_graph_mat = np.concatenate(cg_ls, axis=2)
    for i in range(num_nodes_in_truth):
        for j in range(num_nodes_in_truth):
            drop_clients = list(np.random.choice(np.arange(client_num), int((1-dropout) * client_num)))  
            d = np.argmax(np.bincount(final_graph_mat[i][j][drop_clients]))
            final_graph.graph[i][j] = d
    final_graph.graph -= 1
    return final_graph
