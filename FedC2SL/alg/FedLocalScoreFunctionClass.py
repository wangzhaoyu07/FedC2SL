import math
from functools import lru_cache
from typing import Any, Callable, Dict, List

import pandas as pd
from causallearn.score.LocalScoreFunction import (
    local_score_BDeu,
    local_score_BIC,
    # local_score_BIC_from_cov,
    local_score_cv_general,
    local_score_cv_multi,
    local_score_marginal_general,
    local_score_marginal_multi,
)
from causallearn.utils.ScoreUtils import *
from numpy import ndarray

def get_fl_data(original_data, client_num=10,is_iid=True):
    """把数据切割为联邦场景下的数据集"""
    if is_iid == True:
        permuted_indices = np.random.permutation(len(original_data))
        fl_data = []
        for i in range(client_num):
            fl_data.append(np.mat(original_data[permuted_indices[i::client_num]]))
    else:
        fl_data = [np.mat(original_data[np.where(original_data[:,-1]==i)][:,:-1]) for i in range(client_num)]
    
    return fl_data


class LocalScoreClass(object):
    def __init__(
        self,
        data: Any,
        local_score_fun: Callable[[Any, int, List[int], Any], float],
        parameters=None,
        client_num=10,
        is_iid=True,
        dropout=0
    ):
        self.data = np.array(data)
        self.client_num=client_num
        self.fl_data = get_fl_data(self.data, self.client_num, is_iid)
        self.local_score_fun = local_score_fun
        self.parameters = parameters
        self.score_cache = {}
        self.dropout=dropout

    def score(self, i: int, PAi: List[int]) -> float:
        if i not in self.score_cache:
            self.score_cache[i] = {}

        hash_key = tuple(sorted(PAi))

        if not self.score_cache[i].__contains__(hash_key):
            self.score_cache[i][hash_key] = self.fed_score_fun(i, PAi)

        return self.score_cache[i][hash_key]

    def fed_score_fun(self, i: int, PAi: List[int]) -> float:
        r = 0
        drop_clients = list(np.random.choice(np.arange(self.client_num), int((1-self.dropout) * self.client_num)))                         
        for j,data in enumerate(self.fl_data):
            if j in drop_clients:
                r1 = self.local_score_fun(data, i, PAi, self.parameters)
                if r1>r:
                    r=r1
        return r
