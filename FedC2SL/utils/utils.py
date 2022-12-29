import numpy as np
import os
import sys
sys.path.append("")
sys.path.append("../..")
import numpy as np
import pandas as pd
import pickle

from utils.Synthetic import simulate_dag_and_data,simulate_pag_and_data

def read_csv(filename):
    with open(filename) as f:
        headers = [h for h in f.readline().strip().split(',')]
        discrete2num = {h:{} for h in headers}
        discrete2idx = {h:0 for h in headers}
        frequencies = {h:{} for h in headers}
        lines = f.readlines()
        X = []
        
        for line in lines:
            point = line.strip().split(',')
            num_point = []
            for idx, val in enumerate(point):
                col = headers[idx]
                if val not in discrete2num[col]:
                    mapping = np.ceil(discrete2idx[col] / 2)
                    if discrete2idx[col] % 2 == 0:
                        mapping *= -1
                    discrete2num[col][val] = discrete2idx[col]
                    discrete2idx[col] += 1
                num_val = discrete2num[col][val]
                if num_val not in frequencies[col]:
                    frequencies[col][num_val] = 1
                else:
                    frequencies[col][num_val] += 1
                num_point.append(num_val)
            X.append(num_point)
        X = np.array(X, dtype=np.float)
    
    X = discrete2continuous(X, headers, frequencies) * 20
    X = X - np.mean(X, axis=0, keepdims=True)
    return X, headers

def discrete2continuous(X, headers, frequencies):
    data_size = X.shape[0]
    models = {h:create_model(frequencies[h]) for h in headers}
    for row_idx in range(data_size):
        for col_idx, h in enumerate(headers):
            val = models[h](X[row_idx][col_idx])
            X[row_idx][col_idx] = val
    return X

def normal(x, mu, sig):
    return 1. / (np.sqrt(2 * np.pi) * sig) * np.exp(-0.5 * np.square(x - mu) / np.square(sig))


def trunc_normal(x, mu, sig, bounds=None):
    if bounds is None: 
        bounds = (-np.inf, np.inf)

    norm = normal(x, mu, sig)
    norm[x < bounds[0]] = 0
    norm[x > bounds[1]] = 0

    return norm


def sample_trunc(n, mu, sig, bounds=None):
    """ Sample `n` points from truncated normal distribution """
    x = np.linspace(mu - 5. * sig, mu + 5. * sig, 10000)
    y = trunc_normal(x, mu, sig, bounds)
    y_cum = np.cumsum(y) / y.sum()

    yrand = np.random.rand(n)
    sample = np.interp(yrand, y_cum, x)
    return sample

def create_model(freq):
    total = np.sum([freq[val] for val in freq])
    prob_dist = [(freq[i]/total, i) for i in range(len(freq))]
    prob_dist.sort(key=lambda x:-x[0]) # desceding order
    curr = 0
    left = {}
    right = {}
    for prob in prob_dist:
        left[prob[1]] = curr
        curr += prob[0]
        right[prob[1]] = curr
    def sample(val):
        mu = (right[val] - left[val]) / 2 + left[val]
        return mu
        sigma = (right[prob[1]] - left[prob[1]]) / 6
        return sample_trunc(1, mu, sigma, [left[val], right[val]])[0]
    return sample




def get_fl_data(original_data, client_num=10,is_iid=True):
    """把数据切割为联邦场景下的数据集"""
    if is_iid == True:
        permuted_indices = np.random.permutation(len(original_data))
        fl_data = []
        for i in range(client_num):
            fl_data.append(original_data[permuted_indices[i::client_num]])
    else:
        fl_data = [original_data[np.where(original_data[:,-1]==i)][:,:-1] for i in range(client_num)]
    
    return fl_data



def produce_iid_data():
    for node_num in [10,20,50,100]:
        for myround in range(30):
            truth_dag,data = simulate_dag_and_data(node_num=node_num,edge_num=node_num,graph_type='ER')
            df = pd.DataFrame(data,columns=[f"X{i+1}" for i in range(data.shape[1])])
            df.to_csv(f"data/synthetic/iid/synthetic_data_node_num_{node_num}_round_{myround}.csv",index = False)
            pickle.dump(truth_dag,open(f"data/synthetic/iid/synthetic_data_node_num_{node_num}_round_{myround}.pkl",'wb'))

def produce_iid_fci_data():
    for node_num in [10,20,50,100]:
    #for node_num in [100]:
        for myround in range(10):
            truth_pag,df = simulate_pag_and_data(node_num=node_num,edge_num=node_num,graph_type='ER')
            df.to_csv(f"data/synthetic/iid/fci/fci_synthetic_data_node_num_{node_num}_round_{myround}.csv",index = False)
            pickle.dump(truth_pag,open(f"data/synthetic/iid/fci/fci_synthetic_data_node_num_{node_num}_round_{myround}.pkl",'wb'))

def produce_non_iid_data():
    for node_num in [10,20,50,100]:
        for client_num in [4]:
            for myround in range(10):
                truth_dag,data = simulate_dag_and_data(node_num=node_num,edge_num=node_num,graph_type='ER',is_iid=False)
                df = pd.DataFrame(data,columns=[f"X{i+1}" for i in range(data.shape[1])])
                df.to_csv(f"data/synthetic/non_iid/non_iid_data_node_num_{node_num}_client_num_{client_num}_round_{myround}.csv",index = False)
                pickle.dump(truth_dag,open(f"data/synthetic/non_iid/non_iid_data_node_num_{node_num}_client_num_{client_num}_round_{myround}.pkl",'wb'))

def produce_fci_non_iid_data():
    #for node_num in [10,20,50,100]:
    for node_num in [20]:
        for client_num in [4]:
            for myround in range(10):
                truth_pag,df = simulate_pag_and_data(node_num=node_num,edge_num=node_num,graph_type='ER',is_iid=False)
                df.to_csv(f"data/synthetic/non_iid/fci/non_iid_data_node_num_{node_num}_client_num_{client_num}_round_{myround}.csv",index = False)
                pickle.dump(truth_pag,open(f"data/synthetic/non_iid/fci/non_iid_data_node_num_{node_num}_client_num_{client_num}_round_{myround}.pkl",'wb'))



def read_data():
    for node_num in [10,20,50,100]:
        for myround in range(1):
            df = pd.read_csv(f"data/synthetic/synthetic_data_node_num_{node_num}_round_{myround}.csv").values
            truth_dag=pickle.load(open(f"data/synthetic/synthetic_data_node_num_{node_num}_round_{myround}.pkl",'rb'))

            print(df)

if __name__ == '__main__':
    #produce_iid_data()
    #produce_non_iid_data()
    #produce_fci_non_iid_data()
    #read_data()
    produce_iid_data()
    produce_iid_fci_data()
    produce_non_iid_data()
    produce_fci_non_iid_data()
