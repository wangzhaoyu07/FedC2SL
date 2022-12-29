# coding=UTF-8
import os, time
import sys
sys.path.append("")
sys.path.append("../..")
import numpy as np
from alg.FedPC import pc, pc_voting
import alg.fed_GES as fed_GES
from causallearn.search.ConstraintBased import PC
from causallearn.utils.cit import chisq
from causallearn.graph.SHD import SHD
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
import alg
import utils.utils as fed_utils

from utils.Synthetic import *
import pickle
import torch



from alg.FedFCI import fci, fci_voting, fed_fci
import time


class HiddenPrints:
    def __init__(self, activated=True):
        # activated参数表示当前修饰类是否被激活
        self.activated = activated
        self.original_stdout = None

    def open(self):
        sys.stdout.close()
        sys.stdout = self.original_stdout

    def close(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        # 这里的os.devnull实际上就是Linux系统中的“/dev/null”
        # /dev/null会使得发送到此目标的所有数据无效化，就像“被删除”一样
        # 这里使用/dev/null对sys.stdout输出流进行重定向

    def __enter__(self):
        if self.activated:
            self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activated:
            self.open()


def test_fed_pc_on_synthetic_data(node_num_ls=[50],client_num_ls=[2,4,8,16,32,64],dropout_ls=[0],round_num=10,test_type='client_num'):
    """
    test fed_pc algorithm on synthetic data

    Parameters
    ----------
    node_num_ls : list, the number of nodes in the test.
    client_num_ls: list, the number of clients in federated learning.
    dropout_ls : list, the dropout rate of clients in federated learning.
    round_num : int, the number of experiment repetitions
    test_type : str, the type of the test experiment
           client_num: test on different client numbers
           dropout: test on different dropout rates
           node_num: test on different node numbers
    """
    log_dir=f'log/iid/{test_type}/'
    log_name = [f'pc_log_on_{test_type}.txt',f'fed_pc_log_on_{test_type}.txt',f'pc_cit_voting_log_on_{test_type}.txt',f'pc_voting_on_{test_type}.txt',f'ges_rfcd_log_on_{test_type}.txt']
    
    for node_num in node_num_ls:
        for myround in range(round_num):
            print(f'round:{myround}')
            data = pd.read_csv(f"data/synthetic/iid/synthetic_data_node_num_{node_num}_round_{myround}.csv").values
            truth_dag=pickle.load(open(f"data/synthetic/iid/synthetic_data_node_num_{node_num}_round_{myround}.pkl",'rb'))
            truth_cpdag = dag2cpdag(truth_dag)
            num_nodes_in_truth = truth_dag.get_num_nodes()
            #pc
            cg_benchmark = PC.pc(data, 0.05, chisq, True, 0, -1, node_names=None)
            shd_benchmark = SHD(truth_cpdag, cg_benchmark.G)
            with open(log_dir+log_name[0],'a') as f:
                print(
                    f'variable_num,{node_num},round,{myround},time,{cg_benchmark.PC_elapsed:.3f},SHD,{shd_benchmark.get_shd()}',
                    file = f)
            for client_num in client_num_ls:
                for dropout in dropout_ls:
                    print(f'dropout:{dropout},round:{myround}; client_num:{client_num}')
                    #fed pc
                    start_time = time.time()
                    cg = pc(data, 0.05, 'fed_cit', True, 0, -1, node_names=None, client_num=client_num,dropout=dropout)
                    end_time = time.time()
                    shd = SHD(truth_cpdag, cg.G)
                    with open(log_dir+log_name[1],'a') as f:
                        print(
                            f'dropout,{dropout},variable_num,{node_num},cleint_num,{client_num},round,{myround},time,{end_time-start_time},SHD,{shd.get_shd()}',
                        file = f)

                    #pc_with_cit_voting
                    start_time = time.time()
                    cg = pc(data, 0.05, 'cit_with_voting', True, 0, -1, node_names=None, client_num=client_num,dropout=dropout)
                    end_time = time.time()
                    shd = SHD(truth_cpdag, cg.G)
                    with open(log_dir+log_name[2],'a') as f:
                        print(
                            f'dropout,{dropout},variable_num,{node_num},cleint_num,{client_num},round,{myround},time,{end_time-start_time},SHD,{shd.get_shd()}',
                        file = f)
                    #pc voting
                    fl_data = fed_utils.get_fl_data(data,client_num)
                    start_time = time.time()
                    final_graph = pc_voting(fl_data,client_num,truth_cpdag,num_nodes_in_truth,dropout)
                    end_time = time.time()
                    shd = SHD(truth_cpdag, final_graph)
                    with open(log_dir+log_name[3],'a') as f:
                        print(
                            f'node_num,{node_num},client_num,{client_num},dropout,{dropout},round,{myround},time,{end_time-start_time},SHD,{shd.get_shd()}',
                        file = f)
                    #rfcd
                    start_time = time.time()
                    with HiddenPrints():
                        cg = fed_GES.ges(data, score_func='local_score_BDeu', maxP=None, parameters=None,client_num=client_num,dropout=dropout)
                    end_time = time.time()
                    shd = SHD(truth_cpdag, cg['G'])
                    with open(log_dir+log_name[4],'a') as f:
                        print(
                            f'dropout,{dropout},variable_num,{node_num},cleint_num,{client_num},round,{myround},time,{end_time-start_time},SHD,{shd.get_shd()}',
                        file = f)

def test_fed_fci_on_synthetic_data(node_num_ls=[10,20,50,100],client_num_ls=[10],dropout_ls=[0],round_num=10,test_type='node_num'):
    """
    test fed_fci algorithm on synthetic data

    Parameters
    ----------
    node_num_ls : list, the number of nodes in the test.
    client_num_ls: list, the number of clients in federated learning.
    dropout_ls : list, the dropout rate of clients in federated learning.
    round_num : int, the number of experiment repetitions
    test_type : str, the type of the test experiment
           client_num: test on different client numbers
           dropout: test on different dropout rates
           node_num: test on different node numbers
    """
    log_dir=f'log/iid/{test_type}/'
    log_name = [f'fci_log_on_{test_type}.txt',f'fed_fci_log_on_{test_type}.txt',f'fci_cit_voting_log_on_{test_type}.txt',f'fci_voting_on_{test_type}.txt']
    for node_num in node_num_ls:#
        for myround in range(round_num):
            for client_num in client_num_ls:
                df = pd.read_csv(f"data/synthetic/iid/fci/fci_synthetic_data_node_num_{node_num}_round_{myround}.csv")
                truth_pag=pickle.load(open(f"data/synthetic/iid/fci/fci_synthetic_data_node_num_{node_num}_round_{myround}.pkl",'rb'))
                data = df.values
                num_nodes_in_truth = data.shape[1]
                col_names=df.columns
                start_time = time.time()
                G, edges = fci(data, col_names=col_names,alpha=0.05)
                end_time = time.time()
                print('benchmark finish')
                shd = SHD(truth_pag, G)
                with open(log_dir+log_name[0],'a') as f:
                    print(
                        f'variable_num,{node_num},round,{myround},time,{end_time-start_time},SHD,{shd.get_shd()}',
                        file = f)
                for dropout in dropout_ls:
                    start_time = time.time()
                    G, edges = fed_fci(data, indep_test='fed_cit', col_names=col_names,alpha=0.05,client_num = client_num,dropout=dropout)
                    end_time = time.time()
                    shd = SHD(truth_pag, G)
                    with open(log_dir+log_name[1],'a') as f:
                        print(
                            f'dropout,{dropout},variable_num,{node_num},cleint_num,{client_num},round,{myround},time,{end_time-start_time},SHD,{shd.get_shd()}',
                            file = f)
                    
                    start_time = time.time()
                    G, edges = fed_fci(data, indep_test='cit_with_voting', col_names=col_names,alpha=0.05,client_num = client_num,dropout=dropout)
                    end_time = time.time()
                    shd = SHD(truth_pag, G)
                    with open(log_dir+log_name[2],'a') as f:
                        print(
                            f'dropout,{dropout},variable_num,{node_num},cleint_num,{client_num},round,{myround},time,{end_time-start_time},SHD,{shd.get_shd()}',
                            file = f)
                    with HiddenPrints():
                        start_time = time.time()
                        G = fci_voting(data=data,col_names = [f'col_{j}' for j in range(num_nodes_in_truth)],client_num=client_num,truth_cpdag=truth_pag,num_nodes_in_truth=num_nodes_in_truth,dropout=dropout)
                        end_time = time.time()
                        shd = SHD(truth_pag, G)
                        with open(log_dir+log_name[3],'a') as f:
                            print(
                                f'dropout,{dropout},node_num,{node_num},cleint_num,{client_num},round,{myround},time,{end_time-start_time},SHD,{shd.get_shd()}',
                            file = f)    

def test_on_real_data(client_num_ls=[2,4,8,16,32,64],round_num=10):

    print('Now start test_pc_load_bnlearn_discrete_datasets ...')
    print('Please check SHD with truth graph and time cost with https://github.com/cmu-phil/causal-learn/pull/6.')

    benchmark_names = [
        "discrete_sachs"#,"mydata3"
    ]
    bnlearn_data_dir = 'data/science-protein'
    bnlearn_truth_dag_graph_dir = 'data/science-protein'
    log_dir='log/real/'
    #log_name = ['pc_log_on_client_num.txt','fed_pc_log_on_client_num.txt','pc_cit_voting_log_on_client_num.txt','pc_voting_log_on_client_num.txt','notears_admm_log_on_client_num.txt']
    log_name = ['fed_pc_on_real.txt','fed_cit_voting_on_real.txt','pc_voting_on_real.txt','fed_rfcd_on_real.txt']
    for bname in benchmark_names:
        data = np.loadtxt(os.path.join(bnlearn_data_dir, f'{bname}.txt'), skiprows=1)
        truth_dag = txt2generalgraph(os.path.join(bnlearn_truth_dag_graph_dir, 'gt.graph.txt'))
        truth_cpdag = dag2cpdag(truth_dag)
        num_nodes_in_truth = truth_dag.get_num_nodes()
        for client_num in client_num_ls:
            for myround in range(round_num):
                #fed pc
                start_time = time.time()
                cg = pc(data, 0.05, 'fed_cit', True, 0, -1, node_names=None, client_num=client_num,is_iid = True)
                end_time = time.time()
                shd = SHD(truth_cpdag, cg.G)
                with open(log_dir+log_name[0],'a') as f:
                        print(
                            f'round,{myround},client_num,{client_num},time{end_time-start_time:.3f},SHD,{shd.get_shd()}',
                        file = f)
                # pc with cit voting
                start_time = time.time()
                cg = pc(data, 0.05, 'cit_with_voting', True, 0, -1, node_names=None, client_num=client_num,is_iid = True)
                end_time = time.time()
                shd = SHD(truth_cpdag, cg.G)
                with open(log_dir+log_name[1],'a') as f:
                        print(
                            f'round,{myround},client_num,{client_num},time{end_time-start_time:.3f},SHD,{shd.get_shd()}',
                        file = f)
                #pc voting
                fl_data = fed_utils.get_fl_data(data,client_num)
                start_time = time.time()
                final_graph=pc_voting(fl_data,client_num,truth_cpdag,num_nodes_in_truth)
                end_time = time.time()
                shd = SHD(truth_cpdag, final_graph)
                with open(log_dir+log_name[2],'a') as f:
                        print(
                            f'round,{myround},client_num,{client_num},time{end_time-start_time:.3f},SHD,{shd.get_shd()}',
                        file = f)
                #rfcd
                start_time = time.time()
                with HiddenPrints():
                    cg = alg.fed_GES.ges(data, score_func='local_score_BDeu', maxP=None, parameters=None,client_num=client_num)
                end_time = time.time()
                shd = SHD(truth_cpdag, cg['G'])

                with open(log_dir+log_name[3],'a') as f:
                    print(f'round,{myround},client_num,{client_num},time{end_time-start_time:.3f},SHD,{shd.get_shd()}',
                            file = f)
    
if __name__ == '__main__':
    # synthetic data
    test_fed_fci_on_synthetic_data(node_num_ls=[10,20,50,100],client_num_ls=[10],dropout_ls=[0],round_num=10,test_type='node_num')
    
    test_fed_pc_on_synthetic_data(node_num_ls=[10,20,50,100],client_num_ls=[10],dropout_ls=[0],round_num=10,test_type='node_num')
    #real data
    test_on_real_data(client_num_ls=[2,4,8,16,32,64],round_num=10)