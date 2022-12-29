from http import client
import os, time
import sys
sys.path.append("")
sys.path.append("../..")
from alg.FedPC import pc, pc_voting
from causallearn.search.ConstraintBased import PC
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz
from causallearn.graph.SHD import SHD
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from causallearn.search.ScoreBased.GES import ges
import alg.fed_GES
from utils.utils import get_fl_data
import pickle
from utils.Synthetic import *

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
   
def test_on_non_iid_data(node_num_ls=[20],client_num_ls=[4],round_num=10):
    print('Now start test_pc_load_synthetic_data_on_different_client_num ...')
    log_dir='log/non_iid/'
    log_name = [['fed_pc_log_iid.txt','fed_pc_non_iid.txt'],['ges_rfcd_iid.txt','ges_rfcd_non_iid.txt'],['pc_voting_iid.txt','pc_voting_non_iid.txt'],['pc_cit_voting_iid.txt','pc_cit_voting_non_iid.txt']]
    for node_num in node_num_ls:
    #for node_num in [50]:
        for myround in range(round_num):
            for client_num in client_num_ls:
                print(f'round:{myround}')
                data = pd.read_csv(f"data/synthetic/non_iid/non_iid_data_node_num_{node_num}_client_num_{client_num}_round_{myround}.csv").values
                truth_dag=pickle.load(open(f"data/synthetic/non_iid/non_iid_data_node_num_{node_num}_client_num_{client_num}_round_{myround}.pkl",'rb'))
                truth_cpdag = dag2cpdag(truth_dag)
                num_nodes_in_truth = truth_dag.get_num_nodes()
                # fed pc
                # iid
                start_time = time.time()
                cg = pc(data[:,:-1], 0.05, 'fed_cit', True, 0, -1, node_names=None, client_num=client_num,is_iid = True)
                end_time = time.time()
                shd = SHD(truth_cpdag, cg.G)
                with open(log_dir+log_name[0][0],'a') as f:
                    print(
                         f'node_num,{node_num},cleint_num,{client_num},round,{myround},time,{end_time-start_time},SHD,{shd.get_shd()}',
                    file = f)
                # non iid
                start_time = time.time()
                cg = pc(data, 0.05, 'fed_cit', True, 0, -1, node_names=None, client_num=client_num,is_iid = False)
                end_time = time.time()
                shd = SHD(truth_cpdag, cg.G)
                with open(log_dir+log_name[0][1],'a') as f:
                    print(
                         f'node_num,{node_num},cleint_num,{client_num},round,{myround},time,{end_time-start_time},SHD,{shd.get_shd()}',
                    file = f)
                # rfcd
                # iid
                start_time = time.time()
                with HiddenPrints():
                    cg = alg.fed_GES.ges(data[:,:-1], score_func='local_score_BDeu', maxP=None, parameters=None,client_num=client_num,is_iid=True)
                end_time = time.time()
                shd = SHD(truth_cpdag, cg['G'])
                with open(log_dir+log_name[1][0],'a') as f:
                    print(
                         f'node_num,{node_num},cleint_num,{client_num},round,{myround},time,{end_time-start_time},SHD,{shd.get_shd()}',
                    file = f)
                # non iid
                start_time = time.time()
                with HiddenPrints():
                    cg = alg.fed_GES.ges(data, score_func='local_score_BDeu', maxP=None, parameters=None,client_num=client_num,is_iid=False)
                end_time = time.time()
                shd = SHD(truth_cpdag, cg['G'])
                with open(log_dir+log_name[1][1],'a') as f:
                    print(
                         f'node_num,{node_num},cleint_num,{client_num},round,{myround},time,{end_time-start_time},SHD,{shd.get_shd()}',
                    file = f)
                # pc voting
                # iid
                fl_data = get_fl_data(data[:,:-1],client_num,is_iid=True)
                start_time = time.time()
                final_graph = pc_voting(fl_data,client_num,truth_cpdag,num_nodes_in_truth)
                end_time = time.time()
                shd = SHD(truth_cpdag, final_graph)
                with open(log_dir+log_name[2][0],'a') as f:
                    print(
                         f'node_num,{node_num},cleint_num,{client_num},round,{myround},time,{end_time-start_time},SHD,{shd.get_shd()}',
                    file = f)
                # non iid
                fl_data = get_fl_data(data,client_num,is_iid=False)
                start_time = time.time()
                final_graph = pc_voting(fl_data,client_num,truth_cpdag,num_nodes_in_truth)
                end_time = time.time()
                shd = SHD(truth_cpdag, final_graph)
                with open(log_dir+log_name[2][1],'a') as f:
                    print(
                         f'node_num,{node_num},cleint_num,{client_num},round,{myround},time,{end_time-start_time},SHD,{shd.get_shd()}',
                    file = f)
                # pc cit with voting
                # iid
                start_time = time.time()
                cg = pc(data, 0.05, 'cit_with_voting', True, 0, -1, node_names=None, client_num=client_num,is_iid= True)
                end_time = time.time()
                shd = SHD(truth_cpdag, cg.G)
                with open(log_dir+log_name[3][0],'a') as f:
                    print(
                         f'node_num,{node_num},cleint_num,{client_num},round,{myround},time,{end_time-start_time},SHD,{shd.get_shd()}',
                    file = f)
                # non iid
                start_time = time.time()
                cg = pc(data, 0.05, 'cit_with_voting', True, 0, -1, node_names=None, client_num=client_num,is_iid= False)
                end_time = time.time()
                shd = SHD(truth_cpdag, cg.G)
                with open(log_dir+log_name[3][1],'a') as f:
                    print(
                         f'node_num,{node_num},cleint_num,{client_num},round,{myround},time,{end_time-start_time},SHD,{shd.get_shd()}',
                    file = f)

if __name__ == '__main__':

    test_on_non_iid_data(node_num_ls=[20],round_num=10)
