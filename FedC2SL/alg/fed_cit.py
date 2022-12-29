import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import sys
import os
import argparse
import numpy as np
from scipy.special import gamma
from scipy.stats import levy_stable

from scipy.stats.distributions import chi2


from utils.utils  import get_fl_data

CONST_BINCOUNT_UNIQUE_THRESHOLD = 1e5
fisherz = "fisherz"
mv_fisherz = "mv_fisherz"
mc_fisherz = "mc_fisherz"
kci = "kci"
chisq = "chisq"
gsq = "gsq"

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

def read_data(file_path='data/data.csv'):
    """读数据并用数字表示特征，方便后续处理"""
    data = pd.read_csv(file_path)
    feature_name = data.columns
    data = OrdinalEncoder().fit_transform(data)
    per_feature_num = [int(data[:, i].max() + 1) for i in range(len(feature_name))]
    data = pd.DataFrame(columns=feature_name, data=data)
    return data, feature_name, per_feature_num





def parse_args():
    '''绝大部分参数没用，只是为了方便使用fed_chi_square这个函数而保留该函数，只使用了row/col两个参数'''
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='synthesize')
    parser.add_argument('--correlation', default='independent')
    parser.add_argument('--test', default='chi2')
    parser.add_argument('--nworker', type=int, default=10)
    parser.add_argument('--row', type=int, default=20)
    parser.add_argument('--col', type=int, default=20)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--iters', type=int, default=10)
    # implement parallel using pool to accelerate the process
    parser.add_argument('--para_level', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0)
    args = parser.parse_args()
    return args


"""分割线，到下个分割线之前都是Pang Qi学长的代码，未做改动"""


def geometric_mean(alpha, sketch_size, x):
    return np.prod(np.power(np.abs(x), alpha / sketch_size)) / np.power(
        2 * gamma(alpha / sketch_size) * gamma(1 - 1 / sketch_size) * np.sin(np.pi * alpha / 2 / sketch_size) / np.pi,
        sketch_size)



def get_gt(lts):
    """获得全局列联表"""
    gt = np.zeros_like(lts[0])
    for lt in lts:
        gt += lt
    return gt


class CIT(object):
    def __init__(self, data, alpha=0.05, client_num=10, method='chisq',is_iid=True, **kwargs):
        '''
        Parameters
        ----------
        data: numpy.ndarray of shape (n_samples, n_features)
        method: str, in ["fisherz", "mv_fisherz", "mc_fisherz", "kci", "chisq", "gsq"]
        kwargs: placeholder for future arguments, or for KCI specific arguments now
        '''
        self.data = data
        self.is_iid=is_iid
        self.fl_data = None
        self.data_hash = hash(str(data))
        self.sample_size, self.num_features = data.shape
        self.method = method
        self.pvalue_cache = {}
        self.client_num = client_num
        self.args = parse_args()
        self.args.nworker = client_num
        self.alpha = alpha

        if method in ['chisq', 'gsq']:
            def _unique(column):
                return np.unique(column, return_inverse=True)[1]

            self.data = np.apply_along_axis(_unique, 0, self.data).astype(np.int64)
            self.data_hash = hash(str(self.data))
            self.cardinalities = np.max(self.data, axis=0) + 1
            self.fl_data = get_fl_data(self.data, self.client_num,is_iid=is_iid)
        else:
            raise NotImplementedError(f"CITest method {method} is not implemented.")

        self.named_caller = {
            'chisq': self.chisq
        }

    def chisq(self, X, Y, condition_set):
        indexs = list(condition_set) + [X, Y]
        return self._chisq_or_gsq_test(X, Y, condition_set)


    def _chisq_or_gsq_test(self, X, Y, condition_set):
        """by Haoyue@12/18/2021
        Parameters
        ----------
        dataSXY: numpy.ndarray, in shape (|S|+2, n), where |S| is size of conditioning set (can be 0), n is sample size
                 dataSXY.dtype = np.int64, and each row has values [0, 1, 2, ..., card_of_this_row-1]
        cardSXY: cardinalities of each row (each variable)
        G_sq: True if use G-sq, otherwise (False by default), use Chi_sq
        """

        def _Fill2DCountTable(dataXY, cardXY):
            cardX, cardY = cardXY
            xyIndexed = dataXY[0] * cardY + dataXY[1]
            xyJointCounts = np.bincount(xyIndexed, minlength=cardX * cardY).reshape(cardXY)
            return xyJointCounts

        def _Fill3DCountTableByBincount(dataSXY, cardSXY):
            cardX, cardY = cardSXY[-2:]
            cardS = np.prod(cardSXY[:-2])
            cardCumProd = np.ones_like(cardSXY)
            cardCumProd[:-1] = np.cumprod(cardSXY[1:][::-1])[::-1]
            SxyIndexed = np.dot(cardCumProd[None], dataSXY)[0]

            SxyJointCounts = np.bincount(SxyIndexed, minlength=cardS * cardX * cardY).reshape((cardS, cardX, cardY))
            return SxyJointCounts

        def _Fill3DCountTable(dataSXY, cardSXY):
            return _Fill3DCountTableByBincount(dataSXY, cardSXY)

        def _SecureSum(data_ls,idx_ls):
            shape_of_sum = data_ls[0].shape
            data_len = np.prod(shape_of_sum)
            rand_int_mask = np.random.choice([-1,1],data_len)
            rand_float_mask = np.random.rand(data_len)
            rand_mask = rand_int_mask * rand_float_mask * 0xffffffff
            result = rand_mask + data_ls[idx_ls[0]].flatten()
            for i in range(1, len(idx_ls)):
                result += data_ls[idx_ls[i]].flatten()
            result -= rand_mask            
            return result.reshape(shape_of_sum)

        def _GetVxVyN(lts,involved_clients=None):
            if involved_clients==None:
                involved_clients=[i for i in range(len(lts))]
            Vxy_ls = [np.sum(SxyJointCounts, axis=(1, 2)) for SxyJointCounts in lts]
            Vx_ls = [np.sum(SxyJointCounts, axis=2) for SxyJointCounts in lts]
            Vy_ls = [np.sum(SxyJointCounts, axis=1) for SxyJointCounts in lts]
            with HiddenPrints():
                Vxy = _SecureSum(Vxy_ls,involved_clients)
                Vx = _SecureSum(Vx_ls,involved_clients)
                Vy = _SecureSum(Vy_ls,involved_clients)
            return Vx,Vy,Vxy#,(SxJointCounts,SyJointCounts,SMarginalCounts,SxyExpectedCounts)

        def _GetExpectedtable(Vx,Vy,Vxy):
            
            SMarginalCounts = Vxy.astype(int)
            SxJointCounts = Vx.astype(int)
            SyJointCounts = Vy.astype(int)

            SMarginalCounts_inds = SMarginalCounts == 0
            SMarginalCounts[SMarginalCounts_inds] = 1
            SxyExpectedCounts = SxJointCounts[:, :, None] * SyJointCounts[:, None, :] / SMarginalCounts[:, None, None]
            return SxyExpectedCounts

        def _CalculatePValue(lts, eTables, drop_clients=None):
            eTables_zero_inds = eTables == 0
            eTables_zero_to_one = np.copy(eTables)
            eTables_zero_to_one[eTables_zero_inds] = 1  # for legal division
            # array in shape (k,), zero_counts_rows[w]=c (0<=c<m) means layer w has c all-zero rows
            zero_counts_rows = eTables_zero_inds.all(axis=2).sum(axis=1)
            zero_counts_cols = eTables_zero_inds.all(axis=1).sum(axis=1)
            dof_ls = (eTables.shape[1] - 1 - zero_counts_rows) * (eTables.shape[2] - 1 - zero_counts_cols)
            # dof_ls = (cTables.shape[1] - 1) * (cTables.shape[2] - 1)

            sum_of_df = np.sum(dof_ls)

            rv = levy_stable(2.0, 0.0)
            proj_matrix = rv.rvs(size=[self.args.samples, eTables.shape[1]*eTables.shape[2]])
            sum_of_chi_square=0

            for j in range(eTables.shape[0]):
                #samples = np.zeros(self.args.samples)
                sample_ls = []
                for i in range(self.args.nworker):
                    if self.args.dropout > 0 and i in drop_clients:
                        continue
                    # noise_matrix = np.random.normal(size=lts[i][j].shape)  #.astype(int)
                    ob = lts[i][j]  # + noise_matrix
                    inter = np.divide(ob - eTables[j] / self.args.nworker, np.sqrt(eTables_zero_to_one[j]))
                    sample_ls.append(np.matmul(proj_matrix, inter.flatten()))
                samples = _SecureSum(sample_ls,[i for i in range(len(lts))])
                sum_of_chi_square += geometric_mean(2.0, self.args.samples, samples)

            return 1 if sum_of_df == 0 else chi2.sf(sum_of_chi_square, sum_of_df)

        indexs = list(condition_set) + [X, Y]
        lts = []
        cardXY = self.cardinalities[indexs]
        if len(indexs) == 2:  # S is empty
            for i in range(self.client_num):
                xyJointCounts = _Fill2DCountTable(self.fl_data[i][:, indexs].T, cardXY)
                lts.append(xyJointCounts[None])

        # else, S is not empty: conditioning
        else:
            for i in range(self.client_num):
                SxyJointCounts = _Fill3DCountTable(self.fl_data[i][:, indexs].T, cardXY)
                lts.append(SxyJointCounts)
        if self.args.dropout > 0:
            drop_clients = list(np.random.choice(np.arange(self.args.nworker), int(self.args.dropout * self.args.nworker)))  
        else:
            drop_clients = []
        involved_clients = list(set([i for i in range(len(lts))])-set(drop_clients))
        v_x,v_y,v_xy = _GetVxVyN(lts,involved_clients=involved_clients)
        gt_ex=_GetExpectedtable(v_x,v_y,v_xy)

        pval = _CalculatePValue(lts, gt_ex, drop_clients=drop_clients)
        return pval


    def __call__(self, X, Y, condition_set=None, *args):
        if self.method != 'mc_fisherz':
            assert len(args) == 0, "Arguments more than X, Y, and condition_set are provided."
        else:
            assert len(args) == 2, "Arguments other than skel and prt_m are provided for mc_fisherz."
        if condition_set is None: condition_set = tuple()
        assert X not in condition_set and Y not in condition_set, "X, Y cannot be in condition_set."
        i, j = (X, Y) if (X < Y) else (Y, X)
        cache_key = (i, j, frozenset(condition_set))

        if self.method != 'mc_fisherz' and cache_key in self.pvalue_cache: return self.pvalue_cache[cache_key]
        pValue = self.named_caller[self.method](X, Y, condition_set)
        self.pvalue_cache[cache_key] = pValue
        return pValue