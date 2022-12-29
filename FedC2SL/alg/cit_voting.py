import pandas as pd
import sys
from sklearn.preprocessing import OrdinalEncoder
sys.path.append('..')    # To import notears_admm from parent directory
sys.path.append('../..')
import argparse
import numpy as np
from scipy.special import gamma
from scipy.stats import levy_stable, chi2_contingency
from collections import Counter

from scipy.stats.distributions import chi2
import tqdm, copy, random
from utils.utils import get_fl_data


CONST_BINCOUNT_UNIQUE_THRESHOLD = 1e5
fisherz = "fisherz"
mv_fisherz = "mv_fisherz"
mc_fisherz = "mc_fisherz"
kci = "kci"
chisq = "chisq"
gsq = "gsq"


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


def fed_chi2_test(lts, gt_orig, args):
    if args.dropout > 0:
        drop_clients = np.random.choice(np.arange(args.nworker), int(args.dropout * args.nworker))

    gt = gt_orig.copy()
    '''gt = np.zeros_like(gt_orig, dtype=float)
    for i in range(args.nworker):
        noise_matrix = np.random.normal(size=(args.row, args.col))
        lts[i] += noise_matrix
        gt += lts[i]'''

    n = np.sum(gt)
    gt_x = gt.sum(axis=1)
    gt_y = gt.sum(axis=0)
    gt_ex = np.zeros_like(gt, dtype=float)
    for i in range(args.row):
        for j in range(args.col):
            gt_ex[i][j] = 1.0 * gt_x[i] * gt_y[j] / n

    '''if (args.row - 1) * (args.col - 1) == 1:
        lts[0] = lts[0] + 0.5 * np.sign(gt_ex - gt)'''

    rv = levy_stable(2.0, 0.0)
    proj_matrix = rv.rvs(size=[args.samples, args.row * args.col])

    samples = np.zeros(args.samples)
    for i in range(args.nworker):
        if args.dropout > 0 and i in drop_clients:
            continue
        ob = lts[i]
        inter = np.divide(ob - gt_ex / args.nworker, np.sqrt(gt_ex))
        samples += np.matmul(proj_matrix, inter.flatten())

    return geometric_mean(2.0, args.samples, samples)


def get_gt(lts):
    """获得全局列联表"""
    gt = np.zeros_like(lts[0])
    for lt in lts:
        gt += lt
    return gt


class CIT(object):
    def __init__(self, data, alpha=0.05, client_num=10, method='chisq',dropout=0,is_iid = True, **kwargs):
        '''
        Parameters
        ----------
        data: numpy.ndarray of shape (n_samples, n_features)
        method: str, in ["fisherz", "mv_fisherz", "mc_fisherz", "kci", "chisq", "gsq"]
        kwargs: placeholder for future arguments, or for KCI specific arguments now
        '''
        self.data = data
        self.fl_data = None
        self.data_hash = hash(str(data))
        self.sample_size, self.num_features = data.shape
        self.method = method
        self.pvalue_cache = {}
        self.client_num = client_num
        self.args = parse_args()
        self.args.nworker = client_num
        self.alpha = alpha
        self.args.dropout=dropout

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
            SMarginalCounts = np.sum(SxyJointCounts, axis=(1, 2))
            '''SMarginalCountsNonZero = SMarginalCounts != 0
            SxyJointCounts = SxyJointCounts[SMarginalCountsNonZero]'''

            return SxyJointCounts

        def _Fill3DCountTable(dataSXY, cardSXY):
            # about the threshold 1e5, see a rough performance example at:
            # https://gist.github.com/MarkDana/e7d9663a26091585eb6882170108485e#file-count-unique-in-array-performance-md
            if np.prod(cardSXY) < 1e5: return _Fill3DCountTableByBincount(dataSXY, cardSXY)

            return _Fill3DCountTableByBincount(dataSXY, cardSXY)
            # return _Fill3DCountTableByUnique(dataSXY, cardSXY)

        def _GetExpectedtable(SxyJointCounts):
            SMarginalCounts = np.sum(SxyJointCounts, axis=(1, 2))
            SMarginalCounts_inds = SMarginalCounts == 0
            SMarginalCounts[SMarginalCounts_inds] = 1

            SxJointCounts = np.sum(SxyJointCounts, axis=2)
            SyJointCounts = np.sum(SxyJointCounts, axis=1)
            SxyExpectedCounts = SxJointCounts[:, :, None] * SyJointCounts[:, None, :] / SMarginalCounts[:, None, None]
            return SxyExpectedCounts

        def _CalculatePValue(cTables, eTables):
            eTables_zero_inds = eTables == 0
            eTables_zero_to_one = np.copy(eTables)
            eTables_zero_to_one[eTables_zero_inds] = 1  # for legal division
            sum_of_chi_square = np.sum(((cTables - eTables) ** 2) / eTables_zero_to_one)
            # array in shape (k,), zero_counts_rows[w]=c (0<=c<m) means layer w has c all-zero rows
            zero_counts_rows = eTables_zero_inds.all(axis=2).sum(axis=1)
            zero_counts_cols = eTables_zero_inds.all(axis=1).sum(axis=1)
            sum_of_df = np.sum((cTables.shape[1] - 1 - zero_counts_rows) * (cTables.shape[2] - 1 - zero_counts_cols))
            return 1 if sum_of_df == 0 else chi2.sf(sum_of_chi_square, sum_of_df)

        indexs = list(condition_set) + [X, Y]
        lts = []
        cardXY = self.cardinalities[indexs]
        if len(indexs) == 2:  # S is empty
            if self.args.dropout > 0:
                drop_clients = np.random.choice(np.arange(self.args.nworker), int(self.args.dropout * self.args.nworker))
            for i in range(self.client_num):
                if self.args.dropout > 0 and i in drop_clients:
                    continue
                xyJointCounts = _Fill2DCountTable(self.fl_data[i][:, indexs].T, cardXY)
                lts.append(xyJointCounts[None])

        # else, S is not empty: conditioning
        else:
            if self.args.dropout > 0:
                drop_clients = np.random.choice(np.arange(self.args.nworker), int(self.args.dropout * self.args.nworker))
            for i in range(self.client_num):
                if self.args.dropout > 0 and i in drop_clients:
                    continue
                SxyJointCounts = _Fill3DCountTable(self.fl_data[i][:, indexs].T, cardXY)
                lts.append(SxyJointCounts)
        lts_ex = [_GetExpectedtable(lt) for lt in lts] 

        pval_ls = [_CalculatePValue(lts[i], lts_ex[i]) for i in range(len(lts))]
        c = Counter([p_val>self.alpha for p_val in pval_ls])
        pval = max(pval_ls) if c[True]>c[False] else min(pval_ls)

        return pval


    def __call__(self, X, Y, condition_set=None, *args):
        if self.method != 'mc_fisherz':
            if len(args) != 0:
                print(end='')
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