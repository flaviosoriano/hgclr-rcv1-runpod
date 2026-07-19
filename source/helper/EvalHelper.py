import logging
import pickle
from itertools import product
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from omegaconf import OmegaConf
from ranx import evaluate, Qrels, Run
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from source.helper.Helper import Helper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Propensity-scored metric utilities (unchanged)
# ---------------------------------------------------------------------------

@nb.njit()
def in1d(a, b):
    arr = np.concatenate((a, b))
    order = arr.argsort(kind='mergesort')
    sarr = arr[order]
    bool_arr = (sarr[1:] == sarr[:-1])
    flag = np.concatenate((bool_arr, np.asarray([False])))
    ret = np.empty(arr.shape, np.bool_)
    ret[order] = flag
    return ret[:len(a)]


@nb.njit(parallel=True)
def _topk_nb(data, indices, indptr, k, pad_ind, pad_val):
    nr = len(indptr) - 1
    ind = np.full((nr, k), fill_value=pad_ind, dtype=indices.dtype)
    val = np.full((nr, k), fill_value=pad_val, dtype=data.dtype)
    for i in nb.prange(nr):
        s, e = indptr[i], indptr[i + 1]
        num_el = min(k, e - s)
        temp = np.argsort(data[s: e])[::-1][:num_el]
        ind[i, :num_el] = indices[s: e][temp]
        val[i, :num_el] = data[s: e][temp]
    return ind, val


def topk(X, k, pad_ind, pad_val, return_values=False, dtype='float32', use_cython=False):
    ind, val = _topk_nb(X.data, X.indices, X.indptr, k, pad_ind, pad_val)
    if return_values:
        return ind, val.astype(dtype)
    return ind


def compatible_shapes(x, y):
    if (sp.issparse(x) and sp.issparse(y)) \
            or (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        return x.shape == y.shape
    if not (isinstance(x, dict) or isinstance(y, dict)):
        return x.shape[0] == y.shape[0]
    else:
        if isinstance(x, dict):
            return len(x['indices']) == len(x['scores']) == y.shape[0]
        else:
            return len(y['indices']) == len(y['scores']) == x.shape[0]


def format(*args, decimal_points='%0.2f'):
    out = []
    for vals in args:
        out.append(','.join(list(map(lambda x: decimal_points % (x * 100), vals))))
    return '\n'.join(out)


def _broad_cast(mat, like):
    if isinstance(like, np.ndarray):
        return np.asarray(mat)
    elif sp.issparse(mat):
        return mat
    else:
        raise NotImplementedError("Unknown type; please pass csr_matrix, np.ndarray or dict.")


def _get_topk_sparse(X, pad_indx=0, k=5, use_cython=False):
    X = X.tocsr()
    X.sort_indices()
    pad_indx = X.shape[1]
    return topk(X, k, pad_indx, 0, return_values=False, use_cython=use_cython)


def _get_topk_array(X, k=5, sorted=False):
    assert X.shape[1] >= k, "Number of elements in X is < {}".format(k)
    if np.issubdtype(X.dtype, np.integer):
        assert sorted, "sorted must be true with indices"
        return X[:, :k] if X.shape[1] > k else X
    elif np.issubdtype(X.dtype, np.floating):
        _indices = np.argpartition(X, -k)[:, -k:]
        _scores = np.take_along_axis(X, _indices, axis=-1)
        indices = np.argsort(-_scores, axis=-1)
        return np.take_along_axis(_indices, indices, axis=1)


def _get_topk_dict(X, k=5, sorted=False):
    indices = X['indices']
    scores = X['scores']
    assert compatible_shapes(indices, scores)
    assert scores.shape[1] >= k
    if sorted:
        return indices[:, :k] if indices.shape[1] > k else indices
    if scores.shape[1] > k:
        _indices = np.argpartition(scores, -k)[:, -k:]
        _scores = np.take_along_axis(scores, _indices, axis=-1)
        __indices = np.argsort(-_scores, axis=-1)
        _indices = np.take_along_axis(_indices, __indices, axis=-1)
        return np.take_along_axis(indices, _indices, axis=-1)
    else:
        _indices = np.argsort(-scores, axis=-1)
        return np.take_along_axis(indices, _indices, axis=-1)


def _get_topk(X, pad_indx=0, k=5, sorted=False, use_cython=False):
    if sp.issparse(X):
        return _get_topk_sparse(X=X, pad_indx=pad_indx, k=k, use_cython=use_cython)
    elif isinstance(X, np.ndarray):
        return _get_topk_array(X=X, k=k, sorted=sorted)
    elif isinstance(X, dict):
        return _get_topk_dict(X=X, k=k, sorted=sorted)
    else:
        raise NotImplementedError("Unknown type; please pass csr_matrix, np.ndarray or dict.")


def compute_inv_propesity(labels, A, B):
    num_instances, _ = labels.shape
    freqs = np.ravel(np.sum(labels, axis=0))
    C = (np.log(num_instances) - 1) * np.power(B + 1, A)
    wts = 1.0 + C * np.power(freqs + B, -A)
    return np.ravel(wts)


def _setup_metric(X, true_labels, inv_psp=None, k=5, sorted=False, use_cython=False):
    assert compatible_shapes(X, true_labels)
    num_instances, num_labels = true_labels.shape
    indices = _get_topk(X, num_labels, k, sorted, use_cython)
    ps_indices = None
    if inv_psp is not None:
        _mat = sp.spdiags(inv_psp, diags=0, m=num_labels, n=num_labels)
        _psp_wtd = _broad_cast(_mat.dot(true_labels.T).T, true_labels)
        ps_indices = _get_topk(_psp_wtd, num_labels, k, False, use_cython)
        inv_psp = np.hstack([inv_psp, np.zeros((1))])
    idx_dtype = true_labels.indices.dtype
    true_labels = sp.csr_matrix(
        (true_labels.data, true_labels.indices, true_labels.indptr),
        shape=(num_instances, num_labels + 1), dtype=true_labels.dtype)
    true_labels.indices = true_labels.indices.astype(idx_dtype)
    return indices, true_labels, ps_indices, inv_psp


def _eval_flags(indices, true_labels, inv_psp=None):
    if sp.issparse(true_labels):
        nr, nc = indices.shape
        rows = np.repeat(np.arange(nr).reshape(-1, 1), nc)
        eval_flags = true_labels[rows, indices.ravel()].A1.reshape(nr, nc)
    elif type(true_labels) == np.ndarray:
        eval_flags = np.take_along_axis(true_labels, indices, axis=-1)
    if inv_psp is not None:
        eval_flags = np.multiply(inv_psp[indices], eval_flags)
    return eval_flags


def psprecision(X, true_labels, inv_psp, k=5, sorted=False, use_cython=False):
    indices, true_labels, ps_indices, inv_psp = _setup_metric(
        X, true_labels, inv_psp, k=k, sorted=sorted, use_cython=use_cython)
    eval_flags = _eval_flags(indices, true_labels, inv_psp)
    ps_eval_flags = _eval_flags(ps_indices, true_labels, inv_psp)
    return _precision(eval_flags, k) / _precision(ps_eval_flags, k)


def _precision(eval_flags, k=5):
    deno = 1 / (np.arange(k) + 1)
    return np.ravel(np.mean(np.multiply(np.cumsum(eval_flags, axis=-1), deno), axis=0))


def psndcg(X, true_labels, inv_psp, k=5, sorted=False, use_cython=False):
    indices, true_labels, ps_indices, inv_psp = _setup_metric(
        X, true_labels, inv_psp, k=k, sorted=sorted, use_cython=use_cython)
    eval_flags = _eval_flags(indices, true_labels, inv_psp)
    ps_eval_flags = _eval_flags(ps_indices, true_labels, inv_psp)
    _total_pos = np.asarray(true_labels.sum(axis=1), dtype=np.int32)
    _max_pos = max(np.max(_total_pos), k)
    _cumsum = np.cumsum(1 / np.log2(np.arange(1, _max_pos + 1) + 1))
    n = _cumsum[_total_pos - 1]
    return _ndcg(eval_flags, n, k) / _ndcg(ps_eval_flags, n, k)


def _ndcg(eval_flags, n, k=5):
    _cumsum = 0
    _dcg = np.cumsum(np.multiply(eval_flags, 1 / np.log2(np.arange(k) + 2)), axis=-1)
    ndcg = np.zeros((1, k), dtype=np.float32)
    for _k in range(k):
        _cumsum += 1 / np.log2(_k + 1 + 1)
        ndcg[0, _k] = np.mean(
            np.multiply(_dcg[:, _k].reshape(-1, 1), 1 / np.minimum(n, _cumsum))
        )
    return np.ravel(ndcg)

def _label_key_to_idx(label_key: str) -> int:
    """Extract the integer index from a label key string like 'label_123'."""
    return int(label_key.split("_")[-1])

class EvalHelper(Helper):
    def __init__(self, params):
        super(EvalHelper, self).__init__()
        self.params = params
        self.samples = self._load_samples()
        self.relevance_map = self._load_relevance_map()
        self.label_cls = self._load_labels_cls()
        self.text_cls = self._load_texts_cls()
        self.metrics = self._get_metrics()

    def _load_ranking(self, fold_idx):
        with open(f"{self.params.ranking.dir}"
            f"{self.params.model.name}_{self.params.data.name}/"
            f"{self.params.model.name}_{self.params.data.name}_{fold_idx}.rnk", "rb") as ranking_file:
            return pickle.load(ranking_file)

    def _get_cls_ranking(self, ranking, cls):
        cls_ranking = {}
        for text_idx, labels_scores in ranking.items():
            if cls in self.text_cls[int(text_idx.split("_")[-1])]:
                cls_ranking[text_idx] = {}
                for label_idx, scores in labels_scores.items():
                    if cls in self.label_cls[int(label_idx.split("_")[-1])]:
                        cls_ranking[text_idx][label_idx] = scores
        return cls_ranking



    def _eval_ranking(self, ranking, relevance_map, num_labels, inv_propesities, thresholds):
        ps_results = self._compute_ps_metrics(ranking, relevance_map, num_labels, inv_propesities, thresholds)
        tr_results = self._compute_tr_metrics(ranking)
        cls_results = self._compute_cls_metrics(ranking)
        results = {}
        results.update(ps_results)
        results.update(tr_results)
        results.update(cls_results)
        return results



    def _compute_ps_metrics(self, ranking, relevance_map, num_labels, inv_propesities, thresholds):
        text_ids_map = {k: v for v, k in enumerate(ranking.keys())}
        p_rows, p_cols, p_scores = [], [], []
        t_rows, t_cols, t_scores = [], [], []

        for text_idx, labels_scores in ranking.items():
            for label_key, score in labels_scores.items():
                label_idx = _label_key_to_idx(label_key)
                if label_idx >= 0:
                    p_rows.append(text_ids_map[text_idx])
                    p_cols.append(label_idx)
                    p_scores.append(score)

            for label_key in relevance_map[text_idx]:
                t_rows.append(text_ids_map[text_idx])
                t_cols.append(_label_key_to_idx(label_key))
                t_scores.append(1.0)

        pred = csr_matrix((p_scores, (p_rows, p_cols)), shape=(len(text_ids_map), num_labels))
        true = csr_matrix((t_scores, (t_rows, t_cols)), shape=(len(text_ids_map), num_labels))
        return self.__compute_ps_metrics(pred, true, inv_propesities, thresholds)

    def __compute_ps_metrics(self, pred, true, inv_propesities, thresholds):
        psprecisions = psprecision(pred, true, inv_propesities, k=thresholds[-1])
        psndcgs = psndcg(pred, true, inv_propesities, k=thresholds[-1])
        results = {}
        for k in thresholds:
            results[f"psnDCG@{k}"] = round(100 * psndcgs[k - 1], 1)
            results[f"psprecision@{k}"] = round(100 * psprecisions[k - 1], 1)
        return results

    def _compute_tr_metrics(self, ranking):
        result = evaluate(
            Qrels({key: value for key, value in self.relevance_map.items() if key in ranking}),
            Run(ranking),
            self.metrics,
        )
        return {k: round(100 * v, 1) for k, v in result.items()}

    def _compute_cls_metrics(self, ranking):
        results = {}
        for k in [1, 5, 10]:
            true, pred = [], []
            for text_idx in ranking:
                top_k_labels = self._get_top_k_labels(ranking[text_idx], k)
                relevant_labels = self._get_top_k_labels(self.relevance_map[text_idx], k=1)
                cls_idx = _label_key_to_idx(relevant_labels[0])

                if any(label in relevant_labels for label in top_k_labels):
                    true.append(cls_idx)
                    pred.append(cls_idx)
                else:
                    true.append(cls_idx)
                    pred.append(_label_key_to_idx(top_k_labels[0]))

            results[f"Mac-F1@{k}"] = round(100 * round(f1_score(true, pred, average='macro'), 2), 1)
            results[f"Mic-F1@{k}"] = round(100 * round(f1_score(true, pred, average='micro'), 2), 1)
        return results

    def _get_top_k_labels(self, label_scores, k):
        return sorted(label_scores, key=label_scores.get, reverse=True)[:k]

    def run(self):
        # Compute inverse propensity scores from the full label matrix
        labels_ids = [sample['labels_ids'] for sample in self.samples]
        mlb = MultiLabelBinarizer(sparse_output=True)
        labels = mlb.fit_transform(labels_ids)
        inv_propensity = compute_inv_propesity(
            labels,
            self.params.data.propensity.A,
            self.params.data.propensity.B,
        )

        for fold_idx in self.params.data.folds:
            logger.info(
                f"\nEvaluating {self.params.data.name} dataset (fold {fold_idx}) with the following "
                f"params\n {OmegaConf.to_yaml(self.params)}\n"
            )

            logger.info(f"Loading ranking.")
            ranking = self._load_ranking(fold_idx)
            results = []

            for cls in self.params.eval.label_cls:
                cls_ranking = self._get_cls_ranking(ranking, cls)
                logger.info(f"Evaluating {cls} ranking.")
                result = self._eval_ranking(
                    cls_ranking,
                    self.relevance_map,
                    inv_propensity.shape[0],
                    inv_propensity,
                    self.params.eval.thresholds,
                )

                result["fold_idx"] = fold_idx
                result["cls"] = cls
                results.append(result)

        self._checkpoint_result(results, fold_idx)

    def _checkpoint_result(self, results, fold_idx):

        Path(f"{self.params.result.dir}"
             f"{self.params.model.name}_{self.params.data.name}/").mkdir(parents=True, exist_ok=True)

        pd.DataFrame(results).to_csv(f"{self.params.result.dir}"
                                    f"{self.params.model.name}_{self.params.data.name}/"
                                    f"{self.params.model.name}_{self.params.data.name}_{fold_idx}.rts",
                                    sep='\t', index=False, header=True)
        logger.info(f"Saved result in {self.params.result.dir}"
                                    f"{self.params.model.name}_{self.params.data.name}/"
                                    f"{self.params.model.name}_{self.params.data.name}_{fold_idx}.rts")