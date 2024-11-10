# Authors: Shani Cohen (ShaniCohen)
# Python version: 3.8
# Last update: 28.10.2020

import pandas as pd
import numpy as np
import os
import random
from utils_old.utils_general import order_df, split_df_samples
import itertools
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from typing import Set, Dict, List, Optional
from models.kgraphrna import kGraphRNA
# import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
# from torch_geometric.loader import LinkNeighborLoader
# import torch_geometric.transforms as T
import logging
from skbio import Sequence
from itertools import product
logger = logging.getLogger(__name__)

KMERS_COMB = [''.join(p) for p in product(["A", "T", "G", "C"], repeat=3)]
ZERO_KMERS = dict(zip(KMERS_COMB, np.repeat(0, len(KMERS_COMB))))


def get_kmers_freqs(rna_sequence: str, k: int = 3) -> Dict[str, float]:
    kmers_freqs = ZERO_KMERS.copy()
    rna_freqs = Sequence(rna_sequence).kmer_frequencies(k, relative=True, overlap=True)
    kmers_freqs.update(rna_freqs)

    return kmers_freqs


def calc_rna_features(rna_sequences_lst: List[str]) -> pd.DataFrame:
    logger.debug("calculating RNA kmers")
    rna_kmers_outs = list(map(get_kmers_freqs, rna_sequences_lst))
    rna_kmers = pd.DataFrame(rna_kmers_outs)[KMERS_COMB]
    return rna_kmers


def _train(model, device, optimizer, model_args, train_data) -> float:
    # todo - add val loss and training history
    model.train()
    train_data.to(device)
    optimizer.zero_grad()

    pred = model(train_data, model_args)
    ground_truth = train_data['srna', 'targets', 'mrna'].edge_label
    loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    loss.backward()
    optimizer.step()
    out_loss = loss.detach().cpu().numpy().tolist()
    return out_loss


@torch.no_grad()
def _test(model, device, model_args, eval_data) -> (float, float):
    model.eval()
    # 1 - predict on dataset
    preds = []
    ground_truths = []
    srna_nids, mrna_nids = [], []
    with torch.no_grad():
        eval_data.to(device)
        pred = model(eval_data, model_args)
        ground_truth = eval_data['srna', 'targets', 'mrna'].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        out_loss = loss.detach().cpu().numpy().tolist()
        preds.append(pred)
        ground_truths.append(ground_truth)

    scores_arr = list(torch.cat(preds, dim=0).cpu().numpy())
    labels_arr = list(torch.cat(ground_truths, dim=0).cpu().numpy())
    if len(set(labels_arr)) == 1:
        print()
    roc_auc = roc_auc_score(labels_arr, scores_arr)

    return out_loss, roc_auc


def train(model, device, optimizer, model_args, train_data) -> (float, float):
    model.train()
    preds, targets = [], []

    train_data.to(device)
    optimizer.zero_grad()
    pred = model(
        train_data.x_dict,
        train_data.edge_index_dict,
        train_data['srna', 'mrna'].edge_label_index,
    )
    target = train_data['srna', 'mrna'].edge_label
    loss = F.binary_cross_entropy_with_logits(pred, target)

    out_loss = loss.detach().cpu().numpy().tolist()
    preds.append(pred)
    targets.append(target)

    scores_arr = list(torch.cat(preds, dim=0).cpu().numpy())
    targets_arr = list(torch.cat(targets, dim=0).cpu().numpy())

    roc_auc = roc_auc_score(targets_arr, scores_arr)

    return out_loss, roc_auc


@torch.no_grad()
def test(model, test_data) -> (float, float):
    model.eval()
    preds, targets = [], []

    pred = model(
        test_data.x_dict,
        test_data.edge_index_dict,
        test_data['srna', 'mrna'].edge_label_index,
    )
    target = test_data['srna', 'mrna'].edge_label
    loss = F.binary_cross_entropy_with_logits(pred, target)
    out_loss = loss.detach().cpu().numpy().tolist()
    pred = pred.sigmoid().view(-1).cpu()
    preds.append(pred)
    targets.append(target)

    scores_arr = list(torch.cat(preds, dim=0).cpu().numpy())
    targets_arr = list(torch.cat(targets, dim=0).cpu().numpy())

    roc_auc = roc_auc_score(targets_arr, scores_arr)

    return out_loss, roc_auc


class kGraphRNAModelHandler(object):
    """
    Parameters
    ----------

    Attributes
    ----------
    Same as parameters
    """
    # ------  Nodes  ------
    nodes_are_defined = False
    ##  mRNA
    mrna_nodes = None  # init in _prepare_data
    mrna_nid_col = 'mrna_node_id'
    mrna_eco_acc_col = None
    mrna = 'mrna'

    ##  sRNA
    srna_nodes = None  # init in _prepare_data
    srna_nid_col = 'srna_node_id'
    srna_eco_acc_col = None
    srna = 'srna'

    # ------  Edges  ------
    # ---  interactions (sRNA - mRNA)
    srna_mrna_val_col = None  # in case additional edge features are requested
    srna_to_mrna = 'targets'
    binary_intr_label_col = 'interaction_label'

    # ------  Params  ------
    add_mrna_mrna_similarity_edges = False
    # --- random dataset split
    random_split = False
    # Generate fixed negative edges (in val_data) for evaluation with a ratio of x:1
    val_and_test_neg_sampling_ratio = 1.0

    # --- train data
    # for train data, sample negative edges with a ratio of x:1
    cv_neg_sampling_ratio_data = 1.0
    train_neg_sampling_ratio_data = 1.0
    sampling_seed = 20
    # Across the training edges, we use 70% of edges for message passing, and 30% of edges for supervision.
    train_supervision_ratio = 0.3

    # --- train loader
    train_w_loader = False
    # in the first hop, sample at most 20 neighbors
    # in the second hop, sample at most 10 neighbors
    train_num_neighbors = [20, 10]
    # during training, sample negative edges on-the-fly with a ratio of x:1
    train_neg_sampling_ratio_loader = 2.0
    train_batch_size = 128
    # set to True to have the data reshuffled at every epoch
    train_shuffle = True

    # --- test data
    eval_num_neighbors = [20, 0]
    eval_batch_size = 128
    eval_shuffle = False

    shuffle_test = True
    debug_logs = False

    # --- seed generator
    gen = None

    def __init__(self, seed: int = 100):
        super(kGraphRNAModelHandler, self).__init__()
        self.seed = seed
        # set random seeds
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # # When running on the CuDNN backend, two further options must be set
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)

    @staticmethod
    def get_const_hyperparams(**kwargs) -> Dict[str, object]:
        """   torch_geometric.__version__ = '2.1.0'  """
        logger.debug(f"getting const kGraphRNA hyperparams")
        const_model_hyperparams = {
        }
        return const_model_hyperparams

    @staticmethod
    def get_model_args(**kwargs) -> Dict[str, object]:
        """   torch_geometric.__version__ = '2.1.0'  """
        logger.debug("getting kGraphRNA model_args")
        model_args = {
            # 1 - constructor args
            'hidden_channels': 64,
            'num_sim_feat': 50,
            # 'random_state': seed,
            # 'verbose': verbose
            # 2 - fit (train) args
            'learning_rate': 0.001,
            'epochs': 120
        }
        return model_args

    def train_and_test(self, X_train: pd.DataFrame, y_train: List[int], X_test: pd.DataFrame, y_test: List[int],
                       model_args: dict, metadata_train: pd.DataFrame, metadata_test: pd.DataFrame,
                       unq_train: pd.DataFrame = None, unq_test: pd.DataFrame = None, predict_all_pairs: bool = False,
                       avoid_scores: bool = False, train_neg_sampling: bool = True, srna_acc_col: str = 'sRNA_accession_id_Eco',
                       mrna_acc_col: str = 'mRNA_accession_id_Eco', is_syn_col: str = 'is_synthetic', **kwargs) -> \
            (Dict[str, dict], Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object]):
        """
        torch_geometric.__version__ = '2.1.0'

        Parameters
        ----------
        X_train: pd.DataFrame (n_samples, N_features),
        y_train: list (n_samples,),
        X_test: pd.DataFrame (t_samples, N_features),
        y_test: list (t_samples,)
        model_args: Dict of model's constructor and fit() arguments
        metadata_train: pd.DataFrame (n_samples, T_features)
        metadata_test: pd.DataFrame (t_samples, T_features)
        unq_train: pd.DataFrame (_samples, T_features)
        unq_test: pd.DataFrame (t_samples, T_features)
        train_neg_sampling: whether to add random negative sampling to train HeteroData
        srna_acc_col: str  sRNA EcoCyc accession id col in metadata_train and metadata_test
        mrna_acc_col: str  mRNA EcoCyc accession id col in metadata_train and metadata_test
        is_syn_col: is synthetic indicator col in metadata_train
        kwargs:
            mrna_eco: all mRNA metadata from EcoCyc
            mrna_similarity: known similarity score between mRNAs (id = EcoCyc accession id)

        Returns
        -------

        scores - Dict in the following format:  {
            'test': {
                <score_nm>: score
                }
        }
        predictions - Dict in the following format:  {
            'test_pred': array-like (t_samples,)  - ordered as y test
        }
        training_history - Dict in the following format:  {
            'train': OrderedDict
            'validation': OrderedDict
        }
        train_val_data - Dict in the following format:  {
            "X_train": pd.DataFrame (n_samples, N_features),
            "y_train": list (n_samples,),
            "X_val": pd.DataFrame (k_samples, K_features),
            "y_val": list (k_samples,),
            "metadata_train": pd.DataFrame (n_samples, T_features),    - Optional (in case metadata_train is not None)
            "metadata_val": list (k_samples,)    - Optional (in case metadata_train is not None)
        }
        shap_args - Dict in the following format:  {
            "model_nm": str,
            "trained_model": the trained machine learning model,
            "X_train": pd.DataFrame,
            "X_val": Optional[pd.DataFrame],
            "X_test": pd.DataFrame
        }
        """
        logger.debug(f"training an kGraphRNA model  ->  train negative sampling = {train_neg_sampling}")
        if not self.nodes_are_defined:
            self._define_nodes_and_features(**kwargs)

        if predict_all_pairs:  # predict all pairs
            unq_test = self._map_interactions_to_edges(unique_intr=unq_test, srna_acc_col=srna_acc_col,
                                                       mrna_acc_col=mrna_acc_col)
            unq_test[self.binary_intr_label_col] = 1
            avoid_scores = True

        out_test_pred = pd.DataFrame({
            self.srna_nid_col: unq_test[self.srna_nid_col],
            self.mrna_nid_col: unq_test[self.mrna_nid_col]
        })
        # init train & test sets (HeteroData)
        train_data, test_data = self._init_train_test_hetero_data(unq_train=unq_train, unq_test=unq_test,
                                                                  train_neg_sampling=train_neg_sampling)
        # insert node features
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_data[self.srna].x = torch.from_numpy(np.array(self.srna_nodes[self.srna_feat_cols])).to(torch.float32)
        train_data[self.mrna].x = torch.from_numpy(np.array(self.mrna_nodes[self.mrna_feat_cols])).to(torch.float32)
        test_data[self.srna].x = torch.from_numpy(np.array(self.srna_nodes[self.srna_feat_cols])).to(torch.float32)
        test_data[self.mrna].x = torch.from_numpy(np.array(self.mrna_nodes[self.mrna_feat_cols])).to(torch.float32)
        # 7 - init kGraphRNA model
        # 7.2 - init model
        model_args['hidden_channels'] = 32
        model = kGraphRNA(hidden_channels=model_args['hidden_channels'],
                          srna=self.srna, mrna=self.mrna, srna_to_mrna=self.srna_to_mrna)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # 8 - train kGraphRNA model
        for epoch in range(model_args['epochs']):
            # --- train
            model.train()
            preds, targets = [], []
            optimizer.zero_grad()
            pred = model(
                train_data.x_dict,
                train_data.edge_index_dict,
                train_data[self.srna, self.mrna].edge_label_index,
            )
            target = train_data[self.srna, self.mrna].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, target)
            loss.backward()
            optimizer.step()

            preds.append(pred)
            targets.append(target)
            scores_arr = list(torch.cat(preds, dim=0).detach().cpu().numpy())
            targets_arr = list(torch.cat(targets, dim=0).detach().cpu().numpy())
            train_loss = loss.detach().cpu().numpy().tolist()
            roc_auc = roc_auc_score(targets_arr, scores_arr)

            # --- test
            if predict_all_pairs:
                test_str = ""
            else:
                test_loss, test_auc = test(model=model, test_data=test_data)
                test_str = f" Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}"
            logger.debug(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train AUC: {roc_auc:.4f}{test_str}")

        # if self.train_w_loader:
        #     hg_model = self._train_hgnn_with_loader(hg_model=hga_model, train_data=train_data, model_args=model_args)
        # else:
        #     hg_model = self._train_hgnn(hga_model=hga_model, train_data=train_data, model_args=model_args)
        training_history, train_val_data = {}, {}

        # 9 - evaluate - calc scores
        logger.debug("evaluating test set")
        predictions, scores = {}, {}
        test_scores, test_pred_df = self._eval_hgnn(trained_model=model, eval_data=test_data, model_args=model_args,
                                                    avoid_scores=avoid_scores, **kwargs)
        assert pd.isnull(test_pred_df).sum().sum() == 0, "some null predictions"

        # 10 - update outputs
        _len = len(out_test_pred)
        out_test_pred = pd.merge(out_test_pred, test_pred_df, on=[self.srna_nid_col, self.mrna_nid_col], how='left')
        assert len(out_test_pred) == _len
        test_y_pred = out_test_pred['y_score']

        scores.update({'test': test_scores})
        predictions.update({'test_pred': test_y_pred,
                            'test_pred_df': test_pred_df, 'out_test_pred': out_test_pred})

        # 11 - SHAP args
        shap_args = {
            "model_nm": "kGraphRNA",
            "trained_model": model,
            "X_train": pd.DataFrame(),
            "X_val": None,
            "X_test": pd.DataFrame()
        }

        return scores, predictions, training_history, train_val_data, shap_args

    #
    # @staticmethod
    # def log_df_stats(df: pd.DataFrame, label_col: str, df_nm: str = None):
    #     df_nm = 'df' if pd.isnull(df_nm) else df_nm
    #     _pos_df = sum(df[label_col])
    #     logger.debug(f' {df_nm}: {len(df)} interactions (P: {_pos_df}, N: {len(df) - _pos_df})')
    #     return
    #
    @staticmethod
    def log_df_rna_eco(rna_name: str, rna_eco_df: pd.DataFrame, acc_col: str):
        logger.debug(f' {rna_name} data from EcoCyc: {len(set(rna_eco_df[acc_col]))} unique {rna_name}s of E.coli')
        return

    @staticmethod
    def _map_rna_nodes_and_edges(out_rna_node_id_col: str, rna_eco: pd.DataFrame, e_acc_col: str,
                                 rna_sim: pd.DataFrame, s_acc_1_col: str, s_acc_2_col: str, s_score_col: str) -> \
            (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """ map accession to node id """
        # 1 - Nodes = EcoCyc data
        rna_nodes = rna_eco
        # 1.1 - map from original id (acc) to node id (consecutive values)
        rna_nodes[out_rna_node_id_col] = np.arange(len(rna_nodes))
        rna_nodes = order_df(df=rna_nodes, first_cols=[out_rna_node_id_col])
        rna_map = rna_eco[[out_rna_node_id_col, e_acc_col]]
        # 2 - Edges = similarity data
        if type(rna_sim) is pd.DataFrame:
            _len = len(rna_sim)
            rna_edges = pd.merge(rna_sim, rna_map, left_on=s_acc_1_col, right_on=e_acc_col, how='left').rename(
                columns={out_rna_node_id_col: f'{out_rna_node_id_col}_1'})
            rna_edges = pd.merge(rna_edges, rna_map, left_on=s_acc_2_col, right_on=e_acc_col, how='left').rename(
                columns={out_rna_node_id_col: f'{out_rna_node_id_col}_2'})
            assert len(rna_edges) == _len, "duplications post merge"
            _cols = [f'{out_rna_node_id_col}_1', f'{out_rna_node_id_col}_2', s_score_col, s_acc_1_col, s_acc_2_col]
            rna_edges = rna_edges[_cols]
        else:
            rna_edges = None

        return rna_map, rna_nodes, rna_edges

    @staticmethod
    def _pos_neg_split(df: pd.DataFrame, binary_label_col: str) -> (pd.DataFrame, pd.DataFrame):
        df_pos = df[df[binary_label_col] == 1].reset_index(drop=True)
        df_neg = df[df[binary_label_col] == 0].reset_index(drop=True)
        return df_pos, df_neg

    # @staticmethod
    # def _remove_synthetic_samples(X: pd.DataFrame, y: List[int], metadata: pd.DataFrame, is_syn_col: str) -> \
    #         (pd.DataFrame, List[int], pd.DataFrame):
    #     '''
    #
    #     Parameters
    #     ----------
    #     X: pd.DataFrame (n_samples, N_features)
    #     y: list (n_samples,)
    #     metadata: pd.DataFrame (n_samples, T_features)
    #     is_syn_col: is synthetic indicator columns in metadata df
    #
    #     Returns
    #     -------
    #
    #     '''
    #     logger.debug("removing synthetic samples from train")
    #     assert len(X) == len(y) == len(metadata), "X, y, metadata mismatch"
    #     assert sorted(set(y)) in [[0, 1], [1]], "y is not binary"
    #
    #     mask_not_synthetic = ~metadata[is_syn_col].reset_index(drop=True)
    #     metadata = metadata.reset_index(drop=True)[mask_not_synthetic]
    #     X = X.reset_index(drop=True)[mask_not_synthetic]
    #     y = list(pd.Series(y)[mask_not_synthetic])
    #     logger.debug(f"removed {len(mask_not_synthetic) - len(X)} synthetic samples from X "
    #                  f"(before: {len(mask_not_synthetic)}, after: {len(X)})")
    #     logger.debug(f"removed {len(mask_not_synthetic) - len(metadata)} synthetic samples from metadata "
    #                  f"(before: {len(mask_not_synthetic)}, after: {len(metadata)})")
    #     return X, y, metadata
    def _add_neg_samples(self, unq_intr_pos: pd.DataFrame, ratio: float, _shuffle: bool = False) -> HeteroData:
        assert sum(unq_intr_pos[self.binary_intr_label_col]) == len(unq_intr_pos), "unq_intr_pos has negatives"
        _pos = list(zip(list(unq_intr_pos[self.srna_nid_col]), list(unq_intr_pos[self.mrna_nid_col])))
        _all = list(itertools.product(list(self.srna_nodes[self.srna_nid_col]), list(self.mrna_nodes[self.mrna_nid_col])))
        _unknown = pd.Series(list(set(_all) - set(_pos)))
        _unknown_df = pd.DataFrame({
            self.binary_intr_label_col: 0,
            self.srna_nid_col: _unknown.apply(lambda x: x[0]),
            self.mrna_nid_col: _unknown.apply(lambda x: x[1])
        })

        n = max(int(len(_pos) * ratio), 1)
        _neg_samples = _unknown_df.sample(n=n, random_state=self.sampling_seed)
        out = pd.concat(objs=[unq_intr_pos, _neg_samples], axis=0, ignore_index=True).reset_index(drop=True)
        if _shuffle:
            out = pd.DataFrame(shuffle(out)).reset_index(drop=True)

        return out

    #
    # def _get_unique_inter(self, metadata: pd.DataFrame, y: List[int], srna_acc_col: str, mrna_acc_col: str,
    #                       df_nm: str = None) -> pd.DataFrame:
    #     # 1 - data validation
    #     _len = len(metadata)
    #     srna_acc = metadata[srna_acc_col]
    #     mrna_acc = metadata[mrna_acc_col]
    #     assert sorted(set(y)) in [[0, 1], [1]], "y is not binary"
    #     assert sum(pd.isnull(srna_acc)) + sum(pd.isnull(mrna_acc)) == 0, "some acc id are null"
    #     # 2 - get unique sRNA-mRNA interactions
    #     unq_intr = pd.DataFrame({
    #         srna_acc_col: metadata[srna_acc_col],
    #         mrna_acc_col: metadata[mrna_acc_col],
    #         self.binary_intr_label_col: y
    #     })
    #     self.log_df_stats(df=unq_intr, label_col=self.binary_intr_label_col, df_nm=df_nm)
    #     unq_intr = unq_intr.drop_duplicates().reset_index(drop=True)
    #     # 2 - log unique
    #     self.log_df_stats(df=unq_intr, label_col=self.binary_intr_label_col, df_nm=f"unique_{df_nm}")
    #
    #     return unq_intr
    #
    # def _assert_no_data_leakage(self, unq_train: pd.DataFrame, unq_test: pd.DataFrame, srna_acc_col: str,
    #                             mrna_acc_col: str):
    #     logger.debug("assert no data leakage")
    #     train_tup = set(zip(unq_train[srna_acc_col], unq_train[mrna_acc_col], unq_train[self.binary_intr_label_col]))
    #     test_tup = set(zip(unq_test[srna_acc_col], unq_test[mrna_acc_col], unq_test[self.binary_intr_label_col]))
    #     dupl = sorted(train_tup - (train_tup - test_tup))
    #     assert len(dupl) == 0, f"{len(dupl)} duplicated interactions in train and test"
    #     return
    #
    # def _remove_train_inter_from_test_all_pairs(self, unq_train: pd.DataFrame, unq_test: pd.DataFrame, srna_acc_col: str,
    #                                             mrna_acc_col: str) -> pd.DataFrame:
    #     logger.debug("remove train inter_from_test")
    #     train_tup = set(zip(unq_train[srna_acc_col], unq_train[mrna_acc_col]))
    #     test_tup = set(zip(unq_test[srna_acc_col], unq_test[mrna_acc_col]))
    #     test_no_dup = pd.DataFrame(list(test_tup - train_tup), columns=[srna_acc_col, mrna_acc_col])
    #     logger.debug(f"from {len(test_tup)} interactions {len(test_tup) - len(test_no_dup)} were filtered since appear "
    #                  f"in train --> {len(test_no_dup)} left")
    #     return test_no_dup

    def _map_inter(self, intr: pd.DataFrame, mrna_acc_col: str, srna_acc_col: str, mrna_map: pd.DataFrame,
                   m_map_acc_col: str, srna_map: pd.DataFrame, s_map_acc_col: str) -> pd.DataFrame:
        _len, _cols = len(intr), list(intr.columns.values)
        intr = pd.merge(intr, mrna_map, left_on=mrna_acc_col, right_on=m_map_acc_col, how='left')
        intr = pd.merge(intr, srna_map, left_on=srna_acc_col, right_on=s_map_acc_col, how='left')
        assert len(intr) == _len, "duplications post merge"
        intr = intr[[self.srna_nid_col, self.mrna_nid_col] + _cols]
        return intr

    def _define_nodes_and_features(self, **kwargs):
        # 1 - mRNA
        # 1.1 - get map, nodes and edges
        m_eco_acc_col = kwargs['me_acc_col']
        m_sim_score_col = kwargs['ms_score_col']
        self.log_df_rna_eco(rna_name='mRNA', rna_eco_df=kwargs['mrna_eco'], acc_col=m_eco_acc_col)
        _, mrna_nodes, mrna_mrna_edges = \
            self._map_rna_nodes_and_edges(out_rna_node_id_col=self.mrna_nid_col, rna_eco=kwargs['mrna_eco'],
                                          e_acc_col=m_eco_acc_col, rna_sim=kwargs['mrna_similarity'],
                                          s_acc_1_col=kwargs['ms_acc_1_col'], s_acc_2_col=kwargs['ms_acc_2_col'],
                                          s_score_col=m_sim_score_col)
        # todo - NEW
        # 1.2 - add mRNA features
        mrna_sequences = mrna_nodes['EcoCyc_sequence'].apply(lambda x: x if pd.notnull(x) else '')
        mrna_feat = calc_rna_features(rna_sequences_lst=list(mrna_sequences))
        self.mrna_feat_cols = list(mrna_feat.columns.values)
        # todo - NEW
        # self.mrna_nodes = mrna_nodes
        self.mrna_nodes = pd.concat(objs=[mrna_nodes, mrna_feat], axis=1)
        self.mrna_eco_acc_col = m_eco_acc_col

        # 2 - sRNA
        # todo - add sRNA similarity measure
        # 2.1 - get map, nodes and edges
        s_eco_acc_col = kwargs['se_acc_col']
        s_sim_score_col = kwargs['ss_score_col']
        self.log_df_rna_eco(rna_name='sRNA', rna_eco_df=kwargs['srna_eco'], acc_col=s_eco_acc_col)
        _, srna_nodes, srna_srna_edges = \
            self._map_rna_nodes_and_edges(out_rna_node_id_col=self.srna_nid_col, rna_eco=kwargs['srna_eco'],
                                          e_acc_col=s_eco_acc_col, rna_sim=kwargs['srna_similarity'],
                                          s_acc_1_col=kwargs['ss_acc_1_col'], s_acc_2_col=kwargs['ss_acc_2_col'],
                                          s_score_col=s_sim_score_col)
        # todo - NEW
        # 2.2 - add sRNA features
        srna_sequences = srna_nodes['EcoCyc_sequence'].apply(lambda x: x if pd.notnull(x) else '')
        srna_feat = calc_rna_features(rna_sequences_lst=list(srna_sequences))
        self.srna_feat_cols = list(srna_feat.columns.values)
        # todo - NEW
        # 2.3 - set
        # nodes
        # self.srna_nodes = srna_nodes
        self.srna_nodes = pd.concat(objs=[srna_nodes, srna_feat], axis=1)
        self.srna_eco_acc_col = s_eco_acc_col
        # indicator
        self.nodes_are_defined = True

        return
    #
    # def _define_nodes_and_map_edges(self, unique_train: pd.DataFrame, unique_test: pd.DataFrame, srna_acc_col: str,
    #                                 mrna_acc_col: str, **kwargs) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    #     logger.debug("defining graph nodes and mapping edges")
    #     #  ------------  Similarity  ------------
    #     # 1 - mRNA
    #     # 1.1 - get map, nodes and edges
    #     m_eco_acc_col = kwargs['me_acc_col']
    #     m_sim_score_col = kwargs['ms_score_col']
    #     self.log_df_rna_eco(rna_name='mRNA', rna_eco_df=kwargs['mrna_eco'], acc_col=m_eco_acc_col)
    #     mrna_map, mrna_nodes, mrna_mrna_edges = \
    #         self._map_rna_nodes_and_edges(out_rna_node_id_col=self.mrna_nid_col, rna_eco=kwargs['mrna_eco'],
    #                                       e_acc_col=m_eco_acc_col, rna_sim=kwargs['mrna_similarity'],
    #                                       s_acc_1_col=kwargs['ms_acc_1_col'], s_acc_2_col=kwargs['ms_acc_2_col'],
    #                                       s_score_col=m_sim_score_col)
    #     # 1.2 - set
    #     self.mrna_nodes = mrna_nodes
    #     self.mrna_mrna_edges = mrna_mrna_edges
    #     self.mrna_mrna_val_col = m_sim_score_col
    #
    #     # 2 - sRNA
    #     # todo - add sRNA similarity measure
    #     # 2.1 - get map, nodes and edges
    #     s_eco_acc_col = kwargs['se_acc_col']
    #     s_sim_score_col = kwargs['ss_score_col']
    #     self.log_df_rna_eco(rna_name='sRNA', rna_eco_df=kwargs['srna_eco'], acc_col=s_eco_acc_col)
    #     srna_map, srna_nodes, srna_srna_edges = \
    #         self._map_rna_nodes_and_edges(out_rna_node_id_col=self.srna_nid_col, rna_eco=kwargs['srna_eco'],
    #                                       e_acc_col=s_eco_acc_col, rna_sim=kwargs['srna_similarity'],
    #                                       s_acc_1_col=kwargs['ss_acc_1_col'], s_acc_2_col=kwargs['ss_acc_2_col'],
    #                                       s_score_col=s_sim_score_col)
    #     # 2.2 - set
    #     self.srna_nodes = srna_nodes
    #     self.srna_srna_edges = srna_srna_edges
    #     self.srna_srna_val_col = None
    #
    #     # ------------  interactions  ------------
    #     # 3.1 - train
    #     unique_train = self._map_inter(intr=unique_train, mrna_acc_col=mrna_acc_col, srna_acc_col=srna_acc_col,
    #                                    mrna_map=mrna_map, m_map_acc_col=m_eco_acc_col, srna_map=srna_map,
    #                                    s_map_acc_col=s_eco_acc_col)
    #     unique_train = unique_train.sort_values(by=[self.srna_nid_col, self.mrna_nid_col]).reset_index(drop=True)
    #
    #     # 3.2 - test
    #     unique_test = self._map_inter(intr=unique_test, mrna_acc_col=mrna_acc_col, srna_acc_col=srna_acc_col,
    #                                   mrna_map=mrna_map, m_map_acc_col=m_eco_acc_col, srna_map=srna_map,
    #                                   s_map_acc_col=s_eco_acc_col)
    #     unique_test = unique_test.sort_values(by=[self.srna_nid_col, self.mrna_nid_col]).reset_index(drop=True)
    #     # 3.3 - unified (should not be used)
    #     unique_data = pd.concat(objs=[unique_train, unique_test], axis=0, ignore_index=True).reset_index(drop=True)
    #     self.log_df_stats(df=unique_data, label_col=self.binary_intr_label_col, df_nm=f"unique_data")
    #
    #     return unique_data, unique_train, unique_test

    def _map_interactions_to_edges(self, unique_intr: pd.DataFrame, srna_acc_col: str, mrna_acc_col: str) -> \
            (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        logger.debug("mapping interactions to edges")
        mrna_map = self.mrna_nodes[[self.mrna_nid_col, self.mrna_eco_acc_col]]
        srna_map = self.srna_nodes[[self.srna_nid_col, self.srna_eco_acc_col]]

        unique_intr = self._map_inter(intr=unique_intr, mrna_acc_col=mrna_acc_col, srna_acc_col=srna_acc_col,
                                      mrna_map=mrna_map, m_map_acc_col=self.mrna_eco_acc_col, srna_map=srna_map,
                                      s_map_acc_col=self.srna_eco_acc_col)
        unique_intr = unique_intr.sort_values(by=[self.srna_nid_col, self.mrna_nid_col]).reset_index(drop=True)

        return unique_intr
    #
    # def _generate_train_and_test(self, edges: dict) -> (HeteroData, HeteroData):
    #     # ------  Train Data  ------
    #     train_data = HeteroData()
    #     # ------  Nodes
    #     train_data[self.srna].node_id = torch.arange(len(self.srna_nodes))
    #     train_data[self.mrna].node_id = torch.arange(len(self.mrna_nodes))
    #     train_data[self.mrna].x = None
    #     # ------  Edges
    #     # edges for message passing
    #     train_edge_index = torch.stack([torch.from_numpy(np.array(edges['train']['message_passing']['label_index_0'])),
    #                                     torch.from_numpy(np.array(edges['train']['message_passing']['label_index_1']))],
    #                                    dim=0)
    #     train_data[self.srna, self.srna_to_mrna, self.mrna].edge_index = train_edge_index
    #     train_data = T.ToUndirected()(train_data)
    #     # edges for supervision
    #     train_data[self.srna, self.srna_to_mrna, self.mrna].edge_label = \
    #         torch.from_numpy(np.array(edges['train']['supervision']['label'])).float()
    #     train_data[self.srna, self.srna_to_mrna, self.mrna].edge_label_index = \
    #         torch.stack([torch.from_numpy(np.array(edges['train']['supervision']['label_index_0'])),
    #                      torch.from_numpy(np.array(edges['train']['supervision']['label_index_1']))], dim=0)
    #
    #     # ------  Test Data  ------
    #     test_data = HeteroData()
    #     # ------  Nodes
    #     test_data[self.srna].node_id = torch.arange(len(self.srna_nodes))
    #     test_data[self.mrna].node_id = torch.arange(len(self.mrna_nodes))
    #     test_data[self.mrna].x = None
    #     # ------  Edges
    #     # all train edges for message passing
    #     test_edge_index = torch.stack([torch.from_numpy(np.array(edges['train']['all']['label_index_0'])),
    #                                    torch.from_numpy(np.array(edges['train']['all']['label_index_1']))], dim=0)
    #     test_data[self.srna, self.srna_to_mrna, self.mrna].edge_index = test_edge_index
    #     test_data = T.ToUndirected()(test_data)
    #     # test edges for supervision
    #     test_data[self.srna, self.srna_to_mrna, self.mrna].edge_label = \
    #         torch.from_numpy(np.array(edges['test']['label'])).float()
    #     test_data[self.srna, self.srna_to_mrna, self.mrna].edge_label_index = \
    #         torch.stack([torch.from_numpy(np.array(edges['test']['label_index_0'])),
    #                      torch.from_numpy(np.array(edges['test']['label_index_1']))], dim=0)
    #     if self.debug_logs:
    #         logger.debug(f"\n Train data:\n ============== \n{train_data}\n"
    #                      f"\n Test data:\n ============== \n{test_data}\n")
    #
    #     return train_data, test_data
    #
    # def _generate_train_and_test_and_val(self, edges: dict) -> (HeteroData, HeteroData):
    #     # ------  Train Data  ------
    #     train_data = HeteroData()
    #     # ------  Nodes
    #     train_data[self.srna].node_id = torch.arange(len(self.srna_nodes))
    #     train_data[self.mrna].node_id = torch.arange(len(self.mrna_nodes))
    #     train_data[self.mrna].x = None
    #     # ------  Edges
    #     # edges for message passing
    #     train_edge_index = torch.stack([torch.from_numpy(np.array(edges['train']['message_passing']['label_index_0'])),
    #                                     torch.from_numpy(np.array(edges['train']['message_passing']['label_index_1']))],
    #                                    dim=0)
    #     train_data[self.srna, self.srna_to_mrna, self.mrna].edge_index = train_edge_index
    #     train_data = T.ToUndirected()(train_data)
    #     # edges for supervision
    #     train_data[self.srna, self.srna_to_mrna, self.mrna].edge_label = \
    #         torch.from_numpy(np.array(edges['train']['supervision']['label'])).float()
    #     train_data[self.srna, self.srna_to_mrna, self.mrna].edge_label_index = \
    #         torch.stack([torch.from_numpy(np.array(edges['train']['supervision']['label_index_0'])),
    #                      torch.from_numpy(np.array(edges['train']['supervision']['label_index_1']))], dim=0)
    #
    #     # ------  Test Data  ------
    #     test_data = HeteroData()
    #     # ------  Nodes
    #     test_data[self.srna].node_id = torch.arange(len(self.srna_nodes))
    #     test_data[self.mrna].node_id = torch.arange(len(self.mrna_nodes))
    #     test_data[self.mrna].x = None
    #     # ------  Edges
    #     # all train edges for message passing
    #     test_edge_index = torch.stack([torch.from_numpy(np.array(edges['train']['all']['label_index_0'])),
    #                                    torch.from_numpy(np.array(edges['train']['all']['label_index_1']))], dim=0)
    #     test_data[self.srna, self.srna_to_mrna, self.mrna].edge_index = test_edge_index
    #     test_data = T.ToUndirected()(test_data)
    #     # test edges for supervision
    #     test_data[self.srna, self.srna_to_mrna, self.mrna].edge_label = \
    #         torch.from_numpy(np.array(edges['test']['label'])).float()
    #     test_data[self.srna, self.srna_to_mrna, self.mrna].edge_label_index = \
    #         torch.stack([torch.from_numpy(np.array(edges['test']['label_index_0'])),
    #                      torch.from_numpy(np.array(edges['test']['label_index_1']))], dim=0)
    #
    #     # ------  Val Data  ------
    #     val_data = HeteroData()
    #     # ------  Nodes
    #     val_data[self.srna].node_id = torch.arange(len(self.srna_nodes))
    #     val_data[self.mrna].node_id = torch.arange(len(self.mrna_nodes))
    #     val_data[self.mrna].x = None
    #     # ------  Edges
    #     # all train edges for message passing
    #     val_edge_index = torch.stack([torch.from_numpy(np.array(edges['train']['all']['label_index_0'])),
    #                                   torch.from_numpy(np.array(edges['train']['all']['label_index_1']))], dim=0)
    #     val_data[self.srna, self.srna_to_mrna, self.mrna].edge_index = val_edge_index
    #     val_data = T.ToUndirected()(val_data)
    #     # val edges for supervision
    #     val_data[self.srna, self.srna_to_mrna, self.mrna].edge_label = \
    #         torch.from_numpy(np.array(edges['val']['label'])).float()
    #     val_data[self.srna, self.srna_to_mrna, self.mrna].edge_label_index = \
    #         torch.stack([torch.from_numpy(np.array(edges['val']['label_index_0'])),
    #                      torch.from_numpy(np.array(edges['val']['label_index_1']))], dim=0)
    #
    #     if self.debug_logs:
    #         logger.debug(f"\n Train data:\n ============== \n{train_data}\n"
    #                      f"\n Test data:\n ============== \n{test_data}\n"
    #                      f"\n Val data:\n ============== \n{val_data}\n")
    #
    #     return train_data, test_data, val_data


    def _init_train_test_hetero_data(self, unq_train: pd.DataFrame, unq_test: pd.DataFrame, train_neg_sampling: bool) \
            -> (HeteroData, HeteroData):
        """

        Parameters
        ----------
        unq_train
        unq_test
        train_neg_sampling

        Returns
        -------

        """
        logger.debug(f"initializing train and test hetero data - mRNA-mRNA Similarity = {self.add_mrna_mrna_similarity_edges}")
        df = unq_train
        unq_train_pos, unq_train_neg = self._pos_neg_split(df=unq_train, binary_label_col=self.binary_intr_label_col)
        # 1 - random negative sampling - train
        if train_neg_sampling and not self.train_w_loader:
            # 1.1 - take only positive samples from train and add random negatives
            # todo - future - handle given negative train edges (unq_train_neg)
            _shuffle_train = True
            df = self._add_neg_samples(unq_intr_pos=unq_train_pos, ratio=self.train_neg_sampling_ratio_data,
                                       _shuffle=_shuffle_train)
            # NEW
            # srna_meta = self.srna_nodes.rename(columns={c: f"sRNA_{c}" for c in self.srna_nodes.columns.values if c != self.srna_nid_col})
            # mrna_meta = self.mrna_nodes.rename(columns={c: f"mRNA_{c}" for c in self.mrna_nodes.columns.values if c != self.mrna_nid_col})
            # df_to_save = df[[self.srna_nid_col, self.mrna_nid_col, self.binary_intr_label_col]].copy()
            # df_to_save = pd.merge(df_to_save, srna_meta, on=self.srna_nid_col, how='left')
            # df_to_save = pd.merge(df_to_save, mrna_meta, on=self.mrna_nid_col, how='left')
            # assert len(df_to_save) == len(df)
            # df_to_save['sRNA_accession_id_Eco'] = df_to_save['sRNA_EcoCyc_accession_id']
            # df_to_save['mRNA_accession_id_Eco'] = df_to_save['mRNA_EcoCyc_accession_id']
            # ___path = '/home/shanisa/PhD/Data/models_training_and_benchmarking/outputs'
            # write_df(df=df_to_save, file_path=join(___path, f'train_for_GraphRNA.pickle'))
            # write_df(df=df_to_save, file_path=join(___path, f'train_for_GraphRNA.csv'))
            # NEW
            _shuffle_test = True
            if _shuffle_test:
                unq_test = pd.DataFrame(shuffle(unq_test)).reset_index(drop=True)
                # NEW
                # ___path = '/home/shanisa/PhD/Data/models_training_and_benchmarking/outputs'
                # write_df(df=unq_test, file_path=join(___path, f'all_test_for_GraphRNA.pickle'))
                # write_df(df=unq_test, file_path=join(___path, f'all_test_for_GraphRNA.csv'))
                # NEW
            if self.debug_logs:
                logger.debug(f"_shuffle_train = {_shuffle_train}, _shuffle_test = {_shuffle_test}")

        # 2 - split train edges into message passing & supervision
        unq_train_spr, unq_train_mp = split_df_samples(df=df, ratio=self.train_supervision_ratio)
        edges = {
            'train': {
                'all': {
                    'label': list(df[self.binary_intr_label_col]),
                    'label_index_0': list(df[self.srna_nid_col]),
                    'label_index_1': list(df[self.mrna_nid_col])
                },
                'message_passing': {
                    'label': list(unq_train_mp[self.binary_intr_label_col]),
                    'label_index_0': list(unq_train_mp[self.srna_nid_col]),
                    'label_index_1': list(unq_train_mp[self.mrna_nid_col])
                },
                'supervision': {
                    'label': list(unq_train_spr[self.binary_intr_label_col]),
                    'label_index_0': list(unq_train_spr[self.srna_nid_col]),
                    'label_index_1': list(unq_train_spr[self.mrna_nid_col])
                }
            },
            'test': {
                'label': list(unq_test[self.binary_intr_label_col]),
                'label_index_0': list(unq_test[self.srna_nid_col]),
                'label_index_1': list(unq_test[self.mrna_nid_col])
            }
        }

        logger.debug(f"\n{len(self.srna_nodes)} sRNA nodes, {len(self.mrna_nodes)} mRNA nodes \n"
                     f"Train: {len(edges['train']['all']['label'])} interactions, "
                     f"P: {sum(edges['train']['all']['label'])}, "
                     f"N: {len(edges['train']['all']['label']) - sum(edges['train']['all']['label'])} \n"
                     f"Test: {len(edges['test']['label'])} interactions, "
                     f"P: {sum(edges['test']['label'])}, "
                     f"N: {len(edges['test']['label']) - sum(edges['test']['label'])}")

        # 3 - initialize data sets
        train_data, test_data = self._generate_train_and_test(edges=edges)

        return train_data, test_data
    #
    # def _init_train_test_val_hetero_data(self, unq_train: pd.DataFrame, unq_test: pd.DataFrame, train_neg_sampling: bool) \
    #         -> (HeteroData, HeteroData, HeteroData):
    #     """
    #
    #     Parameters
    #     ----------
    #     unq_train
    #     unq_test
    #     train_neg_sampling
    #
    #     Returns
    #     -------
    #
    #     """
    #     logger.debug("initializing train, test and val hetero data")
    #     df = unq_train
    #     unq_train_pos, unq_train_neg = self._pos_neg_split(df=unq_train, binary_label_col=self.binary_intr_label_col)
    #     # 1 - random negative sampling - train
    #     if train_neg_sampling and not self.train_w_loader:
    #         # 1.1 - take only positive samples from train and add random negatives
    #         # todo - future - handle given negative train edges (unq_train_neg)
    #         _shuffle_train = True
    #         df = self._add_neg_samples(unq_intr_pos=unq_train_pos, ratio=self.train_neg_sampling_ratio_data,
    #                                    _shuffle=_shuffle_train)
    #         _shuffle_test = True
    #         if _shuffle_test:
    #             unq_test = pd.DataFrame(shuffle(unq_test)).reset_index(drop=True)
    #         if self.debug_logs:
    #             logger.debug(f"_shuffle_train = {_shuffle_train}, _shuffle_test = {_shuffle_test}")
    #     # ----- NEW
    #     val_size = int(len(df) * 0.1)
    #     df = df[:-val_size].copy()
    #     val_df = df[-val_size:].copy().reset_index(drop=True)
    #     # ----- NEW
    #
    #     # 2 - split train edges into message passing & supervision
    #     unq_train_spr, unq_train_mp = split_df_samples(df=df, ratio=self.train_supervision_ratio)
    #     edges = {
    #         'train': {
    #             'all': {
    #                 'label': list(df[self.binary_intr_label_col]),
    #                 'label_index_0': list(df[self.srna_nid_col]),
    #                 'label_index_1': list(df[self.mrna_nid_col])
    #             },
    #             'message_passing': {
    #                 'label': list(unq_train_mp[self.binary_intr_label_col]),
    #                 'label_index_0': list(unq_train_mp[self.srna_nid_col]),
    #                 'label_index_1': list(unq_train_mp[self.mrna_nid_col])
    #             },
    #             'supervision': {
    #                 'label': list(unq_train_spr[self.binary_intr_label_col]),
    #                 'label_index_0': list(unq_train_spr[self.srna_nid_col]),
    #                 'label_index_1': list(unq_train_spr[self.mrna_nid_col])
    #             }
    #         },
    #         'test': {
    #             'label': list(unq_test[self.binary_intr_label_col]),
    #             'label_index_0': list(unq_test[self.srna_nid_col]),
    #             'label_index_1': list(unq_test[self.mrna_nid_col])
    #         },
    #         'val': {
    #             'label': list(val_df[self.binary_intr_label_col]),
    #             'label_index_0': list(val_df[self.srna_nid_col]),
    #             'label_index_1': list(val_df[self.mrna_nid_col])
    #         },
    #
    #     }
    #
    #     logger.debug(f"\n{len(self.srna_nodes)} sRNA nodes, {len(self.mrna_nodes)} mRNA nodes \n"
    #                  f"Train: {len(edges['train']['all']['label'])} interactions, "
    #                  f"P: {sum(edges['train']['all']['label'])}, "
    #                  f"N: {len(edges['train']['all']['label']) - sum(edges['train']['all']['label'])} \n"
    #                  f"Test: {len(edges['test']['label'])} interactions, "
    #                  f"P: {sum(edges['test']['label'])}, "
    #                  f"N: {len(edges['test']['label']) - sum(edges['test']['label'])} \n"
    #                  f"Val: {len(edges['val']['label'])} interactions, "
    #                  f"P: {sum(edges['val']['label'])}, "
    #                  f"N: {len(edges['val']['label']) - sum(edges['val']['label'])}")
    #
    #     # 3 - initialize data sets
    #     train_data, test_data, val_data = self._generate_train_and_test_and_val(edges=edges)
    #
    #     return train_data, test_data, val_data
    #
    # def _add_mrna_mrna_similarity_edges(self, data: HeteroData) -> HeteroData:
    #     logger.debug("adding mRNA mRNA similarity edges")
    #     min_score = 0.5
    #     mrna_mrna_edges = self.mrna_mrna_edges[self.mrna_mrna_edges[self.mrna_mrna_val_col] > min_score].reset_index(drop=True)
    #     m1 = torch.from_numpy(np.array(mrna_mrna_edges[self.mrna_1_nid_col]))
    #     m2 = torch.from_numpy(np.array(mrna_mrna_edges[self.mrna_2_nid_col]))
    #     w = torch.from_numpy(np.array(mrna_mrna_edges[self.mrna_mrna_val_col])).float()
    #
    #     # all edges are used for message passing
    #     # direction
    #     data[self.mrna, self.mrna_to_mrna, self.mrna].edge_index = torch.stack([m1, m2], dim=0)
    #     data[self.mrna, self.mrna_to_mrna, self.mrna].edge_attr = w
    #     # reversed
    #     data[self.mrna, f"rev_{self.mrna_to_mrna}", self.mrna].edge_index = torch.stack([m2, m1], dim=0)
    #     data[self.mrna, f"rev_{self.mrna_to_mrna}", self.mrna].edge_attr = w
    #
    #     return data
    #
    # def _get_train_mini_batches_loader(self, train_data: HeteroData) -> LinkNeighborLoader:
    #     # Define seed edges:
    #     edge_label_index = train_data[self.srna, self.srna_to_mrna, self.mrna].edge_label_index
    #     edge_label = train_data[self.srna, self.srna_to_mrna, self.mrna].edge_label
    #
    #     _train_check = pd.DataFrame({'edge_label': edge_label.cpu().numpy(), "edge_label_index_0": edge_label_index[0].cpu().numpy(), "edge_label_index_1": edge_label_index[1].cpu().numpy()})
    #
    #     train_loader = LinkNeighborLoader(  # todo - convert to regular dataloader
    #         data=train_data,
    #         num_neighbors=self.train_num_neighbors,
    #         neg_sampling_ratio=self.train_neg_sampling_ratio_loader,
    #         edge_label_index=((self.srna, self.srna_to_mrna, self.mrna), edge_label_index),
    #         edge_label=edge_label,
    #         batch_size=self.train_batch_size,
    #         shuffle=self.train_shuffle,
    #     )
    #
    #     for sampled_data in tqdm.tqdm(train_loader):
    #         s_edge_label_index = sampled_data[self.srna, self.srna_to_mrna, self.mrna].edge_label_index
    #         s_edge_label = sampled_data[self.srna, self.srna_to_mrna, self.mrna].edge_label
    #         _s_check = pd.DataFrame(
    #             {'edge_label': s_edge_label.cpu().numpy(), "edge_label_index_0": s_edge_label_index[0].cpu().numpy(),
    #              "edge_label_index_1": s_edge_label_index[1].cpu().numpy()})
    #
    #         ground_truth = sampled_data[self.srna, self.srna_to_mrna, self.mrna].edge_label
    #         print()
    #
    #     return train_loader
    #
    # def _train_hgnn_with_loader(self, hg_model: kGraphRNA, train_data: HeteroData, model_args: dict) -> kGraphRNA:
    #     logger.debug(f"training kGraphRNA model with loader")
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     if self.debug_logs:
    #         logger.debug(f"Device: '{device}'")
    #
    #     train_loader = self._get_train_mini_batches_loader(train_data=train_data)
    #     hg_model = hg_model.to(device)
    #     optimizer = torch.optim.Adam(hg_model.parameters(), lr=model_args['learning_rate'])
    #
    #     for epoch in range(1, model_args['epochs']):
    #         total_loss = total_examples = 0
    #         for sampled_data in tqdm.tqdm(train_loader):
    #             optimizer.zero_grad()
    #
    #             sampled_data.to(device)
    #             pred = hg_model(sampled_data)
    #
    #             ground_truth = sampled_data[self.srna, self.srna_to_mrna, self.mrna].edge_label
    #             loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    #
    #             loss.backward()
    #             optimizer.step()
    #             total_loss += float(loss) * pred.numel()
    #             total_examples += pred.numel()
    #         if self.debug_logs:
    #             logger.debug(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
    #
    #     return hg_model
    #
    #
    # def _train_hgnn(self, hga_model: kGraphRNA, train_data: HeteroData, model_args: dict) -> kGraphRNA:
    #     logger.debug(f"training kGraphRNA model (without loader) -> epochs = {model_args['epochs']}")
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     if self.debug_logs:
    #         logger.debug(f"Device: '{device}'")
    #
    #     hga_model = hga_model.to(device)
    #     optimizer = torch.optim.Adam(hga_model.parameters(), lr=model_args['learning_rate'], weight_decay=0.001)
    #     # todo - add training history
    #     training_history = {}
    #     for epoch in range(1, model_args['epochs']):
    #         total_loss = total_examples = 0
    #         optimizer.zero_grad()
    #
    #         train_data.to(device)
    #         pred = hga_model(train_data, model_args)
    #
    #         ground_truth = train_data[self.srna, self.srna_to_mrna, self.mrna].edge_label
    #         loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    #
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += float(loss) * pred.numel()
    #         total_examples += pred.numel()
    #
    #         if self.debug_logs:
    #             logger.debug(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
    #
    #     return hga_model
    #
    # def _get_eval_mini_batches_loader(self, eval_data: HeteroData) -> LinkNeighborLoader:
    #     # Define the validation seed edges:
    #     edge_label_index = eval_data[self.srna, self.srna_to_mrna, self.mrna].edge_label_index
    #     edge_label = eval_data[self.srna, self.srna_to_mrna, self.mrna].edge_label
    #
    #     # Define hyper-params
    #     bs_const = self.val_and_test_neg_sampling_ratio + 1 if self.random_split else 1
    #
    #     eval_loader = LinkNeighborLoader(
    #         data=eval_data,
    #         num_neighbors=self.eval_num_neighbors,
    #         edge_label_index=((self.srna, self.srna_to_mrna, self.mrna), edge_label_index),
    #         edge_label=edge_label,
    #         batch_size=int(bs_const * self.eval_batch_size),
    #         shuffle=self.eval_shuffle,
    #     )
    #
    #     sampled_data = next(iter(eval_loader))
    #     i1 = sampled_data[self.srna, self.srna_to_mrna, self.mrna].edge_index[0]
    #     i2 = sampled_data[self.srna, self.srna_to_mrna, self.mrna].edge_index[1]
    #     lbl = sampled_data[self.srna, self.srna_to_mrna, self.mrna].edge_label
    #     test_df = pd.DataFrame({
    #         'edge_label_index_0': torch.cat(i1, dim=0).cpu().numpy(),
    #         'edge_label_index_1': torch.cat(i2, dim=0).cpu().numpy(),
    #         'edge_label_': torch.cat(lbl, dim=0).cpu().numpy()
    #     })
    #
    #     return eval_loader
    #
    # def _eval_dataset_w_loader(self, trained_model: kGraphRNA, dataset_loader: LinkNeighborLoader,
    #                            dataset_nm: str = None) -> (Dict[str, object], pd.DataFrame):
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     logger.debug(f"Device: '{device}'")
    #     # 1 - predict on dataset
    #     preds = []
    #     ground_truths = []
    #     srna_nids, mrna_nids = [], []
    #     for sampled_data in tqdm.tqdm(dataset_loader):
    #         with torch.no_grad():
    #             sampled_data.to(device)
    #             preds.append(trained_model(sampled_data))
    #             ground_truths.append(sampled_data[self.srna, self.srna_to_mrna, self.mrna].edge_label)
    #             srna_nids.append(sampled_data[self.srna, self.srna_to_mrna, self.mrna].edge_label_index[0])
    #             mrna_nids.append(sampled_data[self.srna, self.srna_to_mrna, self.mrna].edge_label_index[1])
    #     # 2 - process outputs
    #     scores_arr = torch.cat(preds, dim=0).cpu().numpy()
    #     ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    #     y_true = list(ground_truth)
    #     # 2.1 - scaling scores_arr to [0 ,1]
    #     minmax_scaler = MinMaxScaler()
    #     y_score = minmax_scaler.fit_transform(scores_arr.reshape(-1, 1)).reshape(-1)
    #     # 2.2 - save
    #     eval_data_preds = pd.DataFrame({
    #         self.srna_nid_col: torch.cat(srna_nids, dim=0).cpu().numpy(),
    #         self.mrna_nid_col: torch.cat(mrna_nids, dim=0).cpu().numpy(),
    #         'y_original_score': list(scores_arr),
    #         'y_score': y_score,
    #         'y_true': y_true
    #     })
    #     eval_data_preds = eval_data_preds.sort_values(by=[self.srna_nid_col, self.mrna_nid_col])
    #
    #     # 3 - calc evaluation metrics
    #     scores = calc_binary_classification_metrics_using_y_score(y_true=y_true, y_score=y_score, dataset_nm=dataset_nm)
    #     if self.debug_logs:
    #         for k, v in scores.items():
    #             # print(f"{k}: {v:.2f}")
    #             print(f"{k}: {v}")
    #
    #     return scores, eval_data_preds
    #
    # def _eval_hgnn_w_loader(self, trained_model: kGraphRNA, eval_data: HeteroData, dataset_nm: str = None) -> \
    #         (Dict[str, object], pd.DataFrame):
    #     eval_loader = self._get_eval_mini_batches_loader(eval_data=eval_data)
    #     scores, eval_data_preds = self._eval_dataset_w_loader(trained_model=trained_model, dataset_loader=eval_loader,
    #                                                           dataset_nm=dataset_nm)
    #     return scores, eval_data_preds
    #
    # def _predict_hgnn(self, trained_model: kGraphRNA, eval_data: HeteroData, model_args: dict = None,
    #                   dataset_nm: str = None, **kwargs) -> pd.DataFrame:
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     if self.debug_logs:
    #         logger.debug(f"Device: '{device}'")
    #     # 1 - predict on dataset
    #     preds = []
    #     # ground_truths = []
    #     srna_nids, mrna_nids = [], []
    #     with torch.no_grad():
    #         eval_data.to(device)
    #         preds.append(trained_model(eval_data, model_args))
    #
    #         # ground_truths.append(eval_data[self.srna, self.srna_to_mrna, self.mrna].edge_label)
    #         srna_nids.append(eval_data[self.srna, self.srna_to_mrna, self.mrna].edge_label_index[0])
    #         mrna_nids.append(eval_data[self.srna, self.srna_to_mrna, self.mrna].edge_label_index[1])
    #     # 2 - process outputs
    #     scores_arr = torch.cat(preds, dim=0).cpu().numpy()
    #     # ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    #     # y_true = list(ground_truth)
    #     # 2.1 - scaling scores_arr to [0 ,1]
    #     minmax_scaler = MinMaxScaler()
    #     y_score = minmax_scaler.fit_transform(scores_arr.reshape(-1, 1)).reshape(-1)
    #     # 2.2 - save
    #     eval_data_preds = pd.DataFrame({
    #         self.srna_nid_col: torch.cat(srna_nids, dim=0).cpu().numpy(),
    #         self.mrna_nid_col: torch.cat(mrna_nids, dim=0).cpu().numpy(),
    #         'y_original_score': list(scores_arr),
    #         'y_score': y_score
    #     })
    #     eval_data_preds = eval_data_preds.sort_values(by='y_score', ascending=False).reset_index(drop=True)
    #
    #     return eval_data_preds
    #
    # @torch.no_grad()
    # def _test(self, model, device, model_args, eval_data) -> (float, float):
    #     model.eval()
    #     # 1 - predict on dataset
    #     preds = []
    #     ground_truths = []
    #     srna_nids, mrna_nids = [], []
    #     with torch.no_grad():
    #         eval_data.to(device)
    #         pred = model(eval_data, model_args)
    #         ground_truth = eval_data['srna', 'targets', 'mrna'].edge_label
    #         loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    #         out_loss = loss.detach().cpu().numpy().tolist()
    #         preds.append(pred)
    #         ground_truths.append(ground_truth)
    #
    #     scores_arr = list(torch.cat(preds, dim=0).cpu().numpy())
    #     labels_arr = list(torch.cat(ground_truths, dim=0).cpu().numpy())
    #     if len(set(labels_arr)) == 1:
    #         print()
    #     roc_auc = roc_auc_score(labels_arr, scores_arr)
    #
    #     return out_loss, roc_auc
    #
    def _eval_hgnn(self, trained_model: kGraphRNA, eval_data: HeteroData, model_args: dict = None,
                   dataset_nm: str = None, avoid_scores: bool = False, **kwargs) -> (Dict[str, object], pd.DataFrame):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.debug_logs:
            logger.debug(f"Device: '{device}'")
        # 1 - predict on dataset
        preds = []
        ground_truths = []
        srna_nids, mrna_nids = [], []
        with torch.no_grad():
            eval_data.to(device)
            pred = trained_model(
                eval_data.x_dict,
                eval_data.edge_index_dict,
                eval_data['srna', 'mrna'].edge_label_index,
            )
            pred = pred.sigmoid().view(-1).cpu()
            preds.append(pred)
            ground_truths.append(eval_data[self.srna, self.srna_to_mrna, self.mrna].edge_label)
            srna_nids.append(eval_data[self.srna, self.srna_to_mrna, self.mrna].edge_label_index[0])
            mrna_nids.append(eval_data[self.srna, self.srna_to_mrna, self.mrna].edge_label_index[1])
        # 2 - process outputs
        scores_arr = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        y_true = list(ground_truth)
        # 2.1 - scaling scores_arr to [0 ,1]
        # minmax_scaler = MinMaxScaler()
        # y_score = minmax_scaler.fit_transform(scores_arr.reshape(-1, 1)).reshape(-1)
        # y_original_score = list(scores_arr)
        y_score = list(scores_arr)
        # 2.2 - save
        eval_data_preds = pd.DataFrame({
            self.srna_nid_col: torch.cat(srna_nids, dim=0).cpu().numpy(),
            self.mrna_nid_col: torch.cat(mrna_nids, dim=0).cpu().numpy(),
            'y_score': y_score,
            'y_true': y_true
        })
        scores = None

        return scores, eval_data_preds
    #
    #
    #
    # def train_val_and_test(self, min_epochs: int, X_train: pd.DataFrame, y_train: List[int], X_test: pd.DataFrame, y_test: List[int],
    #                        model_args: dict, metadata_train: pd.DataFrame, metadata_test: pd.DataFrame,
    #                        unq_train: pd.DataFrame = None, unq_test: pd.DataFrame = None,
    #                        train_neg_sampling: bool = True, srna_acc_col: str = 'sRNA_accession_id_Eco',
    #                        mrna_acc_col: str = 'mRNA_accession_id_Eco', is_syn_col: str = 'is_synthetic', **kwargs) -> \
    #         (Dict[str, dict], Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object]):
    #     """
    #     torch_geometric.__version__ = '2.1.0'
    #
    #     Parameters
    #     ----------
    #     min_epochs: int
    #     X_train: pd.DataFrame (n_samples, N_features),
    #     y_train: list (n_samples,),
    #     X_test: pd.DataFrame (t_samples, N_features),
    #     y_test: list (t_samples,)
    #     model_args: Dict of model's constructor and fit() arguments
    #     metadata_train: pd.DataFrame (n_samples, T_features)
    #     metadata_test: pd.DataFrame (t_samples, T_features)
    #     unq_train: pd.DataFrame (_samples, T_features)
    #     unq_test: pd.DataFrame (t_samples, T_features)
    #     train_neg_sampling: whether to add random negative sampling to train HeteroData
    #     srna_acc_col: str  sRNA EcoCyc accession id col in metadata_train and metadata_test
    #     mrna_acc_col: str  mRNA EcoCyc accession id col in metadata_train and metadata_test
    #     is_syn_col: is synthetic indicator col in metadata_train
    #     kwargs:
    #         mrna_eco: all mRNA metadata from EcoCyc
    #         mrna_similarity: known similarity score between mRNAs (id = EcoCyc accession id)
    #
    #     Returns
    #     -------
    #
    #     scores - Dict in the following format:  {
    #         'test': {
    #             <score_nm>: score
    #             }
    #     }
    #     predictions - Dict in the following format:  {
    #         'test_pred': array-like (t_samples,)  - ordered as y test
    #     }
    #     training_history - Dict in the following format:  {
    #         'train': OrderedDict
    #         'validation': OrderedDict
    #     }
    #     train_val_data - Dict in the following format:  {
    #         "X_train": pd.DataFrame (n_samples, N_features),
    #         "y_train": list (n_samples,),
    #         "X_val": pd.DataFrame (k_samples, K_features),
    #         "y_val": list (k_samples,),
    #         "metadata_train": pd.DataFrame (n_samples, T_features),    - Optional (in case metadata_train is not None)
    #         "metadata_val": list (k_samples,)    - Optional (in case metadata_train is not None)
    #     }
    #     shap_args - Dict in the following format:  {
    #         "model_nm": str,
    #         "trained_model": the trained machine learning model,
    #         "X_train": pd.DataFrame,
    #         "X_val": Optional[pd.DataFrame],
    #         "X_test": pd.DataFrame
    #     }
    #     """
    #     logger.debug(f"training an kGraphRNA model w val  ->  train negative sampling = {train_neg_sampling}")
    #     # 1 - define graph nodes
    #     if not self.nodes_are_defined:
    #         self._define_nodes_and_features(**kwargs)
    #
    #     # if not cv
    #     if unq_train is None or unq_test is None:
    #         out_test_pred = pd.DataFrame({
    #             srna_acc_col: metadata_test[srna_acc_col],
    #             mrna_acc_col: metadata_test[mrna_acc_col]
    #         })
    #         # 2 - remove synthetic data from train
    #         if sum(metadata_train[is_syn_col]) > 0:
    #             logger.warning("removing synthetic samples from train")
    #             X_train, y_train, metadata_train = \
    #                 self._remove_synthetic_samples(X=X_train, y=y_train, metadata=metadata_train, is_syn_col=is_syn_col)
    #
    #         # 3 - get unique interactions data (train and test)
    #         unq_train = self._get_unique_inter(metadata=metadata_train, y=y_train, srna_acc_col=srna_acc_col,
    #                                            mrna_acc_col=mrna_acc_col, df_nm='train')
    #         unq_test = self._get_unique_inter(metadata=metadata_test, y=y_test, srna_acc_col=srna_acc_col,
    #                                           mrna_acc_col=mrna_acc_col, df_nm='test')
    #
    #         # 4 - assert no data leakage between train and test
    #         self._assert_no_data_leakage(unq_train=unq_train, unq_test=unq_test, srna_acc_col=srna_acc_col,
    #                                      mrna_acc_col=mrna_acc_col)
    #
    #         # 5 - map interactions to edges
    #         unq_train = self._map_interactions_to_edges(unique_intr=unq_train, srna_acc_col=srna_acc_col,
    #                                                     mrna_acc_col=mrna_acc_col)
    #         unq_test = self._map_interactions_to_edges(unique_intr=unq_test, srna_acc_col=srna_acc_col,
    #                                                    mrna_acc_col=mrna_acc_col)
    #         # 4.1 - update output df
    #         _len = len(out_test_pred)
    #         out_test_pred = pd.merge(out_test_pred, unq_test, on=[srna_acc_col, mrna_acc_col], how='left')
    #         assert len(out_test_pred) == _len
    #     else:
    #         out_test_pred = pd.DataFrame({
    #             self.srna_nid_col: unq_test[self.srna_nid_col],
    #             self.mrna_nid_col: unq_test[self.mrna_nid_col]
    #         })
    #     # 5 - init train & test sets (HeteroData)
    #     train_data, test_data, val_data = self._init_train_test_val_hetero_data(unq_train=unq_train, unq_test=unq_test,
    #                                                                             train_neg_sampling=train_neg_sampling)
    #
    #     # insert node features
    #     train_data[self.srna].x = torch.from_numpy(np.array(self.srna_nodes[self.srna_feat_cols])).to(torch.float32)
    #     train_data[self.mrna].x = torch.from_numpy(np.array(self.mrna_nodes[self.mrna_feat_cols])).to(torch.float32)
    #     test_data[self.srna].x = torch.from_numpy(np.array(self.srna_nodes[self.srna_feat_cols])).to(torch.float32)
    #     test_data[self.mrna].x = torch.from_numpy(np.array(self.mrna_nodes[self.mrna_feat_cols])).to(torch.float32)
    #     val_data[self.srna].x = torch.from_numpy(np.array(self.srna_nodes[self.srna_feat_cols])).to(torch.float32)
    #     val_data[self.mrna].x = torch.from_numpy(np.array(self.mrna_nodes[self.mrna_feat_cols])).to(torch.float32)
    #
    #     # 6 - add mRNA-mRNA similarity data to train and test
    #     if kwargs['model_nm'] == 'HeteroGraph_Feat_w_sim':
    #         model_args['add_sim'] = True
    #         # model_args['epochs'] = 50
    #         train_data = self._add_mrna_mrna_similarity_edges(data=train_data)
    #         test_data = self._add_mrna_mrna_similarity_edges(data=test_data)
    #     else:
    #         model_args['add_sim'] = False
    #
    #     # 7 - init kGraphRNA model
    #     # 7.1 - init model
    #     model_args['hidden_channels'] = 32
    #     model = kGraphRNA(hidden_channels=model_args['hidden_channels'],
    #                       srna=self.srna, mrna=self.mrna, srna_to_mrna=self.srna_to_mrna)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #
    #     # 8 - train kGraphRNA model
    #     epochs_arr, train_loss_arr, val_loss_arr, val_loss_diff_arr, val_auc_arr, test_loss_arr = [], [], [], [], [], []
    #     keep_run = True
    #     epoch = 1
    #     # for epoch in range(1, 120):
    #     while keep_run:
    #         # --- train
    #         model.train()
    #         preds, targets = [], []
    #         optimizer.zero_grad()
    #         pred = model(
    #             train_data.x_dict,
    #             train_data.edge_index_dict,
    #             train_data[self.srna, self.mrna].edge_label_index,
    #         )
    #         target = train_data[self.srna, self.mrna].edge_label
    #         loss = F.binary_cross_entropy_with_logits(pred, target)
    #         loss.backward()
    #         optimizer.step()
    #
    #         preds.append(pred)
    #         targets.append(target)
    #         scores_arr = list(torch.cat(preds, dim=0).detach().cpu().numpy())
    #         targets_arr = list(torch.cat(targets, dim=0).detach().cpu().numpy())
    #         roc_auc = roc_auc_score(targets_arr, scores_arr)
    #         # loss
    #         train_loss = loss.detach().cpu().numpy().tolist()
    #
    #         # --- val
    #         val_loss, val_auc = test(model=model, test_data=val_data)
    #         val_loss_diff = val_loss_arr[-1] - val_loss if epoch > 1 else float('inf')
    #
    #         # --- test
    #         test_loss, test_auc = test(model=model, test_data=test_data)
    #         logger.debug(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train AUC: {roc_auc:.4f}, "
    #                      f"Val Loss: {val_loss:.4f}, Val Loss diff: {val_loss_diff:.4f}, Val AUC: {val_auc:.4f}, "
    #                      f"Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")
    #         # - save
    #         epochs_arr.append(epoch)
    #         train_loss_arr.append(train_loss)
    #         val_loss_arr.append(val_loss)
    #         val_loss_diff_arr.append(val_loss_diff)
    #         val_auc_arr.append(val_auc)
    #         test_loss_arr.append(test_loss)
    #         # - check
    #         epoch += 1
    #         # diff_flag = True
    #         diff_flag = val_loss_diff < kwargs['minimal_val_diff'] if pd.notnull(kwargs['minimal_val_diff']) else True
    #         if (epoch >= min_epochs) and diff_flag:
    #             keep_run = False
    #
    #     training_history = {
    #         'epoch': epochs_arr,
    #         'train_loss': train_loss_arr,
    #         'val_loss': val_loss_arr,
    #         'val_loss_diff': val_loss_diff_arr,
    #         'val_auc': val_auc_arr,
    #         'test_loss': test_loss_arr
    #     }
    #
    #     train_val_data = {}
    #
    #     # 9 - evaluate - calc scores
    #     logger.debug("evaluating test set")
    #     predictions, scores = {}, {}
    #     test_scores, test_pred_df = self._eval_hgnn(trained_model=model, eval_data=test_data, model_args=model_args,
    #                                                 **kwargs)
    #     assert pd.isnull(test_pred_df).sum().sum() == 0, "some null predictions"
    #
    #     # 10 - update outputs
    #     _len = len(out_test_pred)
    #     out_test_pred = pd.merge(out_test_pred, test_pred_df, on=[self.srna_nid_col, self.mrna_nid_col], how='left')
    #     assert len(out_test_pred) == _len
    #     test_y_pred = out_test_pred['y_score']
    #
    #     scores.update({'test': test_scores})
    #     predictions.update({'test_pred': test_y_pred,
    #                         'test_pred_df': test_pred_df, 'out_test_pred': out_test_pred})
    #
    #     # 11 - SHAP args
    #     shap_args = {
    #         "model_nm": "HeteroFeatGNN",
    #         "trained_model": model,
    #         "X_train": pd.DataFrame(),
    #         "X_val": None,
    #         "X_test": pd.DataFrame()
    #     }
    #
    #     return scores, predictions, training_history, train_val_data, shap_args
    #
    # #
    # # def train_val_and_test_PREV(self, X_train: pd.DataFrame, y_train: List[int], X_test: pd.DataFrame, y_test: List[int],
    # #                        model_args: dict, metadata_train: pd.DataFrame, metadata_test: pd.DataFrame,
    # #                        unq_train: pd.DataFrame = None, unq_test: pd.DataFrame = None,
    # #                        train_neg_sampling: bool = True, srna_acc_col: str = 'sRNA_accession_id_Eco',
    # #                        mrna_acc_col: str = 'mRNA_accession_id_Eco', is_syn_col: str = 'is_synthetic', **kwargs) -> \
    # #         (Dict[str, dict], Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object]):
    # #     """
    # #     torch_geometric.__version__ = '2.1.0'
    # #
    # #     Parameters
    # #     ----------
    # #     X_train: pd.DataFrame (n_samples, N_features),
    # #     y_train: list (n_samples,),
    # #     X_test: pd.DataFrame (t_samples, N_features),
    # #     y_test: list (t_samples,)
    # #     model_args: Dict of model's constructor and fit() arguments
    # #     metadata_train: pd.DataFrame (n_samples, T_features)
    # #     metadata_test: pd.DataFrame (t_samples, T_features)
    # #     unq_train: pd.DataFrame (_samples, T_features)
    # #     unq_test: pd.DataFrame (t_samples, T_features)
    # #     train_neg_sampling: whether to add random negative sampling to train HeteroData
    # #     srna_acc_col: str  sRNA EcoCyc accession id col in metadata_train and metadata_test
    # #     mrna_acc_col: str  mRNA EcoCyc accession id col in metadata_train and metadata_test
    # #     is_syn_col: is synthetic indicator col in metadata_train
    # #     kwargs:
    # #         mrna_eco: all mRNA metadata from EcoCyc
    # #         mrna_similarity: known similarity score between mRNAs (id = EcoCyc accession id)
    # #
    # #     Returns
    # #     -------
    # #
    # #     scores - Dict in the following format:  {
    # #         'test': {
    # #             <score_nm>: score
    # #             }
    # #     }
    # #     predictions - Dict in the following format:  {
    # #         'test_pred': array-like (t_samples,)  - ordered as y test
    # #     }
    # #     training_history - Dict in the following format:  {
    # #         'train': OrderedDict
    # #         'validation': OrderedDict
    # #     }
    # #     train_val_data - Dict in the following format:  {
    # #         "X_train": pd.DataFrame (n_samples, N_features),
    # #         "y_train": list (n_samples,),
    # #         "X_val": pd.DataFrame (k_samples, K_features),
    # #         "y_val": list (k_samples,),
    # #         "metadata_train": pd.DataFrame (n_samples, T_features),    - Optional (in case metadata_train is not None)
    # #         "metadata_val": list (k_samples,)    - Optional (in case metadata_train is not None)
    # #     }
    # #     shap_args - Dict in the following format:  {
    # #         "model_nm": str,
    # #         "trained_model": the trained machine learning model,
    # #         "X_train": pd.DataFrame,
    # #         "X_val": Optional[pd.DataFrame],
    # #         "X_test": pd.DataFrame
    # #     }
    # #     """
    # #     logger.debug(f"training an kGraphRNA model w val  ->  train negative sampling = {train_neg_sampling}")
    # #     # 1 - define graph nodes
    # #     if not self.nodes_are_defined:
    # #         self._define_nodes_and_features(**kwargs)
    # #
    # #     # if not cv
    # #     if unq_train is None or unq_test is None:
    # #         out_test_pred = pd.DataFrame({
    # #             srna_acc_col: metadata_test[srna_acc_col],
    # #             mrna_acc_col: metadata_test[mrna_acc_col]
    # #         })
    # #         # 2 - remove synthetic data from train
    # #         if sum(metadata_train[is_syn_col]) > 0:
    # #             logger.warning("removing synthetic samples from train")
    # #             X_train, y_train, metadata_train = \
    # #                 self._remove_synthetic_samples(X=X_train, y=y_train, metadata=metadata_train, is_syn_col=is_syn_col)
    # #
    # #         # 3 - get unique interactions data (train and test)
    # #         unq_train = self._get_unique_inter(metadata=metadata_train, y=y_train, srna_acc_col=srna_acc_col,
    # #                                            mrna_acc_col=mrna_acc_col, df_nm='train')
    # #         unq_test = self._get_unique_inter(metadata=metadata_test, y=y_test, srna_acc_col=srna_acc_col,
    # #                                           mrna_acc_col=mrna_acc_col, df_nm='test')
    # #
    # #         # 4 - assert no data leakage between train and test
    # #         self._assert_no_data_leakage(unq_train=unq_train, unq_test=unq_test, srna_acc_col=srna_acc_col,
    # #                                      mrna_acc_col=mrna_acc_col)
    # #
    # #         # 5 - map interactions to edges
    # #         unq_train = self._map_interactions_to_edges(unique_intr=unq_train, srna_acc_col=srna_acc_col,
    # #                                                     mrna_acc_col=mrna_acc_col)
    # #         unq_test = self._map_interactions_to_edges(unique_intr=unq_test, srna_acc_col=srna_acc_col,
    # #                                                    mrna_acc_col=mrna_acc_col)
    # #         # 4.1 - update output df
    # #         _len = len(out_test_pred)
    # #         out_test_pred = pd.merge(out_test_pred, unq_test, on=[srna_acc_col, mrna_acc_col], how='left')
    # #         assert len(out_test_pred) == _len
    # #     else:
    # #         out_test_pred = pd.DataFrame({
    # #             self.srna_nid_col: unq_test[self.srna_nid_col],
    # #             self.mrna_nid_col: unq_test[self.mrna_nid_col]
    # #         })
    # #     # 5 - init train & test sets (HeteroData)
    # #     train_data, test_data, val_data = self._init_train_test_val_hetero_data(unq_train=unq_train, unq_test=unq_test,
    # #                                                                             train_neg_sampling=train_neg_sampling)
    # #     # 6 - add mRNA-mRNA similarity data to train and test
    # #     if kwargs['model_nm'] == 'kGraphRNA_w_sim':
    # #         model_args['add_sim'] = True
    # #         model_args['epochs'] = 50
    # #         train_data = self._add_mrna_mrna_similarity_edges(data=train_data)
    # #         test_data = self._add_mrna_mrna_similarity_edges(data=test_data)
    # #     else:
    # #         model_args['add_sim'] = False
    # #     # 7 - init kGraphRNA model
    # #     # 7.1 - set params
    # #     srna_num_emb = len(self.srna_nodes)
    # #     mrna_num_emb = len(self.mrna_nodes)
    # #     metadata = train_data.metadata()
    # #     # todo ---------------------------------------
    # #     # 7.2 - init model
    # #     hga_model = HeteroFeatGNN(srna=self.srna, mrna=self.mrna, srna_to_mrna=self.srna_to_mrna,
    # #                               srna_num_embeddings=srna_num_emb, mrna_num_embeddings=mrna_num_emb,
    # #                               metadata=metadata, model_args=model_args, **kwargs)
    # #     if self.debug_logs:
    # #         logger.debug(hga_model)
    # #
    # #     # 8 - train kGraphRNA model
    # #     if self.train_w_loader:
    # #         hg_model = self._train_val_hgnn_with_loader(hg_model=hga_model, train_data=train_data, model_args=model_args)
    # #         training_history = {}
    # #     else:
    # #         # todo - consider to also implement mini batches
    # #         hg_model, training_history = self._train_val_hgnn(hga_model=hga_model, train_data=train_data, val_data=val_data,
    # #                                                           model_args=model_args, **kwargs)
    # #     train_val_data = {}
    # #
    # #     # 9 - evaluate - calc scores
    # #     logger.debug("evaluating test set")
    # #     predictions, scores = {}, {}
    # #     test_scores, test_pred_df = self._eval_hgnn(trained_model=hg_model, eval_data=test_data, model_args=model_args,
    # #                                                 **kwargs)
    # #     assert pd.isnull(test_pred_df).sum().sum() == 0, "some null predictions"
    # #     # test_scores, test_y_pred = self._eval_hgnn_w_loader(trained_model=hg_model, eval_data=test_data)
    # #
    # #     # 10 - update outputs
    # #     _len = len(out_test_pred)
    # #     out_test_pred = pd.merge(out_test_pred, test_pred_df, on=[self.srna_nid_col, self.mrna_nid_col], how='left')
    # #     assert len(out_test_pred) == _len
    # #     test_y_pred = out_test_pred['y_score']
    # #     test_original_score = out_test_pred['y_original_score']
    # #
    # #     scores.update({'test': test_scores})
    # #     predictions.update({'test_pred': test_y_pred, 'test_original_score': test_original_score,
    # #                         'test_pred_df': test_pred_df, 'out_test_pred': out_test_pred})
    # #
    # #     # 11 - SHAP args
    # #     shap_args = {
    # #         "model_nm": "kGraphRNA",
    # #         "trained_model": hg_model,
    # #         "X_train": pd.DataFrame(),
    # #         "X_val": None,
    # #         "X_test": pd.DataFrame()
    # #     }
    # #
    # #     return scores, predictions, training_history, train_val_data, shap_args
    #
    # def train_and_test_all_pairs(self, X_train: pd.DataFrame, y_train: List[int], X_bench: pd.DataFrame,
    #                              y_bench: List[int], X_test_all_unq_pairs: pd.DataFrame,
    #                              model_args: dict, metadata_train: pd.DataFrame, metadata_bench: pd.DataFrame,
    #                              train_neg_sampling: bool = True, srna_acc_col: str = 'sRNA_accession_id_Eco',
    #                              mrna_acc_col: str = 'mRNA_accession_id_Eco',
    #                              is_syn_col: str = 'is_synthetic', **kwargs) -> \
    #         (pd.DataFrame, Dict[str, object], Dict[str, object]):
    #     """
    #     torch_geometric.__version__ = '2.1.0'
    #
    #     Parameters
    #     ----------
    #     X_train: pd.DataFrame (n_samples, N_features),
    #     y_train: list (n_samples,),
    #     X_bench: pd.DataFrame (t_samples, N_features),
    #     y_bench: list (t_samples,)
    #     X_test_all_unq_pairs: pd.DataFrame
    #     model_args: Dict of model's constructor and fit() arguments
    #     metadata_train: pd.DataFrame (n_samples, T_features)
    #     metadata_bench: pd.DataFrame (t_samples, T_features)
    #     unq_train: pd.DataFrame (_samples, T_features)
    #     unq_test: pd.DataFrame (t_samples, T_features)
    #     train_neg_sampling: whether to add random negative sampling to train HeteroData
    #     srna_acc_col: str  sRNA EcoCyc accession id col in metadata_train and metadata_test
    #     mrna_acc_col: str  mRNA EcoCyc accession id col in metadata_train and metadata_test
    #     is_syn_col: is synthetic indicator col in metadata_train
    #     kwargs:
    #         mrna_eco: all mRNA metadata from EcoCyc
    #         mrna_similarity: known similarity score between mRNAs (id = EcoCyc accession id)
    #
    #     Returns
    #     -------
    #
    #
    #     out_test_pred: pd.DataFrame
    #     training_history - Dict in the following format:  {
    #         'train': OrderedDict
    #         'validation': OrderedDict
    #     }
    #     train_val_data - Dict in the following format:  {
    #         "X_train": pd.DataFrame (n_samples, N_features),
    #         "y_train": list (n_samples,),
    #         "X_val": pd.DataFrame (k_samples, K_features),
    #         "y_val": list (k_samples,),
    #         "metadata_train": pd.DataFrame (n_samples, T_features),    - Optional (in case metadata_train is not None)
    #         "metadata_val": list (k_samples,)    - Optional (in case metadata_train is not None)
    #     }
    #     """
    #     logger.debug(f"training an kGraphRNA model  ->  train negative sampling = {train_neg_sampling}")
    #     # 1 - define graph nodes
    #     if not self.nodes_are_defined:
    #         self._define_nodes_and_features(**kwargs)
    #
    #     # 2 - remove synthetic data from train
    #     if sum(metadata_train[is_syn_col]) > 0:
    #         logger.warning("removing synthetic samples from train")
    #         X_train, y_train, metadata_train = \
    #             self._remove_synthetic_samples(X=X_train, y=y_train, metadata=metadata_train, is_syn_col=is_syn_col)
    #
    #     # 3 - get unique interactions data (train and benchmarking)
    #     unq_train = self._get_unique_inter(metadata=metadata_train, y=y_train, srna_acc_col=srna_acc_col,
    #                                        mrna_acc_col=mrna_acc_col, df_nm='train')
    #     unq_bench = self._get_unique_inter(metadata=metadata_bench, y=y_bench, srna_acc_col=srna_acc_col,
    #                                        mrna_acc_col=mrna_acc_col, df_nm='test')
    #     unq_test = X_test_all_unq_pairs
    #
    #     # 4 - remove train and bench interactions from test
    #     unq_test = self._remove_train_inter_from_test_all_pairs(unq_train=unq_train, unq_test=unq_test,
    #                                                             srna_acc_col=srna_acc_col, mrna_acc_col=mrna_acc_col)
    #     unq_test[self.binary_intr_label_col] = 1
    #
    #     # 5 - map interactions to edges
    #     unq_train = self._map_interactions_to_edges(unique_intr=unq_train, srna_acc_col=srna_acc_col,
    #                                                 mrna_acc_col=mrna_acc_col)
    #     unq_test = self._map_interactions_to_edges(unique_intr=unq_test, srna_acc_col=srna_acc_col,
    #                                                mrna_acc_col=mrna_acc_col)
    #     # 4.1 - update output df
    #     out_test_pred = unq_test
    #
    #     # 5 - init train & test sets (HeteroData)
    #     train_data, test_data = self._init_train_test_hetero_data(unq_train=unq_train, unq_test=unq_test,
    #                                                               train_neg_sampling=train_neg_sampling)
    #     model_args['add_sim'] = False
    #
    #     # 7 - init kGraphRNA model
    #     # 7.1 - set params
    #     srna_num_emb = len(self.srna_nodes)
    #     mrna_num_emb = len(self.mrna_nodes)
    #     metadata = train_data.metadata()
    #     # 7.2 - init model
    #     hga_model = kGraphRNA(srna=self.srna, mrna=self.mrna, srna_to_mrna=self.srna_to_mrna,
    #                           mrna_to_mrna=self.mrna_to_mrna,
    #                           srna_num_embeddings=srna_num_emb, mrna_num_embeddings=mrna_num_emb,
    #                           metadata=metadata, model_args=model_args, **kwargs)
    #     if self.debug_logs:
    #         logger.debug(hga_model)
    #
    #     # 8 - train kGraphRNA model
    #     if self.train_w_loader:
    #         hg_model = self._train_hgnn_with_loader(hg_model=hga_model, train_data=train_data, model_args=model_args)
    #     else:
    #         hg_model = self._train_hgnn(hga_model=hga_model, train_data=train_data, model_args=model_args)
    #     training_history, train_val_data = {}, {}
    #
    #     # 9 - evaluate - calc scores
    #     logger.debug("predicting on all pairs")
    #     predictions = {}
    #     test_pred_df = self._predict_hgnn(trained_model=hg_model, eval_data=test_data, model_args=model_args, **kwargs)
    #     assert pd.isnull(test_pred_df).sum().sum() == 0, "some null predictions"
    #
    #     # 10 - update outputs
    #     _len = len(out_test_pred)
    #     out_test_pred = pd.merge(out_test_pred, test_pred_df, on=[self.srna_nid_col, self.mrna_nid_col], how='left')
    #     out_test_pred = out_test_pred[[x for x in out_test_pred.columns.values if x != self.binary_intr_label_col]]
    #     assert len(out_test_pred) == _len
    #     # test_y_pred = out_test_pred['y_score']
    #
    #     return out_test_pred, training_history, train_val_data
    #
    # def add_rna_acc(self, _df: pd.DataFrame) -> pd.DataFrame:
    #     _len = len(_df)
    #     _df = pd.merge(_df, self.srna_nodes[[self.srna_nid_col, 'EcoCyc_accession_id']], on=self.srna_nid_col,
    #                    how='left').rename(columns={'EcoCyc_accession_id': "sRNA_EcoCyc_accession_id"})
    #     _df = pd.merge(_df, self.mrna_nodes[[self.mrna_nid_col, 'EcoCyc_accession_id']], on=self.mrna_nid_col,
    #                    how='left').rename(columns={'EcoCyc_accession_id': "mRNA_EcoCyc_accession_id"})
    #     assert len(_df) == _len, "duplications post merge"
    #     return _df
    #
    # def get_predictions_df(self, unq_intr: pd.DataFrame, y_true: list, y_score: np.array, out_col_y_true: str = "y_true",
    #                        out_col_y_score: str = "y_score", sort_df: bool = True) -> pd.DataFrame:
    #     # todo - handle multiclass
    #     is_length_compatible = len(unq_intr) == len(y_true) == len(y_score)
    #     assert is_length_compatible, "unq_intr, y_true, y_score and metadata are not compatible in length"
    #     assert pd.isnull(unq_intr).sum().sum() == sum(pd.isnull(y_true)) == sum(pd.isnull(y_score)) == 0, "nulls in data"
    #
    #     _df = unq_intr.copy()
    #     _df[out_col_y_true] = y_true
    #     _df[out_col_y_score] = y_score
    #     # for testing
    #     scores = calc_binary_classification_metrics_using_y_score(y_true=y_true, y_score=y_score)
    #     for k, v in scores.items():
    #         print(f"{k}: {v}")
    #
    #     # add metadata
    #     srna_meta_cols = [c for c in self.srna_nodes.columns.values if c != self.srna_nid_col]
    #     mrna_meta_cols = [c for c in self.mrna_nodes.columns.values if c != self.mrna_nid_col]
    #     _len = len(_df)
    #     _df = pd.merge(_df, self.srna_nodes, on=self.srna_nid_col, how='left').rename(
    #         columns={c: f"sRNA_{c}" for c in srna_meta_cols})
    #     _df = pd.merge(_df, self.mrna_nodes, on=self.mrna_nid_col, how='left').rename(
    #         columns={c: f"mRNA_{c}" for c in mrna_meta_cols})
    #     assert len(_df) == _len, "duplications post merge"
    #
    #     if sort_df:
    #         _df = _df.sort_values(by=out_col_y_score, ascending=False).reset_index(drop=True)
    #
    #     return _df
    #
    # # def run_additional_cross_validation_w_val(self, model_args: dict, feature_cols: List[str], **kwargs) -> Dict[str, Dict[str, object]]:
    # #     """
    # #     Returns
    # #     -------
    # #     cv_outs - Dict in the following format:  {
    # #         'neg_syn': {
    # #             'cv_scores': <cv_scores>,
    # #             'cv_prediction_dfs': <cv_prediction_dfs>,
    # #             'cv_training_history': <cv_training_history>
    # #         },
    # #         'neg_rnd': {
    # #             'cv_scores': <cv_scores>,
    # #             'cv_prediction_dfs': <cv_prediction_dfs>,
    # #             'cv_training_history': <cv_training_history>
    # #         }
    # #     }
    # #
    # #     Where:
    # #     cv_scores - Dict in the following format:  {
    # #         <fold>: {
    # #             'val': {
    # #                 <score_nm>: score
    # #                 }
    # #     }
    # #     cv_prediction_dfs - Dict in the following format:  {
    # #         <fold>: pd.DataFrame
    # #     }
    # #     cv_training_history - Dict in the following format:  {
    # #         <fold>: {
    # #             'train': OrderedDict
    # #             'validation': OrderedDict
    # #         }
    # #     }
    # #     cv_data - Dict in the following format:  {
    # #         <fold>: {
    # #             "X_train": pd.DataFrame (n_samples, N_features),
    # #             "y_train": list (n_samples,),
    # #             "X_val": pd.DataFrame (k_samples, K_features),
    # #             "y_val": list (k_samples,),
    # #             "metadata_train": pd.DataFrame (n_samples, T_features),    - Optional (in case metadata is not None)
    # #             "metadata_val": list (k_samples,)    - Optional (in case metadata is not None)
    # #         }
    # #     }
    # #     """
    # #     logger.debug(f"running additional cross validation - with val set")
    # #     n_splits_rnd = 10
    # #     cv_outs = {}
    # #
    # #     # 1 - define X, y and metadata
    # #     train_for_cv_rnd = kwargs['train_for_cv_rnd']
    # #     cols_to_rem = ['y_score', 'COPRA_sRNA_is_missing', 'COPRA_sRNA', 'COPRA_mRNA', 'COPRA_mRNA_locus_tag',
    # #                         'COPRA_pv', 'COPRA_fdr', 'COPRA_NC_000913', 'COPRA_mRNA_not_in_output',
    # #                         'COPRA_validated_pv', 'COPRA_validated_score']
    # #     X_rnd  = train_for_cv_rnd[feature_cols]
    # #     y_rnd = list(train_for_cv_rnd[self.binary_intr_label_col])
    # #     meta_cols = [c for c in train_for_cv_rnd.columns.values if c not in [self.binary_intr_label_col] + feature_cols + cols_to_rem]
    # #     metadata_rnd = train_for_cv_rnd[meta_cols]
    # #
    # #     # 2 - run CV with random sampling of negatives
    # #     cv_scores, cv_predictions_dfs, cv_training_history, cv_data = \
    # #         self.run_cross_validation_w_val(X=X_rnd, y=y_rnd, metadata=metadata_rnd, n_splits=n_splits_rnd,
    # #                                         model_args=model_args, **kwargs)
    # #
    # #     # 3 - save results
    # #     cv_outs['neg_rnd_w_val'] = {
    # #         "cv_scores": cv_scores,
    # #         "cv_predictions_dfs": cv_predictions_dfs,
    # #         "cv_training_history": cv_training_history,
    # #         "minimal_val_diff": kwargs['minimal_val_diff']
    # #     }
    # #
    # #     return cv_outs
    #
    # def run_additional_cross_validation(self, model_args: dict, feature_cols: List[str], **kwargs) -> Dict[str, Dict[str, object]]:
    #     """
    #     Returns
    #     -------
    #     cv_outs - Dict in the following format:  {
    #         'neg_syn': {
    #             'cv_scores': <cv_scores>,
    #             'cv_prediction_dfs': <cv_prediction_dfs>,
    #             'cv_training_history': <cv_training_history>
    #         },
    #         'neg_rnd': {
    #             'cv_scores': <cv_scores>,
    #             'cv_prediction_dfs': <cv_prediction_dfs>,
    #             'cv_training_history': <cv_training_history>
    #         }
    #     }
    #
    #     Where:
    #     cv_scores - Dict in the following format:  {
    #         <fold>: {
    #             'val': {
    #                 <score_nm>: score
    #                 }
    #     }
    #     cv_prediction_dfs - Dict in the following format:  {
    #         <fold>: pd.DataFrame
    #     }
    #     cv_training_history - Dict in the following format:  {
    #         <fold>: {
    #             'train': OrderedDict
    #             'validation': OrderedDict
    #         }
    #     }
    #     cv_data - Dict in the following format:  {
    #         <fold>: {
    #             "X_train": pd.DataFrame (n_samples, N_features),
    #             "y_train": list (n_samples,),
    #             "X_val": pd.DataFrame (k_samples, K_features),
    #             "y_val": list (k_samples,),
    #             "metadata_train": pd.DataFrame (n_samples, T_features),    - Optional (in case metadata is not None)
    #             "metadata_val": list (k_samples,)    - Optional (in case metadata is not None)
    #         }
    #     }
    #     """
    #     logger.debug(f"running additional cross validation")
    #     n_splits_rnd = 10
    #     cv_outs = {}
    #
    #     # 1 - define X, y and metadata
    #     train_for_cv_rnd = kwargs['train_for_cv_rnd']
    #     cols_to_rem = ['y_score', 'COPRA_sRNA_is_missing', 'COPRA_sRNA', 'COPRA_mRNA', 'COPRA_mRNA_locus_tag',
    #                    'COPRA_pv', 'COPRA_fdr', 'COPRA_NC_000913', 'COPRA_mRNA_not_in_output',
    #                    'COPRA_validated_pv', 'COPRA_validated_score']
    #     X_rnd  = train_for_cv_rnd[feature_cols]
    #     y_rnd = list(train_for_cv_rnd[self.binary_intr_label_col])
    #     meta_cols = [c for c in train_for_cv_rnd.columns.values if c not in [self.binary_intr_label_col] + feature_cols + cols_to_rem]
    #     metadata_rnd = train_for_cv_rnd[meta_cols]
    #
    #     # 2 - run CV with random sampling of negatives
    #     cv_scores, cv_predictions_dfs, cv_training_history, cv_data = \
    #         self.run_cross_validation(X=X_rnd, y=y_rnd, metadata=metadata_rnd, n_splits=n_splits_rnd,
    #                                   model_args=model_args, **kwargs)
    #
    #     # 3 - save results
    #     cv_outs['neg_rnd'] = {
    #         "cv_scores": cv_scores,
    #         "cv_predictions_dfs": cv_predictions_dfs,
    #         "cv_training_history": cv_training_history,
    #     }
    #
    #     return cv_outs
    #
    # def process_loco_fold_data(self, fold_data: Dict[str, pd.DataFrame], srna_acc_col: str = 'sRNA_accession_id_Eco',
    #                            mrna_acc_col: str = 'mRNA_accession_id_Eco', label_col: str = 'interaction_label'):
    #     """
    #
    #     Parameters
    #     ----------
    #     fold_data
    #     srna_acc_col
    #     mrna_acc_col
    #     label_col
    #
    #     Returns
    #     -------
    #
    #     """
    #     unq_train = fold_data['unq_train'].copy()
    #     unq_test = fold_data['unq_test'].copy()
    #
    #     # train
    #     unq_train_gnn = self._map_interactions_to_edges(unique_intr=unq_train, srna_acc_col=srna_acc_col,
    #                                                     mrna_acc_col=mrna_acc_col)
    #     unq_train_gnn = pd.DataFrame(shuffle(unq_train_gnn, random_state=10)).reset_index(drop=True)
    #     # test
    #     unq_test_gnn = self._map_interactions_to_edges(unique_intr=unq_test, srna_acc_col=srna_acc_col,
    #                                                    mrna_acc_col=mrna_acc_col)
    #     unq_test_gnn = pd.DataFrame(shuffle(unq_test_gnn, random_state=10)).reset_index(drop=True)
    #
    #     out_fold_data = {
    #         'unq_train': unq_train_gnn,
    #         'unq_val': unq_test_gnn,
    #     }
    #     return out_fold_data
    #
    # def predict_on_folds(self, model_args: dict, cv_data_unq: Dict[int, Dict[str, object]], **kwargs) -> \
    #         (Dict[int, Dict[str, Dict[str, float]]], Dict[int, pd.DataFrame], Dict[int, dict], Dict[int, dict]):
    #     """
    #     Returns
    #     -------
    #     cv_scores - Dict in the following format:  {
    #         <fold>: {
    #             'val': {
    #                 <score_nm>: score
    #                 }
    #     }
    #     cv_predictions_dfs - Dict in the following format:  {
    #         <fold>: pd.DataFrame
    #     }
    #     cv_training_history - Dict in the following format:  {
    #         <fold>: {
    #             'train': OrderedDict
    #             'validation': OrderedDict
    #         }
    #     }
    #     """
    #     logger.debug("predicting on folds")
    #     dummy_x_train, dummy_x_val = pd.DataFrame(), pd.DataFrame()  # irrelevant when using unique interactions
    #     dummy_y_train, dummy_y_val = list(), list()
    #     dummy_meta_train, dummy_meta_val = pd.DataFrame(), pd.DataFrame()
    #
    #     # 6 - predict on folds
    #     cv_scores = {}
    #     cv_training_history = {}
    #     cv_prediction_dfs = {}
    #     train_neg_sampling = False  # negatives were already added to cv_data_unq
    #     for fold, fold_data_unq in cv_data_unq.items():
    #         logger.debug(f"starting fold {fold}")
    #         # 2.2 - predict on validation set (pos + random sampled neg)
    #         scores, predictions, training_history, _, shap_args = \
    #             self.train_and_test(X_train=dummy_x_train, y_train=dummy_y_train, X_test=dummy_x_val, y_test=dummy_y_val,
    #                                 model_args=model_args, metadata_train=dummy_meta_train, metadata_test=dummy_meta_val,
    #                                 unq_train=fold_data_unq['unq_train'], unq_test=fold_data_unq['unq_val'],
    #                                 train_neg_sampling=train_neg_sampling, **kwargs)
    #         # 2.1 - fold's val scores
    #         cv_scores[fold] = {'val': scores['test']}
    #         # 2.2 - fold's training history
    #         cv_training_history[fold] = training_history
    #         # 2.3 - fold's predictions df
    #         y_val_pred = predictions['test_pred']
    #         unq_val = fold_data_unq['unq_val'][[self.srna_nid_col, self.mrna_nid_col]]
    #         y_val = fold_data_unq['unq_val'][self.binary_intr_label_col]
    #         cv_pred_df = self.get_predictions_df(unq_intr=unq_val, y_true=y_val, y_score=y_val_pred)
    #         cv_prediction_dfs[fold] = cv_pred_df
    #
    #     return cv_scores, cv_prediction_dfs, cv_training_history, cv_data_unq
    #
    # def loco_impl(self, _loco_folds: Dict[str, dict], model_args: dict, **kwargs) -> Dict[str, object]:
    #     # 1 - prepare LOCO folds data
    #     for cond, fold_data in _loco_folds.items():
    #         m_fold_data = self.process_loco_fold_data(fold_data=fold_data)
    #         _loco_folds[cond] = m_fold_data
    #
    #     # 2 - run LOCO
    #     loco_scores, loco_predictions_dfs, loco_training_history, _ = \
    #         self.predict_on_folds(model_args=model_args, cv_data_unq=_loco_folds, **kwargs)
    #     res = {
    #         "cv_scores": loco_scores,
    #         "cv_predictions_dfs": loco_predictions_dfs,
    #         "cv_training_history": loco_training_history
    #     }
    #
    #     return res
    #
    # def run_loco(self, model_args: dict, feature_cols: List[str], **kwargs) -> Dict[str, Dict[str, object]]:
    #     """
    #     Returns
    #     -------
    #     loco_outs - Dict in the following format:  {
    #         'loco': {
    #             'cv_scores': <cv_scores>,
    #             'cv_prediction_dfs': <cv_prediction_dfs>,
    #             'cv_training_history': <cv_training_history>
    #         },
    #         'loco_swap': {
    #             'cv_scores': <cv_scores>,
    #             'cv_prediction_dfs': <cv_prediction_dfs>,
    #             'cv_training_history': <cv_training_history>
    #         }
    #     }
    #
    #     Where:
    #     cv_scores - Dict in the following format:  {
    #         <fold>: {
    #             'val': {
    #                 <score_nm>: score
    #                 }
    #     }
    #     cv_prediction_dfs - Dict in the following format:  {
    #         <fold>: pd.DataFrame
    #     }
    #     cv_training_history - Dict in the following format:  {
    #         <fold>: {
    #             'train': OrderedDict
    #             'validation': OrderedDict
    #         }
    #     }
    #     cv_data - Dict in the following format:  {
    #         <fold>: {
    #             "X_train": pd.DataFrame (n_samples, N_features),
    #             "y_train": list (n_samples,),
    #             "X_val": pd.DataFrame (k_samples, K_features),
    #             "y_val": list (k_samples,),
    #             "metadata_train": pd.DataFrame (n_samples, T_features),    - Optional (in case metadata is not None)
    #             "metadata_val": list (k_samples,)    - Optional (in case metadata is not None)
    #         }
    #     }
    #     """
    #     logger.debug(f"running LOCO")
    #     # 0 - define graph nodes (if needed) and map interaction
    #     if not self.nodes_are_defined:
    #         self._define_nodes_and_features(**kwargs)
    #
    #     loco_folds, loco_folds_all = kwargs['f_loco_folds'].copy(), kwargs['loco_folds'].copy()
    #     loco_folds_swap, loco_folds_swap_all = kwargs['f_loco_folds_swap'].copy(), kwargs['loco_folds_swap'].copy()
    #     loco_outs = {}
    #     # -------------------- Filtered
    #     # 1.1 - random negatives
    #     res = self.loco_impl(_loco_folds=loco_folds, model_args=model_args, **kwargs)
    #     loco_outs['loco'] = res
    #     # 1.2 - swap negatives
    #     res = self.loco_impl(_loco_folds=loco_folds_swap, model_args=model_args, **kwargs)
    #     loco_outs['loco_swap'] = res
    #     # -------------------- All
    #     # 2.1 - random negatives
    #     res = self.loco_impl(_loco_folds=loco_folds_all, model_args=model_args, **kwargs)
    #     loco_outs['loco_all'] = res
    #     # 2.2 - swap negatives
    #     res = self.loco_impl(_loco_folds=loco_folds_swap_all, model_args=model_args, **kwargs)
    #     loco_outs['loco_swap_all'] = res
    #
    #     return loco_outs
    #
    # # def run_cross_validation_w_val(self, X: pd.DataFrame, y: List[int], n_splits: int, model_args: dict,
    # #                                metadata: pd.DataFrame = None, srna_acc_col: str = 'sRNA_accession_id_Eco',
    # #                                mrna_acc_col: str = 'mRNA_accession_id_Eco', is_syn_col: str = 'is_synthetic', **kwargs) \
    # #         -> (Dict[int, Dict[str, Dict[str, float]]], Dict[int, pd.DataFrame], Dict[int, dict], Dict[int, dict]):
    # #     """
    # #     Returns
    # #     -------
    # #
    # #     cv_scores - Dict in the following format:  {
    # #         <fold>: {
    # #             'val': {
    # #                 <score_nm>: score
    # #                 }
    # #     }
    # #     cv_prediction_dfs - Dict in the following format:  {
    # #         <fold>: pd.DataFrame
    # #     }
    # #     cv_training_history - Dict in the following format:  {
    # #         <fold>: {
    # #             'train': OrderedDict
    # #             'validation': OrderedDict
    # #         }
    # #     }
    # #     cv_data - Dict in the following format:  {
    # #         <fold>: {
    # #             "X_train": pd.DataFrame (n_samples, N_features),
    # #             "y_train": list (n_samples,),
    # #             "X_val": pd.DataFrame (k_samples, K_features),
    # #             "y_val": list (k_samples,),
    # #             "metadata_train": pd.DataFrame (n_samples, T_features),    - Optional (in case metadata is not None)
    # #             "metadata_val": list (k_samples,)    - Optional (in case metadata is not None)
    # #         }
    # #     }
    # #     """
    # #     logger.debug(f"running cross validation with {n_splits} folds - w val set")
    # #
    # #     # 1 - remove all synthetic samples
    # #     logger.warning("removing all synthetic samples")
    # #     X_no_syn, y_no_syn, metadata_no_syn = \
    # #         self._remove_synthetic_samples(X=X, y=y, metadata=metadata, is_syn_col=is_syn_col)
    # #
    # #     # 2 - get unique interactions data (train + val)
    # #     unq_intr = self._get_unique_inter(metadata=metadata_no_syn, y=y_no_syn, srna_acc_col=srna_acc_col,
    # #                                       mrna_acc_col=mrna_acc_col, df_nm='all')
    # #     unq_intr_pos, unq_intr_neg = self._pos_neg_split(df=unq_intr, binary_label_col=self.binary_intr_label_col)
    # #
    # #     # 3 - define graph nodes (if needed) and map interaction
    # #     if not self.nodes_are_defined:
    # #         self._define_nodes_and_features(**kwargs)
    # #     unq_intr_pos = self._map_interactions_to_edges(unique_intr=unq_intr_pos, srna_acc_col=srna_acc_col,
    # #                                                    mrna_acc_col=mrna_acc_col)
    # #     # 4 - random negative sampling - all cv data
    # #     # todo - add indication for random vs real neg
    # #     _shuffle = True
    # #     unq_data = self._add_neg_samples(unq_intr_pos=unq_intr_pos, ratio=self.cv_neg_sampling_ratio_data,
    # #                                      _shuffle=_shuffle)
    # #     unq_y = np.array(unq_data[self.binary_intr_label_col])
    # #     unq_intr_data = unq_data[[self.srna_nid_col, self.mrna_nid_col]]
    # #
    # #     # 5 - split data into folds
    # #     cv_data_unq = get_stratified_cv_folds_for_unique(unq_intr_data=unq_intr_data, unq_y=unq_y, n_splits=n_splits,
    # #                                                      label_col=self.binary_intr_label_col, shuffle=False,
    # #                                                      seed=None)
    # #     dummy_x_train, dummy_x_val = pd.DataFrame(), pd.DataFrame()  # irrelevant when using unique interactions
    # #     dummy_y_train, dummy_y_val = list(), list()
    # #     dummy_meta_train, dummy_meta_val = pd.DataFrame(), pd.DataFrame()
    # #
    # #     # 6 - predict on folds
    # #     cv_scores = {}
    # #     cv_training_history = {}
    # #     cv_prediction_dfs = {}
    # #     train_neg_sampling = False  # negatives were already added to cv_data_unq
    # #     for fold, fold_data_unq in cv_data_unq.items():
    # #         logger.debug(f"starting fold {fold}")
    # #         # 2.2 - predict on validation set (pos + random sampled neg)
    # #         scores, predictions, training_history, _, shap_args = \
    # #             self.train_val_and_test(X_train=dummy_x_train, y_train=dummy_y_train, X_test=dummy_x_val, y_test=dummy_y_val,
    # #                                     model_args=model_args, metadata_train=dummy_meta_train, metadata_test=dummy_meta_val,
    # #                                     unq_train=fold_data_unq['unq_train'], unq_test=fold_data_unq['unq_val'],
    # #                                     train_neg_sampling=train_neg_sampling, **kwargs)
    # #         # 2.1 - fold's val scores
    # #         cv_scores[fold] = {'val': scores['test']}
    # #         # 2.2 - fold's training history
    # #         cv_training_history[fold] = training_history
    # #         # 2.3 - fold's predictions df
    # #         y_val_pred = predictions['test_pred']
    # #         y_val_original_score = predictions['test_original_score']
    # #         unq_val = fold_data_unq['unq_val'][[self.srna_nid_col, self.mrna_nid_col]]
    # #         y_val = fold_data_unq['unq_val'][self.binary_intr_label_col]
    # #         cv_pred_df = self.get_predictions_df(unq_intr=unq_val, y_true=y_val, y_score=y_val_pred,
    # #                                              y_original_score=y_val_original_score)
    # #         cv_prediction_dfs[fold] = cv_pred_df
    # #
    # #     return cv_scores, cv_prediction_dfs, cv_training_history, cv_data_unq
    #
    # def run_cross_validation(self, X: pd.DataFrame, y: List[int], n_splits: int, model_args: dict,
    #                          metadata: pd.DataFrame = None, srna_acc_col: str = 'sRNA_accession_id_Eco',
    #                          mrna_acc_col: str = 'mRNA_accession_id_Eco', is_syn_col: str = 'is_synthetic', **kwargs) \
    #         -> (Dict[int, Dict[str, Dict[str, float]]], Dict[int, pd.DataFrame], Dict[int, dict], Dict[int, dict]):
    #     """
    #     Returns
    #     -------
    #
    #     cv_scores - Dict in the following format:  {
    #         <fold>: {
    #             'val': {
    #                 <score_nm>: score
    #                 }
    #     }
    #     cv_prediction_dfs - Dict in the following format:  {
    #         <fold>: pd.DataFrame
    #     }
    #     cv_training_history - Dict in the following format:  {
    #         <fold>: {
    #             'train': OrderedDict
    #             'validation': OrderedDict
    #         }
    #     }
    #     cv_data - Dict in the following format:  {
    #         <fold>: {
    #             "X_train": pd.DataFrame (n_samples, N_features),
    #             "y_train": list (n_samples,),
    #             "X_val": pd.DataFrame (k_samples, K_features),
    #             "y_val": list (k_samples,),
    #             "metadata_train": pd.DataFrame (n_samples, T_features),    - Optional (in case metadata is not None)
    #             "metadata_val": list (k_samples,)    - Optional (in case metadata is not None)
    #         }
    #     }
    #     """
    #     logger.debug(f"running cross validation with {n_splits} folds")
    #
    #     # 1 - remove all synthetic samples
    #     logger.warning("removing all synthetic samples")
    #     X_no_syn, y_no_syn, metadata_no_syn = \
    #         self._remove_synthetic_samples(X=X, y=y, metadata=metadata, is_syn_col=is_syn_col)
    #
    #     # 2 - get unique interactions data (train + val)
    #     unq_intr = self._get_unique_inter(metadata=metadata_no_syn, y=y_no_syn, srna_acc_col=srna_acc_col,
    #                                       mrna_acc_col=mrna_acc_col, df_nm='all')
    #     unq_intr_pos, unq_intr_neg = self._pos_neg_split(df=unq_intr, binary_label_col=self.binary_intr_label_col)
    #
    #     # 3 - define graph nodes (if needed) and map interaction
    #     if not self.nodes_are_defined:
    #         self._define_nodes_and_features(**kwargs)
    #     unq_intr_pos = self._map_interactions_to_edges(unique_intr=unq_intr_pos, srna_acc_col=srna_acc_col,
    #                                                    mrna_acc_col=mrna_acc_col)
    #     # 4 - random negative sampling - all cv data
    #     # todo - add indication for random vs real neg
    #     _shuffle = True
    #     unq_data = self._add_neg_samples(unq_intr_pos=unq_intr_pos, ratio=self.cv_neg_sampling_ratio_data,
    #                                      _shuffle=_shuffle)
    #     unq_y = np.array(unq_data[self.binary_intr_label_col])
    #     unq_intr_data = unq_data[[self.srna_nid_col, self.mrna_nid_col]]
    #
    #     # 5 - split data into folds
    #     cv_data_unq = get_stratified_cv_folds_for_unique(unq_intr_data=unq_intr_data, unq_y=unq_y, n_splits=n_splits,
    #                                                      label_col=self.binary_intr_label_col, shuffle=False,
    #                                                      seed=None)
    #     dummy_x_train, dummy_x_val = pd.DataFrame(), pd.DataFrame()  # irrelevant when using unique interactions
    #     dummy_y_train, dummy_y_val = list(), list()
    #     dummy_meta_train, dummy_meta_val = pd.DataFrame(), pd.DataFrame()
    #
    #     # 6 - predict on folds
    #     cv_scores = {}
    #     cv_training_history = {}
    #     cv_prediction_dfs = {}
    #     train_neg_sampling = False  # negatives were already added to cv_data_unq
    #     for fold, fold_data_unq in cv_data_unq.items():
    #         logger.debug(f"starting fold {fold}")
    #         # 2.2 - predict on validation set (pos + random sampled neg)
    #         scores, predictions, training_history, _, shap_args = \
    #             self.train_and_test(X_train=dummy_x_train, y_train=dummy_y_train, X_test=dummy_x_val, y_test=dummy_y_val,
    #                                 model_args=model_args, metadata_train=dummy_meta_train, metadata_test=dummy_meta_val,
    #                                 unq_train=fold_data_unq['unq_train'], unq_test=fold_data_unq['unq_val'],
    #                                 train_neg_sampling=train_neg_sampling, **kwargs)
    #         # 2.1 - fold's val scores
    #         cv_scores[fold] = {'val': scores['test']}
    #         # 2.2 - fold's training history
    #         cv_training_history[fold] = training_history
    #         # 2.3 - fold's predictions df
    #         y_val_pred = predictions['test_pred']
    #         unq_val = fold_data_unq['unq_val'][[self.srna_nid_col, self.mrna_nid_col]]
    #         y_val = fold_data_unq['unq_val'][self.binary_intr_label_col]
    #         cv_pred_df = self.get_predictions_df(unq_intr=unq_val, y_true=y_val, y_score=y_val_pred)
    #         cv_prediction_dfs[fold] = cv_pred_df
    #
    #     return cv_scores, cv_prediction_dfs, cv_training_history, cv_data_unq
    #
    # def train_and_test_w_random_negatives(self, X_test: pd.DataFrame, y_test: List[int], model_args: dict,
    #                                       feature_cols: List[str], metadata_test: pd.DataFrame = None, **kwargs) -> \
    #         (Dict[str, dict], Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object]):
    #     """
    #     sklearn.__version__ = '0.24.0'
    #
    #     Parameters
    #     ----------
    #     X_test: pd.DataFrame (t_samples, N_features),
    #     y_test: list (t_samples,)
    #     model_args: Dict of model's constructor and fit() arguments
    #     feature_cols
    #     metadata_test: Optional - pd.DataFrame (t_samples, T_features)
    #     kwargs
    #
    #     Returns
    #     -------
    #
    #     scores - Dict in the following format:  {
    #         'test': {
    #             <score_nm>: score
    #             }
    #     }
    #     predictions - Dict in the following format:  {
    #         'test_pred': array-like (t_samples,)
    #     }
    #     training_history - Dict in the following format:  {
    #         'train': OrderedDict
    #         'validation': OrderedDict
    #     }
    #     train_val_data - Dict in the following format:  {
    #         "X_train": pd.DataFrame (n_samples, N_features),
    #         "y_train": list (n_samples,),
    #         "X_val": pd.DataFrame (k_samples, K_features),
    #         "y_val": list (k_samples,),
    #         "metadata_train": pd.DataFrame (n_samples, T_features),    - Optional (in case metadata_train is not None)
    #         "metadata_val": list (k_samples,)    - Optional (in case metadata_train is not None)
    #     }
    #     shap_args - Dict in the following format:  {
    #         "model_nm": str,
    #         "trained_model": the trained machine learning model,
    #     }
    #     """
    #     logger.debug("training an kGraphRNA classifier - with random negatives")
    #
    #     # 1 - define X, y and metadata
    #     train_for_cv_rnd = kwargs['train_for_cv_rnd']
    #     cols_to_rem = ['y_score', 'COPRA_sRNA_is_missing', 'COPRA_sRNA', 'COPRA_mRNA', 'COPRA_mRNA_locus_tag',
    #                    'COPRA_pv', 'COPRA_fdr', 'COPRA_NC_000913', 'COPRA_mRNA_not_in_output',
    #                    'COPRA_validated_pv', 'COPRA_validated_score']
    #     X_train  = train_for_cv_rnd[feature_cols]
    #     y_train = list(train_for_cv_rnd[self.binary_intr_label_col])
    #     meta_cols = [c for c in train_for_cv_rnd.columns.values if c not in [self.binary_intr_label_col] + feature_cols + cols_to_rem]
    #     metadata_train = train_for_cv_rnd[meta_cols]
    #     # 1.2 - reset df indexes
    #     X_train = X_train.reset_index(drop=True)
    #     metadata_train = metadata_train.reset_index(drop=True)
    #
    #     # 2 - train and test - with random negatives
    #     scores, predictions, training_history, train_val_data, shap_args = \
    #         self.train_and_test(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_args=model_args,
    #                             metadata_train=metadata_train, metadata_test=metadata_test, **kwargs)
    #
    #     return scores, predictions, training_history, train_val_data, shap_args