from typing import Dict
import numpy as np
import pandas as pd
from utils_old.utils_general import order_df
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import logging
logger = logging.getLogger(__name__)


def aupr_score(y_true, y_score) -> float:
    precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(recall, precision)


def calc_binary_classification_metrics_using_y_score(y_true: list, y_score: np.array, roc_max_fpr: float = None,
                                                     dataset_nm: str = None) -> Dict[str, object]:
    """
    https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

    Parameters
    ----------
    y_true: true binary labels
    y_score: positive numeric scores
    dataset_nm: Optional

    Returns
    -------
    scores - Dict in the following format: {
                '<dataset_nm>_ROC_AUC': <score>,
                '<dataset_nm>_AUPR': <score>,
                '<dataset_nm>_PARTIAL_ROC_AUC_max_fpr_{roc_max_fpr}': <score> - optional
            }
    """
    _pref = f"{dataset_nm}_" if dataset_nm else ""
    # validate inputs
    _valid_input = True
    _valid_y_ture = sorted(set(y_true)) == [0, 1]
    if not _valid_y_ture:
        logger.error(f"y_true values are {sorted(set(y_true))} (expected [0, 1])  ->  metrics are set to None")
        _valid_input = False
    _valid_y_score = sum([pd.notnull(x) and 0 <= x for x in y_score]) == len(y_score)
    if not _valid_y_score:
        logger.error(f"{len(y_score)- sum([pd.notnull(x) and 0 <= x for x in y_score])} "
                     f"elements in y_score are invalid (expected: 0 <= element)")
        _valid_input = False
    # calculate metrics
    scores = {
        f"{_pref}ROC_AUC": roc_auc_score(y_true=y_true, y_score=y_score) if _valid_input else None,
        f"{_pref}AUPR": aupr_score(y_true=y_true, y_score=y_score) if _valid_input else None
    }

    if roc_max_fpr:
        _valid_max_fpr = 0 < roc_max_fpr < 1
        if not _valid_max_fpr:
            logger.error(f"max_fpr = {roc_max_fpr} is invalid")
        p_auc_score = roc_auc_score(y_true=y_true, y_score=y_score, max_fpr=roc_max_fpr) if _valid_max_fpr else None
        scores.update({
            f"{_pref}PARTIAL_ROC_AUC_max_fpr_{roc_max_fpr}": p_auc_score
        })

    return scores


def get_predictions_df(X: pd.DataFrame, y_true: list, y_score: np.array, out_col_y_true: str = "y_true",
                       out_col_y_score: str = "y_score", metadata: pd.DataFrame = None, sort_df: bool = True) \
        -> pd.DataFrame:
    is_length_compatible = len(X) == len(y_true) == len(y_score) if metadata is None \
        else len(X) == len(y_true) == len(y_score) == len(metadata)
    assert is_length_compatible, "X, y_true, y_score and metadata are not compatible in length"

    _df = pd.DataFrame(X)
    first_cols = [out_col_y_true, out_col_y_score]
    if metadata is not None:
        _df = pd.concat(objs=[_df, metadata], axis=1).reset_index(drop=True)
        first_cols = first_cols + list(metadata.columns.values)
    _df[out_col_y_true] = y_true
    _df[out_col_y_score] = y_score

    _df = order_df(df=_df, first_cols=first_cols)
    if sort_df:
        _df = _df.sort_values(by=out_col_y_score, ascending=False).reset_index(drop=True)

    return _df


def get_stratified_cv_folds(X: pd.DataFrame, y: np.array, n_splits: int, metadata: pd.DataFrame = None) -> \
        Dict[int, Dict[str, object]]:
    """

    Parameters
    ----------
    X - pd.DataFrame (m_samples, N_features)
    y - np.array (m_samples,)
    n_splits - cv splits, int
    metadata - Optional - pd.DataFrame (m_samples, T_features)

    Returns
    -------
    cv_folds - Dict in the following format: {
        1:   {
            "X_train": pd.DataFrame (n_samples, N_features),
            "y_train": list (n_samples,),
            "X_val": pd.DataFrame (k_samples, K_features),
            "y_val": list (k_samples,),
            "metadata_train": pd.DataFrame (n_samples, T_features),    - Optional
            "metadata_val": list (k_samples,)    - Optional
            },
        2:  {
        }, ...
    }
    """
    # logger.debug(f"getting stratified {n_splits} cv folds")
    is_length_compatible = len(X) == len(y) if metadata is None else len(X) == len(y) == len(metadata)
    assert is_length_compatible, "X, y, and metadata are compatible in length"

    # 1 - data preparation
    # 1.1 - encode y labels
    lbl = LabelEncoder()
    lbl.fit(y)
    y_enc = lbl.transform(y)
    # 1.2 - reset df indexes
    X = X.reset_index(drop=True)
    if metadata is not None:
        metadata = metadata.reset_index(drop=True)

    # 2 - split data into folds
    cv_folds = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    # skf = StratifiedKFold(n_splits=val_set_n_splits, shuffle=True, random_state=val_set_seed)
    for i, (train_index, val_index) in enumerate(skf.split(X=np.array(X), y=y_enc)):
        fold_data = {
                "X_train": X.iloc[list(train_index), :],
                "y_train": list(y[train_index]),
                "X_val": X.iloc[list(val_index), :],
                "y_val": list(y[val_index])
        }
        if metadata is not None:
            fold_data.update({
                "metadata_train": metadata.iloc[list(train_index), :],
                "metadata_val": metadata.iloc[list(val_index), :]
            })

        cv_folds[i] = fold_data

    return cv_folds


def get_stratified_cv_folds_for_unique(unq_intr_data: pd.DataFrame, unq_y: np.array, n_splits: int, label_col: str,
                                       shuffle: bool = False, seed: int = None) -> Dict[int, Dict[str, object]]:
    """

    Parameters
    ----------
    unq_intr_data - pd.DataFrame (u_samples, [edge_index_0, edge_index_1])
    unq_y - np.array (u_samples,)
    n_splits - cv splits, int


    Returns
    -------
    cv_folds - Dict in the following format: {
        1:   {
            "unq_train": pd.DataFrame (t_samples, [edge_index_0, edge_index_1, edge_label])
            "unq_val": pd.DataFrame (v_samples, [edge_index_0, edge_index_1, edge_label])
            },
        2:  {
        }, ...
    }
    """
    # logger.debug(f"getting stratified {n_splits} cv folds")
    is_length_compatible = len(unq_y) == len(unq_intr_data)
    assert is_length_compatible, "unq_y and unq_intr_data are compatible in length"
    assert unq_intr_data.shape[1] == 2, "unq_intr_data format is pd.DataFrame (u_samples, [edge_index_0, edge_index_1])"

    # 1 - data preparation
    # 1.1 - encode y labels
    lbl = LabelEncoder()
    lbl.fit(unq_y)
    y_enc = lbl.transform(unq_y)
    # 1.2 - reset df indexes
    unq_intr_data = unq_intr_data.reset_index(drop=True)

    # 2 - split data into folds
    cv_folds = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    for i, (train_index, val_index) in enumerate(skf.split(X=np.array(unq_intr_data), y=y_enc)):
        # train
        unq_train = pd.DataFrame(unq_intr_data.iloc[list(train_index), :]).reset_index(drop=True)
        unq_train[label_col] = list(unq_y[train_index])
        # val
        unq_val = pd.DataFrame(unq_intr_data.iloc[list(val_index), :]).reset_index(drop=True)
        unq_val[label_col] = list(unq_y[val_index])

        fold_data = {
                "unq_train": unq_train,
                "unq_val": unq_val
        }
        cv_folds[i] = fold_data

    return cv_folds
