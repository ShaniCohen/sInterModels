# Authors: Shani Cohen (ShaniCohen)
# Python version: 3.8
# Last update: 28.10.2020


from os.path import join
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
# https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
from typing import Set, Tuple, Dict, List
from models_handlers.model_handlers_utils import calc_binary_classification_metrics_using_y_score,\
    calc_binary_classification_metrics_using_prob_y_score_and_y_pred_thresholds, \
    calc_multiclass_classification_scores, get_stratified_cv_folds, \
    get_predictions_df, split_cv_data
from utils import read_df, write_df, key_val_df_to_dict
from feature_extraction.srnarftarget_features import add_srnarftarget_features
import logging
logger = logging.getLogger(__name__)


class kXGBModelHandler(object):
    """
    Class to handle kXGBoost model training and evaluation.
    xgboost.__version__ = '1.5.0'
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier

    Parameters
    ----------

    Attributes
    ----------
    Same as parameters
    """
    def __init__(self):
        super(kXGBModelHandler, self).__init__()
        self.feature_cols = ['F_total_energy_dG', 'F_unfolding_energy_sRNA', 'F_unfolding_energy_mRNA',
                             'F_hybridization_energy', 'F_all_bp', 'F_GC_bp_prop', 'F_AU_bp_prop', 'F_GU_bp_prop',
                             'F_alignment_len', 'F_all_bp_prop', 'F_mismatches_prop', 'F_bulges_sRNA_prop',
                             'F_bulges_mRNA_prop', 'F_mismatches_count', 'F_bulges_sRNA_count',
                             'F_bulges_mRNA_count', 'F_max_consecutive_bp_prop',
                             'F_mRNA_flanking_region_A_prop', 'F_mRNA_flanking_region_U_prop',
                             'F_mRNA_flanking_region_G_prop', 'F_mRNA_flanking_region_C_prop',
                             'F_mRNA_flanking_region_A+U_prop', 'F_mRNA_flanking_region_G+C_prop',
                             'AAA', 'AAT', 'AAG', 'AAC', 'ATA', 'ATT', 'ATG', 'ATC', 'AGA', 'AGT', 'AGG', 'AGC', 'ACA',
                             'ACT', 'ACG', 'ACC', 'TAA', 'TAT', 'TAG', 'TAC', 'TTA', 'TTT', 'TTG', 'TTC', 'TGA', 'TGT',
                             'TGG', 'TGC', 'TCA', 'TCT', 'TCG', 'TCC', 'GAA', 'GAT', 'GAG', 'GAC', 'GTA', 'GTT', 'GTG',
                             'GTC', 'GGA', 'GGT', 'GGG', 'GGC', 'GCA', 'GCT', 'GCG', 'GCC', 'CAA', 'CAT', 'CAG', 'CAC',
                             'CTA', 'CTT', 'CTG', 'CTC', 'CGA', 'CGT', 'CGG', 'CGC', 'CCA', 'CCT', 'CCG', 'CCC']
        self.intr_feat_cols = ['F_total_energy_dG', 'F_unfolding_energy_sRNA', 'F_unfolding_energy_mRNA',
                               'F_hybridization_energy', 'F_all_bp', 'F_GC_bp_prop', 'F_AU_bp_prop', 'F_GU_bp_prop',
                               'F_alignment_len', 'F_all_bp_prop', 'F_mismatches_prop', 'F_bulges_sRNA_prop',
                               'F_bulges_mRNA_prop', 'F_mismatches_count', 'F_bulges_sRNA_count',
                               'F_bulges_mRNA_count', 'F_max_consecutive_bp_prop',
                               'F_mRNA_flanking_region_A_prop', 'F_mRNA_flanking_region_U_prop',
                               'F_mRNA_flanking_region_G_prop', 'F_mRNA_flanking_region_C_prop',
                               'F_mRNA_flanking_region_A+U_prop', 'F_mRNA_flanking_region_G+C_prop']
        self.kmers_feat_cols = ['AAA', 'AAT', 'AAG', 'AAC', 'ATA', 'ATT', 'ATG', 'ATC', 'AGA', 'AGT', 'AGG', 'AGC', 'ACA',
                                'ACT', 'ACG', 'ACC', 'TAA', 'TAT', 'TAG', 'TAC', 'TTA', 'TTT', 'TTG', 'TTC', 'TGA', 'TGT',
                                'TGG', 'TGC', 'TCA', 'TCT', 'TCG', 'TCC', 'GAA', 'GAT', 'GAG', 'GAC', 'GTA', 'GTT', 'GTG',
                                'GTC', 'GGA', 'GGT', 'GGG', 'GGC', 'GCA', 'GCT', 'GCG', 'GCC', 'CAA', 'CAT', 'CAG', 'CAC',
                                'CTA', 'CTT', 'CTG', 'CTC', 'CGA', 'CGT', 'CGG', 'CGC', 'CCA', 'CCT', 'CCG', 'CCC']

    @staticmethod
    def get_const_hyperparams(**kwargs) -> Dict[str, object]:
        """   xgboost.__version__ = '1.5.0'  """
        logger.debug(f"getting const kXGBoost hyperparams")
        if xgboost.__version__ != '1.5.0':
            logger.warning(f"This model handler supports xgboost version 1.5.0, but got version {xgboost.__version__}")

        const_model_hyperparams = {
            'n_estimators': 500,  # Number of boosting rounds
            # 'max_depth': 9,  # Maximum tree depth for base learners
            # 'learning_rate': ,  # Boosting learning rate (xgb’s “eta”)
            # 'subsample':,  # Subsample ratio of the training instance
            # 'colsample_bytree':, # Subsample ratio of columns when constructing each tree
        }
        return const_model_hyperparams

    @staticmethod
    def get_model_args(objective='binary:logistic', seed: int = 22, verbosity: int = 1, eval_metric='logloss',
                       early_stopping_rounds: int = 3, verbose: bool = False, **kwargs) -> Dict[str, object]:
        """   xgboost.__version__ = '1.5.0'  """
        logger.debug(f"getting kXGBoost model_args")
        if xgboost.__version__ != '1.5.0':
            logger.warning(f"This model handler supports xgboost version 1.5.0, but got version {xgboost.__version__}")

        model_args = {
            # 1 - constructor args
            'objective': objective,  # the learning task and the corresponding learning objective or a custom objective function to be used.  optins: 'binary:hinge', 'binary:logistic', 'binary:logitraw'
            'random_state': seed,
            'verbosity': verbosity,
            # 2 - fit args
            'eval_metric': eval_metric,  # todo - select metric. # Metric used for monitoring the training result and early stopping. options: 'error',
            'early_stopping_rounds': early_stopping_rounds,   # todo - select rounds. # Activates early stopping. Validation metric needs to improve at least once in every early_stopping_rounds round(s) to continue training.
            'verbose': verbose
        }
        return model_args

    @staticmethod
    def eval_dataset(trained_model: XGBClassifier, X: pd.DataFrame, y: list, is_binary: bool, num_classes: int,
                     dataset_nm: str = None, **kwargs) -> (Dict[str, object], np.array):
        """   xgboost.__version__ = '1.5.0'  """
        assert pd.isnull(X).sum().sum() == 0, "X contains Nan values"
        y_pred = trained_model.predict_proba(X=X)
        if is_binary:
            y_score = y_pred[:, 1]
            thresholds = kwargs.get('y_pred_thresholds_features', None)
            if thresholds is not None:
                scores = calc_binary_classification_metrics_using_prob_y_score_and_y_pred_thresholds(
                    y_true=y, y_score=y_score, y_pred_thresholds=thresholds, dataset_nm=dataset_nm)
            else:
                roc_max_fpr = kwargs.get('roc_max_fpr', None)
                scores = calc_binary_classification_metrics_using_y_score(y_true=y, y_score=y_score,
                                                                          roc_max_fpr=roc_max_fpr,
                                                                          dataset_nm=dataset_nm)
        else:
            # todo - for multiclass
            y_score: np.array = None
            scores = calc_multiclass_classification_scores(y_true=y, y_score=y_score, num_classes=num_classes,
                                                           dataset_nm=dataset_nm)
        return scores, y_score

    @staticmethod
    def get_predictions_df(metadata: pd.DataFrame, features: pd.DataFrame, y_true: list, y_score: np.array,
                           out_col_y_true: str = "y_true", out_col_y_score: str = "y_score",
                           sort_df: bool = True) -> pd.DataFrame:
        is_length_compatible = len(metadata) == len(features) == len(y_true) == len(y_score)
        assert is_length_compatible, "unq_intr, y_true, y_score and metadata are not compatible in length"
        assert pd.isnull(features).sum().sum() == sum(pd.isnull(y_true)) == sum(pd.isnull(y_score)) == 0, "nulls in data"

        _df = pd.concat(objs=[metadata, features], axis=1).copy()
        _df[out_col_y_true] = y_true
        _df[out_col_y_score] = y_score
        # for testing
        scores = calc_binary_classification_metrics_using_y_score(y_true=y_true, y_score=y_score)
        for k, v in scores.items():
            print(f"{k}: {v}")
        if sort_df:
            _df = _df.sort_values(by=out_col_y_score, ascending=False).reset_index(drop=True)
        return _df

    def predict_on_folds(self, model_args: dict, cv_data: Dict[int, Dict[str, object]], **kwargs) -> \
            (Dict[int, Dict[str, Dict[str, float]]], Dict[int, pd.DataFrame], Dict[int, dict], Dict[int, dict]):
        """
        Returns
        -------
        cv_scores - Dict in the following format:  {
            <fold>: {
                'val': {
                    <score_nm>: score
                    }
        }
        cv_predictions_dfs - Dict in the following format:  {
            <fold>: pd.DataFrame
        }
        cv_training_history - Dict in the following format:  {
            <fold>: {
                'train': OrderedDict
                'validation': OrderedDict
            }
        }
        """
        logger.debug("predicting on folds")
        cv_scores, cv_training_history, cv_predictions_dfs, cv_shap_args = {}, {}, {}, {}
        for fold, fold_data in cv_data.items():
            logger.debug(f"starting fold {fold}")
            # extract metadata
            metadata_val = fold_data.get('metadata_val')
            if metadata_val is None:
                metadata_val = fold_data.get('metadata_test')
                if metadata_val is not None:
                    metadata_val = metadata_val[['sRNA_EcoCyc_accession_id', 'sRNA',
                                                 'mRNA_EcoCyc_accession_id', 'mRNA']]
            # predict
            scores, predictions, training_history, _, shap_args = \
                self.train_and_test(X_train=fold_data['X_train'], y_train=fold_data['y_train'],
                                    X_test=fold_data['X_val'], y_test=fold_data['y_val'], model_args=model_args,
                                    metadata_train=fold_data['metadata_train'], metadata_test=metadata_val, **kwargs)
            # 2.1 - fold's val scores
            cv_scores[fold] = {'val': scores['test']}
            # 2.2 - fold's predictions df
            y_val_pred = predictions['test_pred']
            X_val = fold_data['X_val']
            y_val = fold_data['y_val']
            cv_predictions_dfs[fold] = get_predictions_df(X=X_val, y_true=y_val, y_score=y_val_pred,
                                                          metadata=metadata_val)
            # 2.3 - fold's training history
            cv_training_history[fold] = training_history
            # 2.4 - fold's shap args
            cv_shap_args[fold] = shap_args
        return cv_scores, cv_predictions_dfs, cv_training_history, cv_shap_args

    def process_loco_fold_data(self, fold_data: Dict[str, pd.DataFrame], feature_cols: List[str],
                               label_col: str = 'interaction_label'):
        """

        Parameters
        ----------
        fold_data
        feature_cols
        label_col

        Returns
        -------

        """
        unq_train = fold_data['unq_train']
        unq_test = fold_data['unq_test']
        # X
        X_train = pd.DataFrame(unq_train[feature_cols])
        X_test = pd.DataFrame(unq_test[feature_cols])
        # y
        y_train = np.array(unq_train[label_col])
        y_test = np.array(unq_test[label_col])
        # metadata
        meta_cols_to_rem = ['y_score', 'COPRA_sRNA_is_missing', 'COPRA_sRNA', 'COPRA_mRNA', 'COPRA_mRNA_locus_tag',
                            'COPRA_pv', 'COPRA_fdr', 'COPRA_NC_000913', 'COPRA_mRNA_not_in_output',
                            'COPRA_validated_pv', 'COPRA_validated_score']
        meta_cols_train = [c for c in unq_train.columns.values if c not in feature_cols + [label_col] + meta_cols_to_rem]
        meta_train = pd.DataFrame(unq_train[meta_cols_train])
        meta_cols_test = [c for c in unq_test.columns.values if c not in feature_cols + [label_col] + meta_cols_to_rem]
        meta_test = pd.DataFrame(unq_test[meta_cols_test])

        assert len(X_train) == len(y_train) == len(meta_train), "length mismatch"
        assert len(X_test) == len(y_test) == len(meta_test), "length mismatch"

        out_fold_data = {
            'X_train': X_train,
            'y_train': y_train,
            'metadata_train': meta_train,
            'X_val': X_test,
            'y_val': y_test,
            'metadata_test': meta_test
        }
        return out_fold_data

    def loco_impl(self, _loco_folds: Dict[str, dict], feature_cols: List[str], model_args: dict,
                  nm_4_feat_selection: str = None, **kwargs) -> Dict[str, object]:
        # 1 - prepare LOCO folds data
        for cond, fold_data in _loco_folds.items():
            m_fold_data = self.process_loco_fold_data(fold_data=fold_data, feature_cols=feature_cols)
            _loco_folds[cond] = m_fold_data

        # 2 - k-mers feature selection per condition
        if nm_4_feat_selection:
            _path = '/home/shanisa/PhD/Data/models_training_and_benchmarking/outputs/all_preds/0.2.1/results/loco_results'
            run_mrmr_feat_selection = False
            if run_mrmr_feat_selection:
                from run_mRMR import run_mrmr
                write_df(df=_loco_folds, file_path=join(_path, f'loco_folds_{nm_4_feat_selection}_4_feat_selection.pickle'))
                label_col = 'y_true'
                kmers_cols = ['AAA', 'AAT', 'AAG', 'AAC', 'ATA', 'ATT', 'ATG', 'ATC', 'AGA', 'AGT', 'AGG', 'AGC', 'ACA',
                              'ACT', 'ACG', 'ACC', 'TAA', 'TAT', 'TAG', 'TAC', 'TTA', 'TTT', 'TTG', 'TTC', 'TGA', 'TGT',
                              'TGG', 'TGC', 'TCA', 'TCT', 'TCG', 'TCC', 'GAA', 'GAT', 'GAG', 'GAC', 'GTA', 'GTT', 'GTG',
                              'GTC', 'GGA', 'GGT', 'GGG', 'GGC', 'GCA', 'GCT', 'GCG', 'GCC', 'CAA', 'CAT', 'CAG', 'CAC',
                              'CTA', 'CTT', 'CTG', 'CTC', 'CGA', 'CGT', 'CGG', 'CGC', 'CCA', 'CCT', 'CCG', 'CCC']

                # 2 - LOCO swap folds - feature selection
                # _loco_folds = read_df(file_path=join(_path, f'loco_folds_{nm_4_feat_selection}_4_feat_selection.pickle'))
                loco_cnd_to_selected_kmers = {}
                for cnd, cnd_data in _loco_folds.items():
                    logger.debug(f"cond: {cnd}")
                    train_df = cnd_data['X_train']
                    train_df[label_col] = cnd_data['y_train']
                    cnd_selected_kmers = run_mrmr(df=train_df, feature_cols=kmers_cols, label_col=label_col)
                    loco_cnd_to_selected_kmers[cnd] = cnd_selected_kmers
                write_df(df=loco_cnd_to_selected_kmers, file_path=join(_path, f'loco_{nm_4_feat_selection}_selected_kmers.pickle'))
            else:
                loco_cnd_to_selected_kmers = read_df(file_path=join(_path, f'loco_{nm_4_feat_selection}_selected_kmers.pickle'))

            # 2 - select kmer features per condition
            for cond, fold_data in _loco_folds.items():
                kmer_features = loco_cnd_to_selected_kmers[cond]
                fold_data['X_train'] = fold_data['X_train'][self.intr_feat_cols + kmer_features]
                fold_data['X_val'] = fold_data['X_val'][self.intr_feat_cols + kmer_features]
                logger.debug(f"cond = {cond}: features = {len(self.intr_feat_cols + kmer_features)} (kmers = {len(kmer_features)})")

        # 3 - run LOCO
        loco_scores, loco_predictions_dfs, loco_training_history, loco_shap_args = \
            self.predict_on_folds(model_args=model_args, cv_data=_loco_folds, **kwargs)
        res = {
            "cv_scores": loco_scores,
            "cv_predictions_dfs": loco_predictions_dfs,
            "cv_training_history": loco_training_history,
            "cv_shap_args": loco_shap_args
        }

        return res

    def run_loco(self, model_args: dict, feature_cols: List[str], **kwargs) -> Dict[str, Dict[str, object]]:
        """
                        'loco_folds': loco_folds,
                'loco_folds_swap': loco_folds_swap,
        Returns
        -------
        # todo
        cv_outs - Dict in the following format:  {
            'neg_syn': {
                'cv_scores': <cv_scores>,
                'cv_prediction_dfs': <cv_prediction_dfs>,
                'cv_training_history': <cv_training_history>
            },
            'neg_rnd': {
                'cv_scores': <cv_scores>,
                'cv_prediction_dfs': <cv_prediction_dfs>,
                'cv_training_history': <cv_training_history>
            }
        }

        Where:
        cv_scores - Dict in the following format:  {
            <fold>: {
                'val': {
                    <score_nm>: score
                    }
        }
        cv_prediction_dfs - Dict in the following format:  {
            <fold>: pd.DataFrame
        }
        cv_training_history - Dict in the following format:  {
            <fold>: {
                'train': OrderedDict
                'validation': OrderedDict
            }
        }
        cv_data - Dict in the following format:  {
            <fold>: {
                "X_train": pd.DataFrame (n_samples, N_features),
                "y_train": list (n_samples,),
                "X_val": pd.DataFrame (k_samples, K_features),
                "y_val": list (k_samples,),
                "metadata_train": pd.DataFrame (n_samples, T_features),    - Optional (in case metadata is not None)
                "metadata_val": list (k_samples,)    - Optional (in case metadata is not None)
            }
        }
        """
        logger.debug(f"running LOCO")
        loco_folds = kwargs['f_loco_folds'].copy()
        loco_folds_swap = kwargs['f_loco_folds_swap'].copy()
        loco_outs = {}
        feature_cols = self.feature_cols
        # -------------------- All features
        # 1.1 - random negatives
        res = self.loco_impl(_loco_folds=loco_folds, feature_cols=feature_cols, model_args=model_args, **kwargs)
        loco_outs['loco'] = res
        # 1.2 - swap negatives
        res = self.loco_impl(_loco_folds=loco_folds_swap, feature_cols=feature_cols, model_args=model_args,
                             nm_4_feat_selection='swap', **kwargs)
        loco_outs['loco_swap'] = res

        return loco_outs

    def run_additional_cross_validation(self, model_args: dict, feature_cols: List[str], **kwargs) -> \
            Dict[str, Dict[str, object]]:
        """
        Returns
        -------
        cv_outs - Dict in the following format:  {
            'neg_syn': {
                'cv_scores': <cv_scores>,
                'cv_prediction_dfs': <cv_prediction_dfs>,
                'cv_training_history': <cv_training_history>
            },
            'neg_rnd': {
                'cv_scores': <cv_scores>,
                'cv_prediction_dfs': <cv_prediction_dfs>,
                'cv_training_history': <cv_training_history>
            }
        }

        Where:
        cv_scores - Dict in the following format:  {
            <fold>: {
                'val': {
                    <score_nm>: score
                    }
        }
        cv_prediction_dfs - Dict in the following format:  {
            <fold>: pd.DataFrame
        }
        cv_training_history - Dict in the following format:  {
            <fold>: {
                'train': OrderedDict
                'validation': OrderedDict
            }
        }
        cv_data - Dict in the following format:  {
            <fold>: {
                "X_train": pd.DataFrame (n_samples, N_features),
                "y_train": list (n_samples,),
                "X_val": pd.DataFrame (k_samples, K_features),
                "y_val": list (k_samples,),
                "metadata_train": pd.DataFrame (n_samples, T_features),    - Optional (in case metadata is not None)
                "metadata_val": list (k_samples,)    - Optional (in case metadata is not None)
            }
        }
        """
        logger.debug(f"running kXGBoost additional cross validation")
        cv_outs = {}
        cv_outs['neg_syn'] = None
        feature_cols = self.feature_cols

        # 2 - get cv_data for random negatives
        logger.debug(f"\n getting cv_data for random negatives")
        df = kwargs['train_neg_rnd_w_kmers'].copy()
        label_col = 'y_true'
        fold_col = "fold"
        X = df[feature_cols]
        y = np.array(df[label_col])
        meta_cols = [c for c in df.columns.values if c not in feature_cols + [label_col]]
        meta = pd.DataFrame(df[meta_cols])
        cv_data_neg_rnd = split_cv_data(X=X, y=y, metadata=meta, fold_col=fold_col)
        # cv_data_neg_rnd = cls.process_cv_folds_swap(cv_data=cv_data_neg_rnd, **kwargs)
        cv_scores, cv_predictions_dfs, cv_training_history, cv_shap_args = \
            self.predict_on_folds(model_args=model_args, cv_data=cv_data_neg_rnd, **kwargs)
        cv_outs['neg_rnd'] = {
            "cv_scores": cv_scores,
            "cv_predictions_dfs": cv_predictions_dfs,
            "cv_training_history": cv_training_history,
        }

        return cv_outs

    def train_and_test(self, X_train: pd.DataFrame, y_train: List[int], X_test: pd.DataFrame, y_test: List[int],
                       model_args: dict, metadata_train: pd.DataFrame = None, metadata_test: pd.DataFrame = None,
                       **kwargs) -> (Dict[str, dict], Dict[str, object], Dict[str, object], Dict[str, object],
                                     Dict[str, object]):
        """
        xgboost.__version__ = '1.5.0'

        Parameters
        ----------
        X_train: pd.DataFrame (n_samples, N_features),
        y_train: list (n_samples,),
        X_test: pd.DataFrame (t_samples, N_features),
        y_test: list (t_samples,)
        model_args: Dict of model's constructor and fit() arguments
        metadata_train: Optional - pd.DataFrame (n_samples, T_features),
        metadata_test: Optional - pd.DataFrame (t_samples, T_features)
        kwargs

        Returns
        -------

        scores - Dict in the following format:  {
            'test': {
                <score_nm>: score
                }
        }
        predictions - Dict in the following format:  {
            'test_pred': array-like (t_samples,)
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
            "X_test": pd.DataFrame,
            "y_train": List[int],
            "y_test": List[int]
        }
        """
        logger.debug("training an kXGBoost classifier")
        if xgboost.__version__ != '1.5.0':
            logger.warning(f"This model handler supports xgboost version 1.5.0, but got version {xgboost.__version__}")

        # 1 - construct
        model = XGBClassifier(
            n_estimators=model_args.get('n_estimators'),
            max_depth=model_args.get('max_depth'),
            objective=model_args['objective'],
            random_state=model_args['random_state'],
            use_label_encoder=False,
            verbosity=model_args['verbosity']
        )
        # 2 - define fit() params
        # 2.1 - split train into training and validation
        val_size = 0.1
        folds_data = get_stratified_cv_folds(X=X_train, y=np.array(y_train), n_splits=int(1/val_size),
                                             metadata=metadata_train)
        train_val_data = folds_data[0]  # randomly select a fold
        del folds_data
        eval_set = [(np.array(train_val_data['X_train']), train_val_data['y_train']),
                    (np.array(train_val_data['X_val']), train_val_data['y_val'])]
        # 2.2 - extract additional fit() params form config
        eval_metric = model_args['eval_metric']
        early_stopping_rounds = model_args['early_stopping_rounds']  # if not None - early stopping is activated
        verbose = model_args['verbose']  # if not None - early stopping is activated

        # 3 - train
        trained_model = model.fit(X=np.array(train_val_data['X_train']), y=train_val_data['y_train'], eval_set=eval_set,
                                  eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        training_history = {
            "train": trained_model.evals_result()['validation_0'],
            "validation": trained_model.evals_result()['validation_1']
        }

        # 4 - calc scores
        predictions, scores = {}, {}
        num_classes = len(np.unique(y_train))
        is_binary = sorted(np.unique(y_train)) == [0, 1]
        # 4.1 - test
        logger.debug("evaluating test set")
        test_scores, test_y_pred = self.eval_dataset(trained_model=trained_model, X=X_test, y=y_test,
                                                     is_binary=is_binary, num_classes=num_classes, **kwargs)
        # for k, v in test_scores.items():
        #     print(f"{k}: {v}")
        test_predictions_df = self.get_predictions_df(metadata=metadata_test, features=X_test, y_true=y_test,
                                                      y_score=test_y_pred)
        scores.update({'test': test_scores})
        predictions.update({'test_pred': test_y_pred,
                            'test_predictions_df': test_predictions_df})
        # 5 - SHAP args
        shap_args = {
            "model_nm": "kXGBoost",
            "trained_model": trained_model,
            "X_train": train_val_data['X_train'].reset_index(drop=True),
            "X_val": train_val_data['X_val'].reset_index(drop=True),
            "X_test": X_test,
            "y_train": train_val_data['y_train'],
            "y_test": y_test
        }

        return scores, predictions, training_history, train_val_data, shap_args

    def train_and_test_w_negatives(self, train_df: pd.DataFrame, train_label_col: str,
                                   X_test: pd.DataFrame, y_test: List[int], model_args: dict,
                                   metadata_test: pd.DataFrame = None, train_set_nm: str = '', **kwargs) -> \
            (Dict[str, dict], Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object]):
        """
        sklearn.__version__ = '0.24.0'

        Parameters
        ----------
        X_test: pd.DataFrame (t_samples, N_features),
        y_test: list (t_samples,)
        model_args: Dict of model's constructor and fit() arguments
        feature_cols
        metadata_test: Optional - pd.DataFrame (t_samples, T_features)
        kwargs

        Returns
        -------

        scores - Dict in the following format:  {
            'test': {
                <score_nm>: score
                }
        }
        predictions - Dict in the following format:  {
            'test_pred': array-like (t_samples,)
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
        }
        """
        logger.debug(f"training kXGBoost - with random negatives {train_set_nm}")
        _path = '/home/shanisa/PhD/Data/models_training_and_benchmarking/outputs/all_preds/0.2.1/results/benchmarking_results'
        run_mrmr_feat_selection = False
        if run_mrmr_feat_selection:
            from run_mRMR import run_mrmr
            write_df(df=train_df, file_path=join(_path, 'train_df.csv'))
            label_col = 'y_true'
            kmers_cols = ['AAA', 'AAT', 'AAG', 'AAC', 'ATA', 'ATT', 'ATG', 'ATC', 'AGA', 'AGT', 'AGG', 'AGC', 'ACA',
                          'ACT', 'ACG', 'ACC', 'TAA', 'TAT', 'TAG', 'TAC', 'TTA', 'TTT', 'TTG', 'TTC', 'TGA', 'TGT',
                          'TGG', 'TGC', 'TCA', 'TCT', 'TCG', 'TCC', 'GAA', 'GAT', 'GAG', 'GAC', 'GTA', 'GTT', 'GTG',
                          'GTC', 'GGA', 'GGT', 'GGG', 'GGC', 'GCA', 'GCT', 'GCG', 'GCC', 'CAA', 'CAT', 'CAG', 'CAC',
                          'CTA', 'CTT', 'CTG', 'CTC', 'CGA', 'CGT', 'CGG', 'CGC', 'CCA', 'CCT', 'CCG', 'CCC']

            # 1 - entire train - feature selection
            # train_df = read_df(file_path=join(_path, 'train_df.csv'))
            train_selected_kmers = run_mrmr(df=train_df, feature_cols=kmers_cols, label_col=label_col)
            write_df(df=train_selected_kmers, file_path=join(_path, 'train_selected_kmers.pickle'))
        else:
            train_selected_kmers = read_df(file_path=join(_path, 'train_selected_kmers.pickle'))

        # 1 - define features
        # feature_cols = self.feature_cols
        feature_cols = self.intr_feat_cols + train_selected_kmers
        logger.debug(f"selected kmers (entire train): {train_selected_kmers}")

        # 2 - prepare the train set
        X_train = train_df[feature_cols]
        y_train = np.array(train_df[train_label_col])
        meta_cols = [c for c in train_df.columns.values if c not in feature_cols + [train_label_col]]
        metadata_train = pd.DataFrame(train_df[meta_cols])
        # 1.2 - reset train_df indexes
        X_train = X_train.reset_index(drop=True)
        metadata_train = metadata_train.reset_index(drop=True)

        # 3 - prepare the test set - extract features
        test_w_features = pd.concat(objs=[X_test, metadata_test], axis=1).copy()
        test_w_features = add_srnarftarget_features(df=test_w_features, srna_seq_col='sRNA_sequence_Eco',
                                                    mrna_seq_col='mRNA_sequence_Eco')
        X_test = test_w_features[feature_cols]

        # 4 - train and test - with random negatives
        scores, predictions, training_history, train_val_data, shap_args = \
            self.train_and_test(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_args=model_args,
                                metadata_train=metadata_train, metadata_test=metadata_test, **kwargs)

        return scores, predictions, training_history, train_val_data, shap_args
