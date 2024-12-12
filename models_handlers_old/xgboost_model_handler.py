import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier
from typing import Dict, List
from models_handlers_old.model_handlers_utils import calc_binary_classification_metrics_using_y_score, \
    get_stratified_cv_folds, get_predictions_df
import logging
logger = logging.getLogger(__name__)


class XGBModelHandler(object):
    """
    Class to handle XGBoost model training and evaluation.
    xgboost.__version__ = '1.5.0'
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier

    Parameters
    ----------

    Attributes
    ----------
    Same as parameters
    """
    def __init__(self):
        super(XGBModelHandler, self).__init__()

    @staticmethod
    def get_model_args(objective='binary:logistic', seed: int = 22, verbosity: int = 1, eval_metric='logloss',
                       early_stopping_rounds: int = 3, verbose: bool = False) -> Dict[str, object]:
        """   xgboost.__version__ = '1.5.0'  """
        logger.debug(f"getting XGBoost model_args")
        if xgboost.__version__ != '1.5.0':
            logger.warning(f"This model handler supports xgboost version 1.5.0, but got version {xgboost.__version__}")
        model_args = {
            # 1 - constructor args
            'objective': objective,
            'random_state': seed,
            'verbosity': verbosity,
            # 2 - fit args
            'eval_metric': eval_metric,
            'early_stopping_rounds': early_stopping_rounds,
            'verbose': verbose
        }
        return model_args

    @staticmethod
    def eval_dataset(trained_model: XGBClassifier, X: pd.DataFrame, y: list, is_binary: bool = True,
                     dataset_nm: str = None, **kwargs) -> (Dict[str, object], np.array):
        """   xgboost.__version__ = '1.5.0'  """
        assert pd.isnull(X).sum().sum() == 0, "X contains Nan values"
        y_pred = trained_model.predict_proba(X=X)
        if is_binary:
            y_score = y_pred[:, 1]
            roc_max_fpr = kwargs.get('roc_max_fpr', None)
            scores = calc_binary_classification_metrics_using_y_score(y_true=y, y_score=y_score,
                                                                      roc_max_fpr=roc_max_fpr,
                                                                      dataset_nm=dataset_nm)
        else:
            y_score: np.array = None
            scores = None
        return scores, y_score

    @classmethod
    def run_cross_validation(cls, X: pd.DataFrame, y: List[int], n_splits: int, model_args: dict,
                             metadata: pd.DataFrame = None, **kwargs) -> (Dict[int, pd.DataFrame], Dict[int, dict]):
        """
        Returns
        -------

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
        # 1 - split dataset into folds
        cv_data = get_stratified_cv_folds(X=X, y=np.array(y), n_splits=n_splits, metadata=metadata)
        # 2 - predict on folds
        cv_training_history = {}
        cv_predictions_dfs = {}
        for fold, fold_data in cv_data.items():
            logger.debug(f"starting fold {fold}")
            # 2.1 - predict in fold's val
            predictions, training_history = \
                cls.train_and_test(X_train=fold_data['X_train'], y_train=fold_data['y_train'],
                                   X_test=fold_data['X_val'], y_test=fold_data['y_val'], model_args=model_args,
                                   metadata_train=fold_data['metadata_train'], **kwargs)
            # 2.2 - fold's training history
            cv_training_history[fold] = training_history
            # 2.3 - fold's predictions df
            y_val_pred = predictions['test_pred']
            X_val = fold_data['X_val']
            y_val = fold_data['y_val']
            metadata_val = fold_data.get('metadata_val')
            cv_predictions_dfs[fold] = get_predictions_df(X=X_val, y_true=y_val, y_score=y_val_pred,
                                                          metadata=metadata_val)
            
        return cv_predictions_dfs, cv_training_history

    @classmethod
    def train_and_test(cls, X_train: pd.DataFrame, y_train: List[int], X_test: pd.DataFrame, y_test: List[int],
                       model_args: dict, metadata_train: pd.DataFrame = None, metadata_test: pd.DataFrame = None,
                       **kwargs) -> (Dict[str, dict], Dict[str, object], Dict[str, object], Dict[str, object]):
        """
        xgboost.__version__ = '1.5.0'

        Parameters
        ----------
        X_train: pd.DataFrame (n_samples, N_features),
        y_train: list (n_samples,),
        X_test: pd.DataFrame (t_samples, N_features),
        y_test: list (t_samples,)
        model_args: Dict of model's constructor and fit() arguments
        metadata_train: Optional - pd.DataFrame (n_samples, T_features) - will be used for early stopping split
        metadata_test: Optional - pd.DataFrame (t_samples, T_features)
        kwargs

        Returns
        -------

        predictions - Dict in the following format:  {
            'test_pred': array-like (t_samples,) - model's prediction scores, ordered as y test
            'out_test_pred': pd.DataFrame including the following information:
                sRNA accession id, mRNA accession id, interaction label (y_true),
                model's prediction score (y_score), metadata columns.
        }
        training_history - Dict in the following format:  {
            'train': OrderedDict
            'validation': OrderedDict
        }
        """
        logger.debug("training an XGBoost classifier")
        if xgboost.__version__ != '1.5.0':
            logger.warning(f"This model handler supports xgboost version 1.5.0, but got version {xgboost.__version__}")

        # 1 - construct
        model = XGBClassifier(
            objective=model_args['objective'],
            random_state=model_args['random_state'],
            use_label_encoder=False,
            verbosity=model_args['verbosity']
        )
        # 2 - define fit() params
        # 2.1 - split train into training and validation for early stopping
        val_size = 0.1
        folds_data = get_stratified_cv_folds(X=X_train, y=np.array(y_train), n_splits=int(1/val_size),
                                             metadata=metadata_train)
        train_val_data = folds_data[0]  # randomly select a split for early stopping
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

        # 4 - predict
        predictions = {}
        num_classes = len(np.unique(y_train))
        is_binary = sorted(np.unique(y_train)) == [0, 1]
        logger.debug("evaluating test set")
        test_scores, test_y_pred = cls.eval_dataset(trained_model=trained_model, X=X_test, y=y_test,
                                                    is_binary=is_binary, num_classes=num_classes, **kwargs)

        # 5 - update outputs
        # 5.1 - test predictions df
        out_test_pred = get_predictions_df(X=X_test, y_true=list(y_test), y_score=test_y_pred, metadata=metadata_test)
        # 5.2 - XGBoost prediction scores (test_y_pred)
        predictions.update({'test_pred': test_y_pred, 'out_test_pred': out_test_pred})

        return predictions, training_history
