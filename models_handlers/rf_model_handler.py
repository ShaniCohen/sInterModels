import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier
from models_handlers.model_handlers_utils import calc_binary_classification_metrics_using_y_score,\
    get_stratified_cv_folds, get_predictions_df
import logging
logger = logging.getLogger(__name__)


class RFModelHandler(object):
    """
    Class to ...
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    Parameters
    ----------

    Attributes
    ----------
    Same as parameters
    """
    def __init__(self):
        super(RFModelHandler, self).__init__()

    @staticmethod
    def get_model_args(seed: int = 22, verbose: int = 0) -> Dict[str, object]:
        """   sklearn.__version__ = '0.24.0'  """
        logger.debug("getting RandomForestClassifier model_args")
        model_args = {
            # 1 - constructor args
            'random_state': seed,
            'verbose': verbose
        }
        model_hyperparams = {
            'n_estimators': 300,  # The number of trees in the forest
            'max_depth': None
        }
        model_args.update(model_hyperparams)
        return model_args

    @staticmethod
    def eval_dataset(trained_model: RandomForestClassifier, X: pd.DataFrame, y: list, is_binary: bool = True,
                     dataset_nm: str = None, **kwargs) -> (Dict[str, object], np.array):
        """   sklearn.__version__ = '0.24.0'  """
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

        Parameters
        ----------
        X: pd.DataFrame (n_samples, N_features),
        y: list (n_samples,),
        n_splits: number of folds for cross validation
        model_args: Dict of model's constructor and fit() arguments
        metadata: Optional - pd.DataFrame (n_samples, T_features)
        kwargs

        Returns
        -------

        cv_prediction_dfs - Dict in the following format:  {
            <fold>: pd.DataFrame including the following information:
                    sRNA accession id, mRNA accession id, interaction label (y_true), model's score (y_score),
                    metadata columns (including features).
        }
        }
        cv_training_history - Dict in the following format:  {
            <fold>: {
                'train': OrderedDict
                'validation': OrderedDict
            }
        }
        """
        logger.debug(f"running cross validation with {n_splits} folds")
        # 1 - split dataset into folds
        cv_data = get_stratified_cv_folds(X=X, y=np.array(y), n_splits=n_splits, metadata=metadata)
        # 2 - predict on folds
        cv_training_history = {}
        cv_prediction_dfs = {}
        for fold, fold_data in cv_data.items():
            logger.debug(f"starting fold {fold}")
            # 2.1 - predict on fold's val scores
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
            cv_prediction_dfs[fold] = get_predictions_df(X=X_val, y_true=y_val, y_score=y_val_pred,
                                                         metadata=metadata_val)

        return cv_prediction_dfs, cv_training_history

    @classmethod
    def train_and_test(cls, X_train: pd.DataFrame, y_train: List[int], X_test: pd.DataFrame, y_test: List[int],
                       model_args: dict, metadata_train: pd.DataFrame = None, metadata_test: pd.DataFrame = None,
                       **kwargs) -> (Dict[str, object], Dict[str, object]):
        """
        sklearn.__version__ = '0.24.0'

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
        logger.debug("training a RandomForest classifier")
        # 1 - construct
        model = RandomForestClassifier(random_state=model_args['random_state'],
                                       n_estimators=model_args['n_estimators'],
                                       max_depth=model_args['max_depth'],
                                       verbose=model_args['verbose'])

        # 2 - train
        trained_model = model.fit(X=np.array(X_train), y=y_train)
        training_history = {
            # "train": trained_model.evals_result()['validation_0'],
            # "validation": trained_model.evals_result()['validation_1']
        }

        # 3 - predict
        predictions = {}
        num_classes = len(np.unique(y_train))
        is_binary = sorted(np.unique(y_train)) == [0, 1]
        logger.debug("evaluating test set")
        test_scores, test_y_pred = cls.eval_dataset(trained_model=trained_model, X=X_test, y=y_test,
                                                    is_binary=is_binary, num_classes=num_classes, **kwargs)
        # 4 - update outputs
        # 4.1 - test predictions df
        out_test_pred = get_predictions_df(X=X_test, y_true=list(y_test), y_score=test_y_pred, metadata=metadata_test)
        # 4.2 - RandomForest prediction scores (test_y_pred)
        predictions.update({'test_pred': test_y_pred, 'out_test_pred': out_test_pred})

        return predictions, training_history
