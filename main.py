from typing import Dict
import pandas as pd
from os.path import join
from utils.utils_general import get_logger_config_dict, write_df
from data_handlers.data_handler import DataHandler
from models_handlers.graph_rna_model_handler import GraphRNAModelHandler
from models_handlers.xgboost_model_handler import XGBModelHandler
from models_handlers.rf_model_handler import RFModelHandler
import logging.config
logging.config.dictConfig(get_logger_config_dict(filename="main", file_level='DEBUG'))
logger = logging.getLogger(__name__)


def main():
    # ----- configuration
    data_path = "/sise/home/shanisa/GraphRNA/data"
    outputs_path = "/sise/home/shanisa/GraphRNA/outputs"

    # ----- load data
    train_fragments, test_complete, test_filtered, kwargs = load_data(data_path=data_path)

    # ----- run GraphRNA
    graph_rna = GraphRNAModelHandler()
    test = test_complete
    cv_predictions_dfs, test_predictions = \
        train_and_evaluate(model_h=graph_rna, train_fragments=train_fragments, test=test, **kwargs)
    write_df(df=test_predictions, file_path=join(outputs_path, f"test_predictions_GraphRNA.csv"))

    # ----- run XGBoost
    xgb = XGBModelHandler()
    test = test_filtered
    cv_predictions_dfs, test_predictions = \
        train_and_evaluate(model_h=xgb, train_fragments=train_fragments, test=test, **kwargs)
    write_df(df=test_predictions, file_path=join(outputs_path, f"test_predictions_XGBoost.csv"))

    # ----- run RandomForest
    rf = RFModelHandler()
    test = test_filtered
    cv_predictions_dfs, test_predictions = \
        train_and_evaluate(model_h=rf, train_fragments=train_fragments, test=test, **kwargs)
    write_df(df=test_predictions, file_path=join(outputs_path, f"test_predictions_RandomForest.csv"))

    return


def load_data(data_path: str):
    """
    :param data_path: str
    :return:
    train_fragments, test_complete, test_filtered - dicts in the following format:
        {
            'X': pd.DataFrame,
            'y': List[int],
            'metadata': pd.DataFrame
        }
    kwargs - dict in the following format:
        {
            'srna_eco': pd.DataFrame,
            'mrna_eco': pd.DataFrame,
            'se_acc_col': str (the column in srna_eco containing unique id per sRNA),
            'me_acc_col': str (the column in mrna_eco containing unique id per mRNA)
        }
    """
    # ----- data
    # 1 - load interactions datasets
    """
    train_fragments: includes 14806 interactions: 7403 positives, 7403 synthetic negatives (see evaluation 1).
                     (synthetic samples are ignored when training GraphRNA)
                     RNA sequences are chimeric fragments
    test_complete: includes 391 interactions: 227 positives, 164 negatives (see evaluation 4).
    test_filtered: includes 342 interactions: 199 positives, 143 negatives (see evaluation 3).
    """
    dh = DataHandler(data_path=data_path)
    train_fragments, test_complete, test_filtered = dh.load_interactions_datasets()
    # 2 - load RNA data  (for GraphRNA)
    """
    srna_eco: includes 94 unique sRNAs of Escherichia coli K12 MG1655 (NC_000913) from EcoCyc.
    mrna_eco: includes 4300 unique mRNAs of Escherichia coli K12 MG1655 (NC_000913) from EcoCyc.

    Note: sRNA/mRNA accession ids in the RNA data must match the accession ids in the interactions datasets.
    """
    srna_eco, mrna_eco, srna_eco_accession_id_col, mrna_eco_accession_id_col = dh.load_rna_data()
    # 2.1 - update kwargs
    kwargs = {
        'srna_eco': srna_eco,
        'mrna_eco': mrna_eco,
        'se_acc_col': srna_eco_accession_id_col,
        'me_acc_col': mrna_eco_accession_id_col
    }

    return train_fragments, test_complete, test_filtered, kwargs


def train_and_evaluate(model_h, train_fragments: Dict[str, object], test: Dict[str, object], **kwargs) -> \
        (Dict[int, pd.DataFrame], pd.DataFrame):
    """
    Returns
    -------

    cv_prediction_dfs - Dict in the following format:  {
        <fold>: pd.DataFrame including the following information:
                sRNA accession id, mRNA accession id, interaction label (y_true),
                model's prediction score (y_score OR y_graph_score), metadata columns.
        }
    }

    test_predictions_df - pd.DataFrame including the following information:
                          sRNA accession id, mRNA accession id, interaction label (y_true),
                          model's prediction score (y_score OR y_graph_score), metadata columns.
    """
    # 1 - define model args
    model_args = model_h.get_model_args()
    # 2 - run cross validation
    cv_n_splits = 10
    cv_predictions_dfs, cv_training_history = \
        model_h.run_cross_validation(X=train_fragments['X'], y=train_fragments['y'],
                                     metadata=train_fragments['metadata'], n_splits=cv_n_splits,
                                     model_args=model_args, **kwargs)
    # 3 - train and test
    predictions, training_history = \
        model_h.train_and_test(X_train=train_fragments['X'], y_train=train_fragments['y'], X_test=test['X'],
                               y_test=test['y'], model_args=model_args,
                               metadata_train=train_fragments['metadata'], metadata_test=test['metadata'], **kwargs)
    test_predictions_df = predictions['out_test_pred']

    return cv_predictions_dfs, test_predictions_df


if __name__ == "__main__":
    main()
