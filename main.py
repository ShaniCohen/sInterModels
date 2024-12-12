from typing import Dict
import pandas as pd
from os.path import join
from utils_old.utils_general import get_logger_config_dict, write_df
from data_handlers_old.data_handler import DataHandler, _load_rna_data, _load_interactions_data
from models_handlers.kgraphrna_model_handler import kGraphRNAModelHandler
from models_handlers_old.graph_rna_model_handler import GraphRNAModelHandler
from models_handlers_old.xgboost_model_handler import XGBModelHandler
from models_handlers_old.rf_model_handler import RFModelHandler
from feature_extraction.feature_extractor import extract_local_inter_and_3_mer_diff_features
import logging.config
logging.config.dictConfig(get_logger_config_dict(filename="main", file_level='DEBUG'))
logger = logging.getLogger(__name__)


def main():
    # ----- configuration
    data_path = "/sise/home/shanisa/GraphRNA/data"
    data_path_2 = "/sise/home/shanisa/data_sInterModels"

    # ----- load data -----
    train, test_complete, test_filtered, kwargs = load_data(data_path=data_path)
    # Note: these are small example datasets (train and test) demonstrating the E2E flow
    train_example, test_example, srna_data, mrna_data, srna_acc_col, mrna_acc_col = \
        load_rna_and_inter_data(data_path=data_path_2)

    # ----- GraphRNA -----
    # 1 - train and test
    graph_rna = GraphRNAModelHandler()
    test = test_complete
    test_predictions = train_and_evaluate(model_h=graph_rna, train=train, test=test, **kwargs)

    # ----- kGraphRNA -----
    # 1 - train and test
    k_graph_rna = kGraphRNAModelHandler()
    test_predictions = train_and_test_graph_model(model_h=k_graph_rna, train=train_example, test=test_example,
                                                  srna_data=srna_data, mrna_data=mrna_data, srna_acc_col=srna_acc_col,
                                                  mrna_acc_col=mrna_acc_col)

    # ----- Decision Forests - sInterRF and sInterXGB -----
    # 1 - extract local interaction and 3-mer-diff features

    # 2 - train and test
    # -- sInterRF
    rf = RFModelHandler()
    test = test_filtered
    test_predictions = train_and_evaluate(model_h=rf, train=train, test=test, **kwargs)

    # -- sInterXGB
    xgb = XGBModelHandler()
    test = test_filtered
    test_predictions = train_and_evaluate(model_h=xgb, train=train, test=test, **kwargs)
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


def load_rna_and_inter_data(data_path: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str):
    """
    :param data_path: str
    :return:
    """
    # 1 - load interactions datasets for example
    train_example, test_example = _load_interactions_data(data_path=data_path)
    # 2 - load sRNA and mRNA data
    """
    srna_data: includes 94 unique sRNAs of Escherichia coli K12 MG1655 (NC_000913) from EcoCyc.
    mrna_data: includes 4300 unique mRNAs of Escherichia coli K12 MG1655 (NC_000913) from EcoCyc.

    Note: sRNA/mRNA accession ids in the RNA data must match the accession ids in the interactions datasets.
    """
    srna_data, mrna_data, srna_accession_id_col, mrna_accession_id_col = _load_rna_data(data_path=data_path)
    # 3 - asserts
    # 3.1 - sRNA acc col
    assert srna_accession_id_col in srna_data.columns.values
    assert srna_accession_id_col in train_example.columns.values
    assert srna_accession_id_col in test_example.columns.values
    # 3.2 - mRNA acc col
    assert mrna_accession_id_col in mrna_data.columns.values
    assert mrna_accession_id_col in train_example.columns.values
    assert mrna_accession_id_col in test_example.columns.values

    return train_example, test_example, srna_data, mrna_data, srna_accession_id_col, mrna_accession_id_col


def train_and_test_graph_model(model_h, train: pd.DataFrame, test: pd.DataFrame, srna_data: pd.DataFrame,
                               mrna_data: pd.DataFrame, srna_acc_col: str, mrna_acc_col: str,) -> pd.DataFrame:
    """
    :param model_h:
    :param train:
    :param test:
    :param srna_data: pd.DataFrame
    :param mrna_data: pd.DataFrame
    :param srna_acc_col: str - the column in train, test and srna_data containing unique id per sRNA
    :param mrna_acc_col: str - the column in train, test and mrna_data containing unique id per mRNA

    :return:
        test_predictions_df - pd.DataFrame including the following information:
                          sRNA accession id, mRNA accession id, interaction label (y_true),
                          model's prediction score (y_score OR y_graph_score), metadata columns.
    """
    # 1 - define model args
    model_args = model_h.get_model_args()
    kwargs = {}
    # 2 - train and test
    scores, predictions, training_history, train_val_data, shap_args = \
        model_h.train_and_test(model_args=model_args, unq_train=train, unq_test=test,
                               srna_data=srna_data, mrna_data=mrna_data, srna_acc_col=srna_acc_col,
                               mrna_acc_col=mrna_acc_col, **kwargs)
    test_predictions_df = predictions['out_test_pred']

    return test_predictions_df


def train_and_evaluate(model_h, train: Dict[str, object], test: Dict[str, object], **kwargs) -> \
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
    # 2 - train and test
    predictions, training_history = \
        model_h.train_and_test(X_train=train['X'], y_train=train['y'], X_test=test['X'],
                               y_test=test['y'], model_args=model_args,
                               metadata_train=train['metadata'], metadata_test=test['metadata'], **kwargs)
    test_predictions_df = predictions['out_test_pred']

    return test_predictions_df


if __name__ == "__main__":
    main()
