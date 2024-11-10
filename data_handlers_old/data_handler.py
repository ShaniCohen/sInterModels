import pandas as pd
from typing import Dict
from os.path import join
from utils_old.utils_general import read_df
import logging
logger = logging.getLogger(__name__)


def _load_interactions_data(data_path: str, srna_accession_id_col: str = "sRNA_EcoCyc_accession_id",
                            mrna_accession_id_col: str = "mRNA_EcoCyc_accession_id") -> (pd.DataFrame, pd.DataFrame):
    # train dataset for example
    train_example = read_df(join(data_path, "DATA_train_example.csv"))
    assert srna_accession_id_col in train_example.columns.values, f"{srna_accession_id_col} not in train df columns"
    assert mrna_accession_id_col in train_example.columns.values, f"{mrna_accession_id_col} not in train df columns"

    # test dataset for example
    test_example = read_df(join(data_path, "DATA_test_example.csv"))
    assert srna_accession_id_col in test_example.columns.values, f"{srna_accession_id_col} not in test df columns"
    assert mrna_accession_id_col in test_example.columns.values, f"{mrna_accession_id_col} not in test df columns"

    return train_example, test_example


def _load_rna_data(data_path: str, srna_accession_id_col: str = "EcoCyc_accession_id",
                   mrna_accession_id_col: str = "EcoCyc_accession_id"):
    """
    srna_eco: includes 94 unique sRNAs of Escherichia coli K12 MG1655 (NC_000913) from EcoCyc.
    mrna_eco: includes 4300 unique mRNAs of Escherichia coli K12 MG1655 (NC_000913) from EcoCyc.

    :return:
    srna_eco - pd.DataFrame
    srna_accession_id_col - str: the column in srna_eco containing unique id per sRNA
    mrna_eco - pd.DataFrame
    mrna_accession_id_col - str: the column in mrna_eco containing unique id per mRNA
    """
    # sRNA data from EcoCyc
    srna_eco = read_df(join(data_path, "DATA_srna_eco.csv"))
    assert srna_accession_id_col in srna_eco.columns.values, f"{srna_accession_id_col} not in df columns"

    # mRNA data from EcoCyc
    mrna_eco = read_df(join(data_path, "DATA_mrna_eco.csv"))
    assert mrna_accession_id_col in mrna_eco.columns.values, f"{mrna_accession_id_col} not in df columns"

    return srna_eco, mrna_eco, srna_accession_id_col, mrna_accession_id_col


class DataHandler(object):
    """
    Class to ...

    Parameters
    ----------

    Attributes
    ----------
    Same as parameters
    """
    def __init__(self, data_path: str):
        super(DataHandler, self).__init__()
        self.data_path = data_path
        # interactions data files
        self.train_fragments_file = "DATA_train_fragments.csv"
        self.test_complete_file = "DATA_test_complete.csv"
        self.test_filtered_file = "DATA_test_filtered.csv"
        # RNA data files
        self.srna_data_file = "DATA_srna_eco.csv"
        self.mrna_data_file = "DATA_mrna_eco.csv"
        self.srna_accession_id_col = "EcoCyc_accession_id"
        self.mrna_accession_id_col = "EcoCyc_accession_id"

        # interactions features and label columns
        self.features_cols = ['F_total_energy_dG', 'F_unfolding_energy_sRNA', 'F_unfolding_energy_mRNA',
                              'F_hybridization_energy', 'F_all_bp', 'F_GC_bp_prop', 'F_AU_bp_prop', 'F_GU_bp_prop',
                              'F_alignment_len', 'F_all_bp_prop', 'F_mismatches_prop', 'F_bulges_sRNA_prop',
                              'F_bulges_mRNA_prop', 'F_mismatches_count', 'F_bulges_sRNA_count', 'F_bulges_mRNA_count',
                              'F_max_consecutive_bp_prop', 'F_mRNA_flanking_region_A_prop',
                              'F_mRNA_flanking_region_U_prop', 'F_mRNA_flanking_region_G_prop',
                              'F_mRNA_flanking_region_C_prop', 'F_mRNA_flanking_region_A+U_prop',
                              'F_mRNA_flanking_region_G+C_prop']
        self.label_col = "interaction_label"

    def split_dataset(self, dataset: pd.DataFrame) -> Dict[str, object]:
        """
        :return:
        dict in the following format:
        {
            'X': pd.DataFrame,
            'y': List[int],
            'metadata': pd.DataFrame
        }
        """
        meta_cols = [c for c in dataset.columns.values if c not in self.features_cols + [self.label_col]]
        dataset = {
            'X': pd.DataFrame(dataset[self.features_cols]),
            'y': list(dataset.get(self.label_col)),
            'metadata': pd.DataFrame(dataset[meta_cols])
        }

        return dataset

    def load_interactions_datasets(self):
        """
        train_fragments: includes 14806 interactions: 7403 positives, 7403 synthetic negatives (see evaluation 1).
                         (synthetic samples are ignored when training GraphRNA)
                         RNA sequences are chimeric fragments
        test_complete: includes 391 interactions: 227 positives, 164 negatives (see evaluation 4).
        test_filtered: includes 342 interactions: 199 positives, 143 negatives (see evaluation 3).

        :return:
        train_fragments, test_complete, test_filtered - dicts in the following format:
        {
            'X': pd.DataFrame,
            'y': List[int],
            'metadata': pd.DataFrame
        }
        """
        # train (fragments, synthetic negatives)
        train_fragments = read_df(join(self.data_path, self.train_fragments_file))
        train_fragments = self.split_dataset(dataset=train_fragments)

        # test complete
        test_complete = read_df(join(self.data_path, self.test_complete_file))
        test_complete = self.split_dataset(dataset=test_complete)

        # test filtered
        test_filtered = read_df(join(self.data_path, self.test_filtered_file))
        test_filtered = self.split_dataset(dataset=test_filtered)

        return train_fragments, test_complete, test_filtered

    def load_rna_data(self):
        """
        srna_eco: includes 94 unique sRNAs of Escherichia coli K12 MG1655 (NC_000913) from EcoCyc.
        mrna_eco: includes 4300 unique mRNAs of Escherichia coli K12 MG1655 (NC_000913) from EcoCyc.

        :return:
        srna_eco - pd.DataFrame
        srna_eco_accession_id_col - str: the column in srna_eco containing unique id per sRNA
        mrna_eco - pd.DataFrame
        mrna_eco_accession_id_col - str: the column in mrna_eco containing unique id per mRNA
        """
        # sRNA data from EcoCyc
        srna_eco = read_df(join(self.data_path, self.srna_data_file))
        srna_eco_accession_id_col = self.srna_accession_id_col
        assert srna_eco_accession_id_col in srna_eco.columns.values, \
            f"{srna_eco_accession_id_col} not in {self.srna_data_file} columns"

        # mRNA data from EcoCyc
        mrna_eco = read_df(join(self.data_path, self.mrna_data_file))
        mrna_eco_accession_id_col = self.mrna_accession_id_col
        assert mrna_eco_accession_id_col in mrna_eco.columns.values, \
            f"{mrna_eco_accession_id_col} not in {self.mrna_data_file} columns"

        return srna_eco, mrna_eco, srna_eco_accession_id_col, mrna_eco_accession_id_col
