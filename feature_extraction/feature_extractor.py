import re
from os.path import join
from typing import Dict, List, Tuple
from utils_old.utils_general import read_df, order_df
from rna_processing.rnaup_wrapper import RNAupWrapper
from rna_processing.rna_analyzer import RnaAnalyzer
from rna_processing.rna_duplex_generator import RNADuplexGenerator
# from joblib import Parallel, delayed
import logging
from typing import Dict, List
import pandas as pd
from itertools import product
import numpy as np
logger = logging.getLogger(__name__)

KMERS_COMB = [''.join(p) for p in product(["A", "T", "G", "C"], repeat=3)]
ZERO_KMERS = dict(zip(KMERS_COMB, np.repeat(0, len(KMERS_COMB))))


def _get_kmers_freqs(rna_sequence: str, k: int = 3) -> Dict[str, float]:
    from skbio import Sequence
    kmers_freqs = ZERO_KMERS.copy()
    rna_freqs = Sequence(rna_sequence).kmer_frequencies(k, relative=True, overlap=True)
    kmers_freqs.update(rna_freqs)

    return kmers_freqs


# def _calc_rna_features(rna_sequences_lst: List[str]) -> pd.DataFrame:
#     rna_kmers_outs = list(map(_get_kmers_freqs, rna_sequences_lst))
#     rna_kmers = pd.DataFrame(rna_kmers_outs)[KMERS_COMB]
#     return rna_kmers


def _extract_3_mer_diff_features(df: pd.DataFrame, col_srna_seq: str, col_mrna_seq: str, feature_cols_prfx: str) \
        -> (pd.DataFrame, List[str]):
    """
    """
    logger.info("extracting 3-mer-diff features")
    # sRNA 3-mers
    logger.info("calc sRNA features")
    srna_kmers_outs = list(map(_get_kmers_freqs, df[col_srna_seq]))
    srna_kmers = pd.DataFrame(srna_kmers_outs)[KMERS_COMB]

    # mRNA 3-mers
    logger.info("calc mRNA features")
    mrna_kmers_outs = list(map(_get_kmers_freqs, df[col_mrna_seq]))
    mrna_kmers = pd.DataFrame(mrna_kmers_outs)[KMERS_COMB]

    # 3-mer diff
    logger.info("adding diff features")
    kmers_diff = mrna_kmers.subtract(srna_kmers)
    kmers_diff.columns = [f'{feature_cols_prfx}_{c}' for c in kmers_diff.columns.values]
    feat_cols = list(kmers_diff.columns.values)

    # concat
    df_w_feat = pd.concat([df, kmers_diff], axis=1)

    return df_w_feat, feat_cols


def _extract_local_inter_features(genome_record_v3, inter_data: pd.DataFrame, energy_feature_cols: List[str], col_inter_id: str,
                                  col_mrna_strand: str, feature_cols_prfx: str) -> (pd.DataFrame, List[str]):
    logger.debug("extracting local interaction features")
    # 1 - extend mRNA for context features computation
    context_left_right_extension = 20
    inter_data = _extend_mrna_sequence(_genome_record_v3=genome_record_v3, data=inter_data,
                                       col_inter_id=col_inter_id, col_mrna_strand=col_mrna_strand,
                                       _const_left_right_extension=context_left_right_extension)
    # 2 - modify the energy feature columns
    inter_data.columns = [c.replace('RNAup_', f'{feature_cols_prfx}_') if c in energy_feature_cols
                          else c for c in inter_data.columns.values]
    # 3 - compute local interaction features
    inter_data, feat_cols = _comp_local_interaction_features(df=inter_data, col_inter_id=col_inter_id,
                                                             feature_cols_prfx=feature_cols_prfx)
    return inter_data, feat_cols


def _run_rnaup_w_copra_logic(rnaup_dir_path: str, genome_record_v3, inter_data: pd.DataFrame, col_inter_id: str,
                             col_srna_start: str, col_srna_end: str, col_srna_seq: str,
                             col_mrna_start: str, col_mrna_end: str, col_mrna_strand: str) -> (pd.DataFrame, List[str]):
    logger.info(f'running RNAup with Copra logic for {len(inter_data)} samples...')

    # 1 ------- define fragments according to Copra logic
    ra = RnaAnalyzer(genome_sequence=genome_record_v3.seq)
    _id = inter_data[col_inter_id]
    # --- sRNA
    # define input sequence as the entire sRNA molecule
    inter_data['sRNA_seq_for_RNAup'] = inter_data[col_srna_seq]
    inter_data['sRNA_seq_for_RNAup_start'] = inter_data[col_srna_start].astype(int)
    inter_data['sRNA_seq_for_RNAup_end'] = inter_data[col_srna_end].astype(int)
    inter_data['sRNA_seq_for_RNAup_len'] = inter_data[col_srna_seq].apply(len)
    # --- mRNA
    # define input sequence according to CopraRNA logic
    # (200 nt upstream and +100 nt downstream with respect to the start codon)
    _mrna_start = inter_data[col_mrna_start].astype(int)
    _mrna_end = inter_data[col_mrna_end].astype(int)
    _mrna_strand = inter_data[col_mrna_strand]

    mrna_extended_seq_start, mrna_extended_seq_end, mrna_extended_seq_sequence, mrna_extended_seq_len = \
        _get_sequence_by_Copra_logic(rna_analyzer=ra, rna_start=_mrna_start, rna_end=_mrna_end,
                                     rna_strand=_mrna_strand, __id=_id)

    inter_data['mRNA_seq_for_RNAup'] = mrna_extended_seq_sequence
    inter_data['mRNA_seq_for_RNAup_start'] = mrna_extended_seq_start
    inter_data['mRNA_seq_for_RNAup_end'] = mrna_extended_seq_end
    inter_data['mRNA_seq_for_RNAup_len'] = mrna_extended_seq_len

    # 2 ------- run RNAup
    input_file = 'duplex.seq'
    output_file = 'output.out'
    rw = RNAupWrapper(rnaup_dir_path=rnaup_dir_path, input_file=input_file, output_file=output_file)

    logger.debug(f"Running RNAup - Sequential")
    rna_up_outs = []
    for _id, sRNA_seq, mRNA_seq in zip(inter_data[col_inter_id], inter_data['sRNA_seq_for_RNAup'],
                                       inter_data['mRNA_seq_for_RNAup']):
        out_record = rw.get_interaction_outputs(_id=_id, sRNA_seq=str(sRNA_seq), mRNA_seq=str(mRNA_seq))
        rna_up_outs.append(out_record)

    # 3 ------- merge and assert
    energy_feature_cols = ['RNAup_total_energy_dG', 'RNAup_unfolding_energy_mRNA', 'RNAup_unfolding_energy_sRNA',
                           'RNAup_hybridization_energy']
    rna_up_outs = pd.DataFrame(rna_up_outs)
    inter_data = pd.merge(inter_data, rna_up_outs, on=col_inter_id, how='left')
    assert len(inter_data) == len(rna_up_outs)

    return inter_data, energy_feature_cols


def extract_local_inter_and_3_mer_diff_features(rnaup_dir_path: str, genome_record_v3, inter_data: pd.DataFrame,
                                                col_srna_start: str = 'sRNA_EcoCyc_start',
                                                col_srna_end: str = 'sRNA_EcoCyc_end',
                                                col_srna_seq: str = 'sRNA_EcoCyc_sequence',
                                                col_mrna_start: str = 'mRNA_EcoCyc_start',
                                                col_mrna_end: str = 'mRNA_EcoCyc_end',
                                                col_mrna_seq: str = 'mRNA_EcoCyc_sequence',
                                                col_mrna_strand: str = 'mRNA_EcoCyc_strand') -> \
        (pd.DataFrame, List[str]):
    """

    Parameters
    ----------
    rnaup_dir_path
    genome_record_v3
    inter_data: dataset of unique interacting sRNA-mRNA pairs. df must have the columns specified below.
    col_srna_start
    col_srna_end
    col_srna_seq
    col_mrna_start
    col_mrna_end
    col_mrna_seq
    col_mrna_strand

    Returns
    -------
    pd.DataFrame: The dataset with additional columns of features and metadata
    List[str]: List of feature columns

    """
    # todo - provide genome_record_v3 in the outside func + provide this data in Zenodo
    feat_cols = []
    feature_cols_prfx = 'F'
    # 1 - validate data
    # 1.1 - assert no missing columns
    expected_cols = [col_srna_start, col_srna_end, col_srna_seq, col_mrna_start, col_mrna_end, col_mrna_strand]
    missing_cols = set(expected_cols) - set(inter_data.columns.values)
    assert len(missing_cols) == 0, f"inter_data is missing the following columns: {missing_cols}"
    # 1.2 - assert no nulls
    assert pd.isnull(inter_data[expected_cols]).sum().sum() == 0, \
        f"inter_data contains Nan values. " \
        f"please complete the values or remove samples with missing data in the following columns: {expected_cols}"

    # 2 - run RNAup to get local interaction duplex and energy features
    temp_id_col = "interaction_id"
    inter_data[temp_id_col] = np.arange(len(inter_data))
    inter_data, energy_feature_cols = \
        _run_rnaup_w_copra_logic(rnaup_dir_path=rnaup_dir_path, genome_record_v3=genome_record_v3,
                                 inter_data=inter_data, col_inter_id=temp_id_col, col_srna_start=col_srna_start,
                                 col_srna_end=col_srna_end, col_srna_seq=col_srna_seq, col_mrna_start=col_mrna_start,
                                 col_mrna_end=col_mrna_end, col_mrna_strand=col_mrna_strand)
    # 3 - extract local interaction features
    inter_data, li_feature_cols = \
        _extract_local_inter_features(genome_record_v3=genome_record_v3, inter_data=inter_data,
                                      energy_feature_cols=energy_feature_cols, col_inter_id=temp_id_col,
                                      col_mrna_strand=col_mrna_strand, feature_cols_prfx=feature_cols_prfx)
    feat_cols.append(li_feature_cols)
    # 4 - extract 3-mer-diff features
    inter_data, tmd_feature_cols = \
        _extract_3_mer_diff_features(df=inter_data, col_srna_seq=col_srna_seq, col_mrna_seq=col_mrna_seq,
                                     feature_cols_prfx=feature_cols_prfx)
    feat_cols.append(tmd_feature_cols)

    return inter_data, feat_cols


def _get_sequence_by_Copra_logic(rna_analyzer, rna_start: pd.Series, rna_end: pd.Series, rna_strand: pd.Series,
                                 __id: pd.Series) -> (pd.Series, pd.Series, pd.Series, pd.Series):
    '''

    Get sequence according to Copra logic: 200 upstream (5' end), 100 downstream (3' end)

    Parameters
    ----------
    rna_analyzer
    rna_start
    rna_end
    rna_strand
    __id

    Returns
    -------

    '''
    rna_info = list(map(rna_analyzer.get_sequence_from_genome_by_Copra_logic, rna_start, rna_end, rna_strand, __id))
    rna_info = pd.concat(rna_info, ignore_index=True).reset_index(drop=True)

    seq_start = rna_info['seq_start']
    seq_end = rna_info['seq_end']
    seq = rna_info['seq']
    seq_len = rna_info['seq_len']

    return seq_start, seq_end, seq, seq_len


def _get_extended_sequence(rna_analyzer, seq_start: pd.Series, seq_end: pd.Series, seq_strand: pd.Series,
                           left_right_extension, __id: pd.Series, sequence: pd.Series) \
        -> (pd.Series, pd.Series, pd.Series, pd.Series):
    '''
    This function should be used only for extending an EXISTING DNA sequence (with different start and end points)
    This SHOULD NOT bu used for getting DNA from genome according to Copra logic

    Parameters
    ----------
    rna_analyzer
    seq_start
    seq_end
    seq_strand
    left_right_extension
    __id
    sequence

    Returns
    -------

    '''
    new_seq_info = list(map(rna_analyzer.extend_fragment_using_genome_sequence, seq_start, seq_end, seq_strand,
                            left_right_extension, __id, sequence))
    new_seq_info = pd.concat(new_seq_info, ignore_index=True).reset_index(drop=True)

    new_seq_start = new_seq_info['new_frag_start']
    new_seq_end = new_seq_info['new_frag_end']
    new_seq_sequence = new_seq_info['new_frag_sequence']
    new_seq_len = new_seq_info['new_frag_len']

    return new_seq_start, new_seq_end, new_seq_sequence, new_seq_len


def _extend_mrna_sequence(_genome_record_v3, data: pd.DataFrame, col_inter_id: str, col_mrna_strand: str,
                          _const_left_right_extension: int) -> pd.DataFrame:
    ra = RnaAnalyzer(genome_sequence=_genome_record_v3.seq)
    _id = data[col_inter_id]
    # 1 - extend the sequence sent to RNAup
    _mrna_fragment_start = data['mRNA_seq_for_RNAup_start'].astype(int)
    _mrna_fragment_end = data['mRNA_seq_for_RNAup_end'].astype(int)
    _mrna_fragment_strand = data[col_mrna_strand]
    _mrna_fragment_sequence = data['mRNA_seq_for_RNAup']  # this is DNA (with T)
    _mrna_left_right_extension = np.repeat(_const_left_right_extension, len(data))

    mrna_extended_frag_start, mrna_extended_frag_end, mrna_extended_frag_sequence, mrna_extended_frag_len = \
        _get_extended_sequence(rna_analyzer=ra, seq_start=_mrna_fragment_start, seq_end=_mrna_fragment_end,
                               seq_strand=_mrna_fragment_strand, left_right_extension=_mrna_left_right_extension,
                               __id=_id, sequence=_mrna_fragment_sequence)
    # 2 - convert input sequences from DNA to RNA
    mrna_extended_frag_sequence = mrna_extended_frag_sequence.apply(lambda x: x.replace("T", "U"))
    _mrna_fragment_sequence = _mrna_fragment_sequence.apply(lambda x: x.replace("T", "U"))
    # 3 - get the left and right extensions
    extensions = list(map(lambda x, y: x.split(y), mrna_extended_frag_sequence, _mrna_fragment_sequence))
    data['ext_mRNA_seq_for_RNAup'] = mrna_extended_frag_sequence  # this is RNA (with U)
    data['left_ext_mRNA_seq_for_RNAup'] = list(map(lambda x: str(x[0]), extensions))
    data['right_ext_mRNA_seq_for_RNAup'] = list(map(lambda x: str(x[1]), extensions))

    return data

#
# # 5 - add RNAup features
# def run_rnaup_with_copra_logic(_genome_record_v3, _data: pd.DataFrame) -> pd.DataFrame:
#     logger.info(f'run RNAup with Copra logic')
#     # 1 ------- define fragments according to Copra logic
#     mrna_strand_col = "mRNA_EcoCyc_strand"
#     ra = RnaAnalyzer(genome_sequence=_genome_record_v3.seq)
#     _id = _data["interaction_id"]
#     # sRNA
#     # define fragment as the entire sRNA molecule
#     srna_extended_frag_start = _data['sRNA_EcoCyc_start'].astype(int)
#     srna_extended_frag_end = _data['sRNA_EcoCyc_end'].astype(int)
#     srna_extended_frag_sequence = _data['sRNA_EcoCyc_sequence']
#     srna_extended_frag_len = _data['sRNA_EcoCyc_sequence'].apply(len)
#     # mRNA
#     # define fragments according to Copra
#     _mrna_start = _data['mRNA_EcoCyc_start'].astype(int)
#     _mrna_end = _data['mRNA_EcoCyc_end'].astype(int)
#     _mrna_strand = _data[mrna_strand_col]
#
#     mrna_extended_frag_start, mrna_extended_frag_end, mrna_extended_frag_sequence, mrna_extended_frag_len = \
#         get_sequence_by_Copra_logic(rna_analyzer=ra, rna_start=_mrna_start, rna_end=_mrna_end,
#                                     rna_strand=_mrna_strand, __id=_id)
#
#     # sRNA
#     _data['sRNA_seq_for_RNAup'] = srna_extended_frag_sequence
#     _data['sRNA_seq_for_RNAup_start'] = srna_extended_frag_start
#     _data['sRNA_seq_for_RNAup_end'] = srna_extended_frag_end
#     _data['sRNA_seq_for_RNAup_len'] = srna_extended_frag_len
#     # mRNA
#     _data['mRNA_seq_for_RNAup'] = mrna_extended_frag_sequence
#     _data['mRNA_seq_for_RNAup_start'] = mrna_extended_frag_start
#     _data['mRNA_seq_for_RNAup_end'] = mrna_extended_frag_end
#     _data['mRNA_seq_for_RNAup_len'] = mrna_extended_frag_len
#     interaction_id_col = "interaction_id"
#
#     # 2 ------- run RNAup
#     logger.info(f'run RNAup for {len(_data)} interactions')
#     rna_up_dir_path = join('/', 'home', 'shanisa', 'ViennaRNA', 'bin')
#     input_file = 'duplex.seq'
#     output_file = 'output.out'
#     rw = RNAupWrapper(rnaup_dir_path=rna_up_dir_path, input_file=input_file, output_file=output_file)
#
#     logger.debug(f"Running RNAup NO Parallel")
#     rna_up_outs = []
#     for _id, sRNA_seq, mRNA_seq in zip(_data[interaction_id_col], _data['sRNA_seq_for_RNAup'],
#                                        _data['mRNA_seq_for_RNAup']):
#         out_record = rw.get_interaction_outputs(_id=_id, sRNA_seq=str(sRNA_seq), mRNA_seq=str(mRNA_seq))
#         rna_up_outs.append(out_record)
#
#     # n_jobs = 50
#     # logger.debug(f"Running RNAup with Parallel (n_jobs = {n_jobs})")
#     # # job_nums = np.tile(np.arange(n_jobs), len(_data))[:len(_data)]
#     # job_nums = np.arange(len(_data))
#     # rna_up_outs = Parallel(n_jobs=n_jobs, verbose=10, backend="threading")(
#     #     delayed(rw.get_interaction_outputs)(_id=_id, sRNA_seq=str(sRNA_seq), mRNA_seq=str(mRNA_seq), job_num=job_num)
#     #     for _id, sRNA_seq, mRNA_seq, job_num in
#     #     zip(_data[interaction_id_col], _data['sRNA_seq_for_RNAup'], _data['mRNA_seq_for_RNAup'], job_nums))
#
#     rna_up_outs = pd.DataFrame(rna_up_outs)
#     _data = pd.merge(_data, rna_up_outs, on=interaction_id_col, how='left')
#     assert len(_data) == len(rna_up_outs)
#
#     # 3 ------- extend mRNA inputs for context features
#     context_left_right_extension = 20
#     _data = extend_mrna_sequence(_genome_record_v3=_genome_record_v3, _data=_data, mrna_strand_col=mrna_strand_col,
#                                  _const_left_right_extension=context_left_right_extension)
#
#     return _data


def _comp_local_interaction_features(df: pd.DataFrame, col_inter_id: str, feature_cols_prfx: str,
                                     prev_feat: bool = False) -> (pd.DataFrame, List[str]):
    logger.debug('generate duplexes and compute local interaction features')
    # 1 - define duplex params
    _extended_mrna_input_seq = df['ext_mRNA_seq_for_RNAup'].apply(lambda x: x.replace("T", "U"))
    _mrna_input_seq = df['mRNA_seq_for_RNAup'].apply(lambda x: x.replace("T", "U"))
    for i in range(len(_mrna_input_seq)):
        assert _mrna_input_seq[i] in _extended_mrna_input_seq[i], "sequences mismatch"
    extensions = list(map(lambda x, y: x.split(y), _extended_mrna_input_seq, _mrna_input_seq))
    left_ext_mrna_seq_for_rnaup = list(map(lambda x: str(x[0]), extensions))
    right_ext_mrna_seq_for_rnaup = list(map(lambda x: str(x[1]), extensions))
    prev_feat = np.full(len(df), prev_feat)
    # 2 - generate duplex and calculate features
    duplexes = list(map(_get_duplex, df[col_inter_id], df['RNAup_returned_error'],
                        df['RNAup_is_sRNA_longer'], df['sRNA_seq_for_RNAup'],
                        df['mRNA_seq_for_RNAup'], df['RNAup_interacting_sequence'],
                        df['RNAup_dot_brackets'], df['RNAup_position_in_sRNA'],
                        df['RNAup_position_in_mRNA'], left_ext_mrna_seq_for_rnaup,
                        right_ext_mrna_seq_for_rnaup, prev_feat))
    # 3 - add features
    f_df = pd.DataFrame(list(map(lambda x: x.get_dup_features(get_id=True, features_prfx=feature_cols_prfx),
                                 [d for d in duplexes if pd.notnull(d)]))).rename(columns={"id": col_inter_id})
    _len = len(df)
    df = pd.merge(df, f_df, on=col_inter_id, how='left')
    assert len(df) == _len, "duplications post merge"
    feature_cols = [c for c in df.columns.values if re.match(f"{feature_cols_prfx}_", c)]
    # 4 - add new columns to _df
    df['duplex_str'] = list(map(lambda x: x.get_duplex_string() if pd.notnull(x) else None, duplexes))
    df['left_ext_mRNA_seq_for_RNAup'] = left_ext_mrna_seq_for_rnaup
    df['right_ext_mRNA_seq_for_RNAup'] = right_ext_mrna_seq_for_rnaup
    # 5 - remove unnecessary cols and order
    un_cols = ['sRNA_seq_for_RNAup_start', 'sRNA_seq_for_RNAup_end', 'sRNA_seq_for_RNAup_len',
               'mRNA_seq_for_RNAup_start', 'mRNA_seq_for_RNAup_end', 'mRNA_seq_for_RNAup_len',
               'RNAup_is_sRNA_longer', 'ext_mRNA_seq_for_RNAup', 'left_ext_mRNA_seq_for_RNAup',
               'right_ext_mRNA_seq_for_RNAup', 'mRNA_interacting_and_flanking_region', 'mRNA_left_flanking_region',
               'mRNA_right_flanking_region', 'duplex_str']
    # ['sRNA_seq_for_RNAup', 'mRNA_seq_for_RNAup', 'RNAup_interacting_sequence', 'RNAup_dot_brackets',
    # 'RNAup_position_in_mRNA', 'RNAup_position_in_sRNA', 'RNAup_returned_error', 'RNAup_error_desc']
    df = order_df(df=df[[c for c in df.columns.values if c not in un_cols]], last_cols=feature_cols)
    return df, feature_cols


def _split_str(string: str, first_out_idx: int, second_out_idx: int, split_by: str = '&') -> (str, str):
    outs = string.split(split_by)
    return outs[first_out_idx], outs[second_out_idx]


def _get_duplex(interaction_id: int, error_returned: bool, is_srna_longer: bool, srna_sequence: str, mrna_sequence: str,
                interacting_sequences: str, dot_brackets: str, srna_inter_pos: Tuple[int, int],
                mrna_inter_pos: Tuple[int, int], mrna_left_extension: str = "", mrna_right_extension: str = "",
                prev_feat: bool = False, _debug: bool = False):
    if _debug:
        logger.debug(f"interaction_id: {interaction_id}")
        if interaction_id == 594:
            print()
    if error_returned:
        return None
    if is_srna_longer:
        srna_num = '1'
        mrna_num = '2'
    else:
        srna_num = '2'
        mrna_num = '1'
    # 1 - define sRNA and mRNA interacting sequence
    srna_interacting_sequence, mrna_interacting_sequence = \
        _split_str(string=interacting_sequences, first_out_idx=int(srna_num) - 1, second_out_idx=int(mrna_num) - 1)
    # 2 - define sRNA and mRNA dot brackets
    srna_dot_brackets, mrna_dot_brackets = \
        _split_str(string=dot_brackets, first_out_idx=int(srna_num) - 1, second_out_idx=int(mrna_num) - 1)
    # 3 - convert input sequences from DNA to RNA
    srna_sequence = srna_sequence.replace("T", "U")
    mrna_sequence = mrna_sequence.replace("T", "U")
    # 3.1 - extensions for context features
    mrna_left_extension = mrna_left_extension.replace("T", "U")
    mrna_right_extension = mrna_right_extension.replace("T", "U")
    # 4 - define duplex args
    args = {
        f'rna{srna_num}_name': 'sRNA',
        f'rna{srna_num}_sequence': srna_sequence,
        f'rna{srna_num}_interacting_sequence': srna_interacting_sequence,
        f'rna{srna_num}_dot_brackets': srna_dot_brackets,
        f'rna{srna_num}_inter_pos': srna_inter_pos,
        f'rna{mrna_num}_name': 'mRNA',
        f'rna{mrna_num}_sequence': mrna_sequence,
        f'rna{mrna_num}_interacting_sequence': mrna_interacting_sequence,
        f'rna{mrna_num}_dot_brackets': mrna_dot_brackets,
        f'rna{mrna_num}_inter_pos': mrna_inter_pos,
        'mrna_left_extension': mrna_left_extension,
        'mrna_right_extension': mrna_right_extension,
        'duplex_id': interaction_id
    }

    dg = RNADuplexGenerator(**args)
    dg.generate_duplex(prev_feat=prev_feat)
    return dg

#
# class FeatureExtractor(object):
#     """
#     Parameters
#     ----------
#
#     Attributes
#     ----------
#     """
#     def __init__(self, df: pd.DataFrame, initial_feature_cols: List[str], initial_discrete_feature_cols: List[str],
#                  out_prfx: str, interaction_id_col: str, mrna_context_seq_col: str = 'mRNA_context_sequence'):
#         """
#
#         Parameters
#         ----------
#         df
#         initial_feature_cols
#         initial_discrete_feature_cols
#         out_prfx
#         """
#         super(FeatureExtractor, self).__init__()
#         self.df = df
#         # features
#         assert len(np.intersect1d(initial_feature_cols, df.columns.values)) == len(initial_feature_cols), \
#             f"some features are missing in df"
#         assert len(np.intersect1d(initial_feature_cols, initial_discrete_feature_cols)) == len(initial_discrete_feature_cols), \
#             f"discrete_features_cols should be a subset of all features_cols"
#
#         self.f_prfx = out_prfx  # 'F
#         # these won't be modified
#         self.other_cols = [c for c in df.columns.values if c not in initial_feature_cols]
#         self.interaction_id_col = interaction_id_col
#         self.mrna_context_seq_col = mrna_context_seq_col
#
#         # these will be modified
#         self.features_cols = initial_feature_cols
#         self.discrete_features_cols = initial_discrete_feature_cols
#
#     def _align_features_prefix(self):
#         logger.debug("  aligning features prefix")
#         # modify the actual column name in df
#         self.df.columns = [c.replace('RNAup_', f'{self.f_prfx}_') if c in self.features_cols
#                            else c for c in self.df.columns.values]
#         # modify the lists
#         self.features_cols = [c.replace('RNAup_', f'{self.f_prfx}_') for c in self.features_cols]
#         self.discrete_features_cols = [c.replace('RNAup_', f'{self.f_prfx}_') for c in self.discrete_features_cols]
#         return
#
#     # def _add_mrna_context_counts(self):
#     #     from rna_processing.rna_duplex_generator import RNADuplexGenerator
#     #     rdg = RNADuplexGenerator
#     #     # 1 - define col names
#     #     nm = {
#     #         'A_count': self.f_prfx + 'mRNA_context_A_count',
#     #         'U_count': self.f_prfx + 'mRNA_context_U_count',
#     #         'G_count': self.f_prfx + 'mRNA_context_G_count',
#     #         'C_count': self.f_prfx + 'mRNA_context_C_count',
#     #         'AU_count': self.f_prfx + 'mRNA_context_A+U_count',
#     #         'GC_count': self.f_prfx + 'mRNA_context_G+C_count'
#     #     }
#     #     # 2 - add counts
#     #     self.df[[nm['A_count'], nm['U_count'], nm['G_count'], nm['C_count']]] = \
#     #         pd.DataFrame(list(map(rdg.get_context_counts, self.df[self.mrna_context_seq_col])))
#     #     self.df[nm['AU_count']] = self.df[nm['A_count']] + self.df[nm['U_count']]
#     #     self.df[nm['GC_count']] = self.df[nm['G_count']] + self.df[nm['C_count']]
#     #     # 3 - update relevant lists
#     #     self.features_cols = self.features_cols + list(nm.values())
#     #     self.discrete_features_cols = self.discrete_features_cols + list(nm.values())
#     #
#     #     return
#
#     # def _add_mrna_context_features(self):
#     #     logger.debug("  adding mRNA context features")
#     #     self._add_mrna_context_counts()  # props already exist
#     #     return
#
#     def update_df_and_cols(self, new_df: pd.DataFrame):
#         _curr_cols = list(self.df.columns.values)
#         for c in new_df.columns.values:
#             # if new col
#             if c not in _curr_cols:
#                 # if feature col
#                 if re.match(f"^{self.f_prfx}_(.*?)", c):
#                     self.features_cols = self.features_cols + [c]
#                 else:
#                     self.other_cols = self.other_cols + [c]
#         self.df = new_df
#         return
#
#     def extract_features(self) -> (pd.DataFrame, List[str], List[str]):
#         """
#         Manually add, modify and remove features from self.df
#
#         Returns
#         -------
#
#         """
#         logger.debug("extracting features")
#         self._align_features_prefix()
#         _df, _discrete_features = _comp_local_interaction_features(df=self.df, col_inter_id=self.interaction_id_col,
#                                                                    feature_cols_prfx=self.f_prfx)
#         self.update_df_and_cols(new_df=_df)
#         assert sorted(self.features_cols + self.other_cols) == sorted(self.df.columns.values), \
#             "misalignment between lists and actual columns"
#         self.df = order_df(df=self.df, last_cols=self.features_cols)
#         logger.debug(f"extracted {len(self.features_cols)} features ({len(self.discrete_features_cols)} are discrete)")
#
#         return self.df, self.features_cols, self.discrete_features_cols
#





