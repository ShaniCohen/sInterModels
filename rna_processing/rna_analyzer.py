import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
import logging
logger = logging.getLogger(__name__)


class RnaAnalyzer(object):
    # todo- update docstrings
    """
    Class for data analysis

    Parameters
    ----------

    Attributes
    ----------
    Same as parameters
    """

    def __init__(self, rna_annot: str = None, rna_name: str = None, rna_sequence: str = None,
                 genome_sequence=None):
        super(RnaAnalyzer, self).__init__()
        self.rna_annot = rna_annot
        self.rna_name = rna_name
        self.rna_sequence = rna_sequence
        self.genome_sequence = genome_sequence  # DNA sequence (5' to 3')
        self.rna_Rfam_family = None

    def extend_fragment_using_genome_sequence(self, fragment_start: int, fragment_end: int, fragment_strand: str,
                                              left_right_extension: int, interaction_id: int = None,
                                              frag_seq_for_validation: str = None) -> pd.DataFrame:
        """

        Parameters
        ----------
        fragment_start: the coordinate in self.genome_sequence where fragment starts
        fragment_end: the coordinate in self.genome_sequence where fragment ends
        fragment_strand: '-' OR '+'
        left_right_extension: the number of nt to extend on left and right sides
        interaction_id: optional
        frag_seq_for_validation: optional

        Returns
        -------
        DataFrame with the following columns:
            'interaction_id': interaction_id
            'new_frag_start': the extended fragment start position
            'new_frag_end': the extended fragment end position
            'new_frag_sequence': the extended fragment sequence
            'new_frag_len': the extended fragment sequence length
        """
        # print(interaction_id)
        # if interaction_id == 1:
        #     print()

        # logger.debug("extend_fragment_using_genome_sequence")
        assert fragment_strand in ['-', '+']
        left_extension = left_right_extension
        right_extension = left_right_extension

        new_frag_sta = max(1, fragment_start - left_extension)
        new_frag_end = min(len(self.genome_sequence), fragment_end + right_extension)
        new_frag_sequence = self.genome_sequence[new_frag_sta - 1: new_frag_end]
        new_frag_len = len(new_frag_sequence)

        assert new_frag_len <= left_extension + fragment_end - fragment_start + 1 + right_extension

        if fragment_strand == '-':
            new_frag_sequence = new_frag_sequence.reverse_complement()

        if frag_seq_for_validation is not None:
            assert frag_seq_for_validation in new_frag_sequence

        res = pd.DataFrame([{
            'interaction_id': interaction_id,
            'new_frag_start': new_frag_sta,
            'new_frag_end': new_frag_end,
            'new_frag_sequence': str(new_frag_sequence),
            'new_frag_len': new_frag_len
        }])
        return res

    def get_sequence_from_genome_by_Copra_logic(self, rna_start: int, rna_end: int, rna_strand: str,
                                                interaction_id: int = None) -> pd.DataFrame:
        """
        From the start_point of the RNA (mRNA), extend 200 upstream, 100 downstream
        if strand is +
            start_point is rna_start
            upstream is left  (200)
            downstream is right   (100)
        if strand is -
            start_point is rna_end
            upstream is right  (200)
            downstream is left   (100)

        Parameters
        ----------
        rna_start: the coordinate in self.genome_sequence where rna starts
        rna_end: the coordinate in self.genome_sequence where rna ends
        rna_strand: '-' OR '+'
        interaction_id: optional

        Returns
        -------
        DataFrame with the following columns:
            'interaction_id': interaction_id
            'seq_start': the start position of the RNA sequence
            'seq_end': the end position of the RNA sequence
            'seq': the RNA sequence
            'seq_len': the length position of the RNA sequence
        """
        # print(interaction_id)
        # if interaction_id in [208, 209, 210, 212]:
        #     print()
        upstream_ext = 200
        downstream_ext = 100
        assert rna_strand in ['-', '+']

        # strand +
        if rna_strand == '+':
            start_point = rna_start
            left_extension = upstream_ext
            right_extension = downstream_ext - 1   # -1 stands for counting the start point
        # strand -
        else:
            start_point = rna_end
            left_extension = downstream_ext - 1   # -1 stands for counting the start point
            right_extension = upstream_ext

        _seq_sta = max(1, start_point - left_extension)
        _seq_end = min(len(self.genome_sequence), start_point + right_extension)
        _seq = self.genome_sequence[_seq_sta - 1: _seq_end]
        _seq_len = len(_seq)

        assert _seq_len <= left_extension + right_extension + 1   # +1 stands for counting the start point

        if rna_strand == '-':
            _seq = _seq.reverse_complement()

        res = pd.DataFrame([{
            'interaction_id': interaction_id,
            'seq_start': _seq_sta,
            'seq_end': _seq_end,
            'seq': str(_seq),
            'seq_len': _seq_len
        }])
        return res

    # def extend_fragments(self, fragments, to_size, how="both", use_genome=False):
    #     sequence = self.genome_sequence if use_genome else self.rna_sequence
    #     res = {}
    #     for fragment in fragments:
    #         positions_to_add = max(0, to_size-len(fragment))
    #         if positions_to_add > 0:
    #             if len(re.findall(fragment, sequence)) == 1:
    #                 sta, end = re.search(fragment, sequence).span()
    #                 sta = max(0, sta - positions_to_add) if how == "left" else sta
    #                 end = min(len(sequence), end + positions_to_add) if how == "right" else end
    #                 if "both":
    #                     new_sta = max(0, sta - int(positions_to_add/2))
    #                     added_left = sta - new_sta
    #                     end = min(len(sequence), end + (positions_to_add - added_left))
    #                     sta = new_sta
    #                 res.update({fragment: sequence[sta:end]})
    #             elif len(re.findall(fragment, sequence)) > 1:
    #                 raise ValueError(f"fragment {fragment} found {len(re.findall(fragment, sequence))} times in sequence")
    #             else:
    #                 print(f"no exact match for fragment {fragment}")
    #                 res.update({fragment: None})
    #         else:
    #             res.update({fragment: fragment})
    #     return res

    def map_fragments_to_sequence(self, fragments_info, how="exact_match", map_to_complete_genome=False):
        """
        :param fragments_info: array of lists [[fragment(str), count(int), group(str)], ... ]
        :param how:
        :param map_to_complete_genome:
        :return:
        """
        sequence = self.genome_sequence if map_to_complete_genome else self.rna_sequence
        if how == "exact_match":
            matrix_rows = {}
            frags, counts, groups, no_match = [], [], [], []
            for i, lst in enumerate(fragments_info):
                vec = np.zeros(len(sequence))
                frag, count, group = lst[0], lst[1], lst[2]
                if len(re.findall(frag, sequence)) > 0:
                    sta, end = re.search(frag, sequence).span()
                    vec[sta:end] = 1
                    matrix_rows.update({i: vec})
                    frags = frags + [frag]
                    counts = counts + [count]
                    groups = groups + [group]
                    # todo - what if two mappings are possible?
                    # todo - implement mapping to strain genome (around the sRNA) instead of mapping directly to sRNA
                    if len(re.findall(frag, sequence)) > 1:
                        print(f"WARNING - found {len(re.findall(frag, sequence))} exact matches")
                else:
                    no_match = no_match + [lst]
            return pd.DataFrame.from_dict(matrix_rows, orient='index'), frags, counts, groups, no_match
        else:
            raise ValueError("only how='exact_match' is supported")

    def cluster_fragments_of_rna(self, fragments_info, n_regions=2, map_to_genome=False):
        """
        :param fragments_info: array of lists [[fragment(str), count(int), group(str)], ... ]
        :param n_regions:
        :param map_to_genome:
        :return:
        """
        # todo - mapping to the genome instead of mapping to the rna sequence
        # todo - implement "smart" match when mapping to rna_sequence - not only exact
        X, frags, counts, groups, no_match = self.map_fragments_to_sequence(fragments_info=fragments_info,
                                                                            map_to_complete_genome=map_to_genome)
        X['cluster'] = KMeans(n_clusters=n_regions, random_state=0).fit(X).labels_
        X['cluster'] = np.add(list(X['cluster']), np.ones((len(X)))).astype('int')
        X['group'], X['count'], X['fragment'], X['rna_name'] = groups, counts, frags, self.rna_name
        X = X.sort_values(by=['cluster', 'group', 'count'])
        meta_cols = ['rna_name', 'fragment', 'count', 'group', 'cluster']
        matrix_cols = [c for c in X.columns.values if c not in meta_cols]
        return X[matrix_cols], X[meta_cols], no_match

