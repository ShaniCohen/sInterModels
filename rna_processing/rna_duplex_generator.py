import pandas as pd
import numpy as np
from typing import Tuple, List
import re
import logging
logger = logging.getLogger(__name__)


class RNADuplexGenerator(object):
    """
    Class for generating an interaction duplex

    Parameters
    ----------

    Attributes
    ----------
    Same as parameters
    """

    line1 = "                     "
    line2 = "                     "
    line3 = "        "
    line4 = "mRNA                  "
    line5 = "                      "
    line6 = "sRNA                  "
    line7 = "        "
    line8 = "                     "
    line9 = "                     "
    header_len = len(line4)
    _5_prime_header = "5'-"
    _3_prime_header = "3'-"
    _5_or_3_prime_seq_len = len(line4) - len(line3) - len(_5_prime_header)
    _prime_edge_len = 3
    _prime_middle_dots_len = 3
    _prime_near_inter_len = 5
    assert _5_or_3_prime_seq_len == _prime_edge_len + _prime_middle_dots_len + _prime_near_inter_len

    line1_footer = ""
    line2_footer = "             "
    line3_footer = ""
    line4_footer = "              "
    line5_footer = "              "
    line6_footer = "              "
    line7_footer = ""
    line8_footer = "             "
    line9_footer = ""
    _5_prime_footer = "-5'"
    _3_prime_footer = "-3'"
    footer_len = len(line4_footer)

    dup_features = None
    discrete_dup_features_names = None
    default_features_prfx = 'FE'

    def __init__(self, rna1_name: str, rna2_name: str, rna1_sequence: str, rna2_sequence: str,
                 rna1_interacting_sequence: str, rna2_interacting_sequence: str, rna1_dot_brackets: str,
                 rna2_dot_brackets: str, rna1_inter_pos: Tuple[int, int], rna2_inter_pos: Tuple[int, int],
                 mrna_left_extension: str, mrna_right_extension: str, duplex_id: int = None):
        """

        Parameters
        ----------
        rna1_name: 'mRNA' or 'sRNA'
        rna2_name: 'mRNA' or 'sRNA'
        rna1_sequence: the RNA sequence of rna1 (U instead of T)
        rna2_sequence: the RNA sequence rna2 (U instead of T)
        rna1_interacting_sequence: the interacting sequence in 'rna1_sequence', e.g. 'GCUGUGA'
        rna2_interacting_sequence: the interacting sequence in 'rna2_sequence', e.g. 'UCGCGC'
        rna1_dot_brackets: the dot brackets representation of 'rna1_interacting_sequence', e.g. '((.(((('
        rna2_dot_brackets: the dot brackets representation of 'rna2_interacting_sequence', e.g. '))))))'
        rna1_inter_pos: the interaction position (start, end) in 'rna1_sequence', e.g. (4, 10)
        rna2_inter_pos: the interaction position (start, end) in 'rna2_sequence', e.g. (4, 9)
        mrna_left_extension: extension of mRNA sequence from the left side
        mrna_right_extension: extension of mRNA sequence from the right side
        duplex_id: optional, numeric i.d. for the duplex instance
        """

        super(RNADuplexGenerator, self).__init__()
        self.rna1_name = rna1_name
        self.rna2_name = rna2_name
        self.rna1_sequence = rna1_sequence
        self.rna2_sequence = rna2_sequence
        self.rna1_interacting_sequence = rna1_interacting_sequence
        self.rna2_interacting_sequence = rna2_interacting_sequence
        self.rna1_dot_brackets = rna1_dot_brackets
        self.rna2_dot_brackets = rna2_dot_brackets
        self.interacting_area_len_prev = max(len(self.rna1_interacting_sequence), len(self.rna2_interacting_sequence))
        self.alignment_len = None  # init in self.generate_duplex_figure

        self.rna1_inter_pos = rna1_inter_pos
        self.rna2_inter_pos = rna2_inter_pos
        self.mrna_left_extension = mrna_left_extension
        self.mrna_right_extension = mrna_right_extension
        self.dup_id = duplex_id

        # self.rna1_name = 'mRNA'
        # self.rna2_name = 'sRNA'
        # self.rna1_sequence = 'GGUGCUGUGAGCUGAAGCUGGAUGGUCUUGCCGUCAGUAUUCUGUAUGAAAAUGGCGUUUUAGUCAG'
        # self.rna2_sequence = 'UAUUCGCGCACCCCGGUCUAGCCGGGGUCAUUUUUUA'
        # self.rna1_interacting_sequence = 'GCUGUGA'
        # self.rna2_interacting_sequence = 'UCGCGC'
        # self.rna1_dot_brackets = '((.(((('
        # self.rna2_dot_brackets = '))))))'
        # self.interacting_area_len = max(len(self.rna1_interacting_sequence), len(self.rna2_interacting_sequence))
        # self.rna1_inter_pos = (4, 10)
        # self.rna2_inter_pos = (4, 9)

        assert self.rna1_sequence[self.rna1_inter_pos[0]-1:self.rna1_inter_pos[1]] == self.rna1_interacting_sequence
        assert self.rna2_sequence[self.rna2_inter_pos[0]-1:self.rna2_inter_pos[1]] == self.rna2_interacting_sequence
        assert len(self.rna1_interacting_sequence) == len(self.rna1_dot_brackets)
        assert len(self.rna2_interacting_sequence) == len(self.rna2_dot_brackets)

    @staticmethod
    def get_connection(s1: str, s2: str) -> str:
        return ":" if (s1 == 'G' and s2 == "U") or (s1 == 'U' and s2 == "G") else "|"

    def generate_interacting_area(self) -> (str, str, str, str, str):
        """

        Returns
        -------
        top line
        middle line
        bottom line
        """
        rna1_bulges = ""
        rna1_paired = ""
        connections = ""
        rna2_paired = ""
        rna2_bulges = ""
        # assume mRNA is larger
        i1 = 0
        i2 = -1
        for i in np.arange(self.interacting_area_len):
            if i1 < len(self.rna1_dot_brackets):
                b1 = self.rna1_dot_brackets[i1]
                s1 = self.rna1_interacting_sequence[i1]
            else:
                b1 = None
            if abs(i2) <= len(self.rna2_dot_brackets):
                b2 = self.rna2_dot_brackets[i2]
                s2 = self.rna2_interacting_sequence[i2]
            else:
                b2 = None
            if b1 == '(' and b2 == ')':
                # update rna 1
                rna1_bulges = rna1_bulges + " "
                rna1_paired = rna1_paired + s1
                i1 = i1 + 1
                # update rna 2
                rna2_bulges = rna2_bulges + " "
                rna2_paired = rna2_paired + s2
                i2 = i2 - 1
                # update connections
                connections = connections + self.get_connection(s1=s1, s2=s2)
            elif b1 == '.' and (b2 == ')' or b2 is None):
                # update rna 1
                rna1_bulges = rna1_bulges + s1
                rna1_paired = rna1_paired + " "
                i1 = i1 + 1
                # update rna 2 - blank
                rna2_bulges = rna2_bulges + " "
                rna2_paired = rna2_paired + " "
                # update connections - blank
                connections = connections + " "
            elif (b1 == '(' or b1 is None) and b2 == '.':
                # update rna 1 - blank
                rna1_bulges = rna1_bulges + " "
                rna1_paired = rna1_paired + " "
                # update rna 2
                rna2_bulges = rna2_bulges + s2
                rna2_paired = rna2_paired + " "
                i2 = i2 - 1
                # update connections - blank
                connections = connections + " "
            elif b1 == '.' and b2 == '.':
                # update rna 1
                rna1_bulges = rna1_bulges + s1
                rna1_paired = rna1_paired + " "
                i1 = i1 + 1
                # update rna 2
                rna2_bulges = rna2_bulges + s2
                rna2_paired = rna2_paired + " "
                i2 = i2 - 1
                # update connections - blank
                connections = connections + " "
            else:
                raise ValueError('invalid dot brackets structure')

        assert len(rna1_bulges) == len(rna1_paired) == len(connections) == len(rna2_paired) == len(rna2_bulges)

        return rna1_bulges, rna1_paired, connections, rna2_paired, rna2_bulges

    def create_alignment(self) -> (str, str, str, str, str, int):
        """

        Returns
        -------
        top line
        middle line
        bottom line
        """
        rna1_dot_bra = list(self.rna1_dot_brackets)
        rna1_intr_seq = list(self.rna1_interacting_sequence)
        rna2_dot_bra = list(self.rna2_dot_brackets)
        rna2_intr_seq = list(self.rna2_interacting_sequence)
        rna1_bulges = ""
        rna1_paired = ""
        connections = ""
        rna2_paired = ""
        rna2_bulges = ""
        # assume mRNA is larger (i.e. mRNA in rna1)
        b1, s1, b2, s2 = None, None, None, None
        while len(rna1_intr_seq) > 0 or len(rna2_intr_seq) > 0:
            if pd.isnull(b1):
                # get next nt from rna1
                if len(rna1_intr_seq) > 0:
                    b1 = rna1_dot_bra.pop(0)
                    s1 = rna1_intr_seq.pop(0)
                else:
                    b1 = 'end'
            if pd.isnull(b2):
                # get next nt from rna2
                if len(rna2_intr_seq) > 0:
                    b2 = rna2_dot_bra.pop(-1)
                    s2 = rna2_intr_seq.pop(-1)
                else:
                    b2 = 'end'
            if b1 == '(' and b2 == ')':
                # update rna 1
                rna1_bulges = rna1_bulges + " "
                rna1_paired = rna1_paired + s1
                # update rna 2
                rna2_bulges = rna2_bulges + " "
                rna2_paired = rna2_paired + s2
                # update connections
                connections = connections + self.get_connection(s1=s1, s2=s2)
                # both s1 and s2 were used
                b1, s1 = None, None
                b2, s2 = None, None
            elif b1 == '.' and (b2 == ')' or b2 == 'end'):
                # update rna 1
                rna1_bulges = rna1_bulges + s1
                rna1_paired = rna1_paired + " "
                # update rna 2 - blank
                rna2_bulges = rna2_bulges + " "
                rna2_paired = rna2_paired + " "
                # update connections - blank
                connections = connections + " "
                # only s1 was used
                b1, s1 = None, None
            elif (b1 == '(' or b1 == 'end') and b2 == '.':
                # update rna 1 - blank
                rna1_bulges = rna1_bulges + " "
                rna1_paired = rna1_paired + " "
                # update rna 2
                rna2_bulges = rna2_bulges + s2
                rna2_paired = rna2_paired + " "
                # update connections - blank
                connections = connections + " "
                # only s2 was used
                b2, s2 = None, None
            elif b1 == '.' and b2 == '.':
                # update rna 1
                rna1_bulges = rna1_bulges + s1
                rna1_paired = rna1_paired + " "
                # update rna 2
                rna2_bulges = rna2_bulges + s2
                rna2_paired = rna2_paired + " "
                # update connections - blank
                connections = connections + " "
                # both s1 and s2 were used
                b1, s1 = None, None
                b2, s2 = None, None

        assert len(rna1_bulges) == len(rna1_paired) == len(connections) == len(rna2_paired) == len(rna2_bulges)
        alignment_len = len(connections)

        return rna1_bulges, rna1_paired, connections, rna2_paired, rna2_bulges, alignment_len

    def adjust_by_rna_type(self, rna1_bulges: str, rna1_paired: str, connections: str, rna2_paired: str, rna2_bulges: str) \
            -> (str, str, str, str, str):
        if self.rna1_name == 'sRNA':
            # reverse sRNA to be from 3' to 5'
            srna_bulges = rna1_bulges[::-1]
            srna_paired = rna1_paired[::-1]
            # reverse mRNA to be from 5' to 3'
            mrna_paired = rna2_paired[::-1]
            mrna_bulges = rna2_bulges[::-1]
            # reverse connections
            connections = connections[::-1]
        else:
            mrna_bulges = rna1_bulges
            mrna_paired = rna1_paired
            srna_paired = rna2_paired
            srna_bulges = rna2_bulges

        return mrna_bulges, mrna_paired, connections, srna_paired, srna_bulges

    def update_lines_3_to_7(self, mrna_5_prime: str, mrna_3_prime: str, mrna_bulges: str, mrna_paired: str,
                            srna_5_prime: str, srna_3_prime: str, srna_bulges: str, srna_paired: str, connections: str):
        self.line3 = self.line3 + mrna_5_prime + mrna_bulges + mrna_3_prime
        self.line4 = self.line4 + mrna_paired + self.line4_footer
        self.line5 = self.line5 + connections + self.line5_footer
        self.line6 = self.line6 + srna_paired + self.line6_footer
        self.line7 = self.line7 + srna_3_prime + srna_bulges + srna_5_prime
        return

    @staticmethod
    def calc_features_iteratively(mrna_bulges: str, mrna_paired: str, connections: str, srna_paired: str,
                                  srna_bulges: str, prfx: str, prev_feat: bool) -> (dict, List[str], str):
        """

        Parameters
        ----------
        mrna_bulges
        mrna_paired
        connections
        srna_paired
        srna_bulges
        prfx
        prev_feat

        Returns
        -------
        a dictionary of calculated features
        a list of discrete features names
        a string representing the bulges and mismatches in the interacting area.
            m = bulge in the mRNA sequence
            s = bulge in the sRNA sequence
            x = mismatch

        """
        GC_pairs = 0
        AU_pairs = 0
        GU_pairs = 0
        mismatches = 0
        bulges_sRNA = 0
        bulges_mRNA = 0
        bulges_and_mismatch_str = ""
        for i in np.arange(len(connections)):
            # 1 - GC or AU pair
            conn_str = connections[i]
            if conn_str == '|':
                pair = f'{mrna_paired[i]}{srna_paired[i]}'
                if pair in ['GC', 'CG']:
                    GC_pairs += 1
                else:
                    AU_pairs += 1
                bulges_and_mismatch_str += " "
            # 2 - GU pair
            elif conn_str == ':':
                GU_pairs += 1
                bulges_and_mismatch_str += " "
            # 3 - mismatch or bulge
            else:
                # 3.1 - mRNA bulge (m)
                if srna_bulges[i] == " ":
                    bulges_mRNA += 1
                    bulges_and_mismatch_str += "m"
                # 3.2 - sRNA bulge (s)
                elif mrna_bulges[i] == " ":
                    bulges_sRNA += 1
                    bulges_and_mismatch_str += "s"
                # 3.3 - mismatch (x)
                else:
                    mismatches += 1
                    bulges_and_mismatch_str += "x"

        assert mismatches + bulges_sRNA + bulges_mRNA + GC_pairs + AU_pairs + GU_pairs == len(connections), \
            "all connections were counted"
        assert len(bulges_and_mismatch_str) == len(connections), "bulges_and_mismatch_str was calculated correctly"

        # nt sum
        all_bp = GC_pairs + AU_pairs + GU_pairs
        alignment_len = all_bp + mismatches + bulges_sRNA + bulges_mRNA  # = len(bulges_and_mismatch_str)

        iter_features = {
            f"{prfx}_all_bp": all_bp,
            f"{prfx}_GC_bp_prop": GC_pairs / all_bp,
            f"{prfx}_AU_bp_prop": AU_pairs / all_bp,
            f"{prfx}_GU_bp_prop": GU_pairs / all_bp,
            f"{prfx}_alignment_len": alignment_len,
            f"{prfx}_all_bp_prop": all_bp / alignment_len,
            f"{prfx}_mismatches_prop": mismatches / alignment_len,
            f"{prfx}_bulges_sRNA_prop": bulges_sRNA / alignment_len,
            f"{prfx}_bulges_mRNA_prop": bulges_mRNA / alignment_len,
        }
        if prev_feat:
            iter_features.update({
                f"{prfx}_GC_bp": GC_pairs,
                f"{prfx}_AU_bp": AU_pairs,
                f"{prfx}_GU_bp": GU_pairs,
                f"{prfx}_mismatches": mismatches,
                f"{prfx}_bulges_sRNA": bulges_sRNA,
                f"{prfx}_bulges_mRNA": bulges_mRNA,
            })
        iter_discrete_feature_names = [f"{prfx}_all_bp", f"{prfx}_alignment_len"]

        return iter_features, iter_discrete_feature_names, bulges_and_mismatch_str

    @staticmethod
    def get_context_counts(context_region: str):
        if pd.isnull(context_region):
            return None, None, None, None
        A_count = context_region.count("A")
        U_count = context_region.count("U")
        G_count = context_region.count("G")
        C_count = context_region.count("C")
        all_count = A_count + U_count + G_count + C_count
        assert all_count == len(context_region)

        return A_count, U_count, G_count, C_count

    @classmethod
    def get_context_props(cls, context_region: str):
        if pd.isnull(context_region):
            return None, None, None, None
        A_count, U_count, G_count, C_count = cls.get_context_counts(context_region=context_region)
        A_prop = A_count / len(context_region)
        U_prop = U_count / len(context_region)
        G_prop = G_count / len(context_region)
        C_prop = C_count / len(context_region)

        return A_prop, U_prop, G_prop, C_prop

    def calc_mrna_context_features(self) -> (dict, List[str]):
        prfx = self.default_features_prfx
        # 1 - get mRNA interaction positions and sequences
        if self.rna1_name == 'mRNA':
            mrna_seq = self.rna1_sequence
            mrna_inter_sta_idx = self.rna1_inter_pos[0] - 1
            mrna_inter_end_idx = self.rna1_inter_pos[1]
            mrna_inter_seq = self.rna1_interacting_sequence
        else:
            mrna_seq = self.rna2_sequence
            mrna_inter_sta_idx = self.rna2_inter_pos[0] - 1
            mrna_inter_end_idx = self.rna2_inter_pos[1]
            mrna_inter_seq = self.rna2_interacting_sequence
        # 2 - infer the extension needed for content measuring
        left_addition_size = len(self.mrna_left_extension)
        right_addition_size = len(self.mrna_right_extension)
        # 3 - extend the interaction seed
        # 3.1 - find the extended indexes on the mRNA sequence
        left_ind = mrna_inter_sta_idx - left_addition_size
        mid_sta = max(0, left_ind)
        right_ind = mrna_inter_end_idx + right_addition_size
        mid_end = min(len(mrna_seq), right_ind)
        # 3.2 - if additional left extension is needed
        if left_ind < 0:
            left_ext_sta_ind = max(0, len(self.mrna_left_extension) - abs(left_ind))
        else:
            left_ext_sta_ind = len(self.mrna_left_extension)
        # 3.3 - if additional right extension is needed
        if right_ind > len(mrna_seq):
            chars_to_add = right_ind - len(mrna_seq)
            right_ext_end_ind = min(len(self.mrna_right_extension), chars_to_add)
        else:
            right_ext_end_ind = 0
        # 4 - define the mRNA context region (interacting + flanking)
        _left = self.mrna_left_extension[left_ext_sta_ind:]
        _mid = mrna_seq[mid_sta:mid_end]
        _right = self.mrna_right_extension[:right_ext_end_ind]
        context_region = f'{_left}{_mid}{_right}'

        assert len(context_region) <= left_addition_size + len(mrna_inter_seq) + right_addition_size, \
            "length of context region is bounded by left and right extensions"

        # 5 - calculate features over the mRNA context area
        A_count, U_count, G_count, C_count = self.get_context_counts(context_region=context_region)
        all_count = A_count + U_count + G_count + C_count

        res = {
            "mRNA_context_sequence": context_region,
            f"{prfx}_mRNA_context_A_prop": A_count/all_count,
            f"{prfx}_mRNA_context_U_prop": U_count/all_count,
            f"{prfx}_mRNA_context_G_prop": G_count/all_count,
            f"{prfx}_mRNA_context_C_prop": C_count/all_count
        }
        _discrete_features = []

        return res, _discrete_features

    def calc_mrna_flanking_region_composition_features(self) -> (dict, List[str]):
        # print(f"duplex: {self.dup_id}")
        prfx = self.default_features_prfx
        # 1 - get mRNA interaction positions and sequences
        if self.rna1_name == 'mRNA':
            mrna_seq = self.rna1_sequence
            mrna_inter_sta_idx = self.rna1_inter_pos[0] - 1
            mrna_inter_end_idx = self.rna1_inter_pos[1]
            mrna_inter_seq = self.rna1_interacting_sequence
        else:
            mrna_seq = self.rna2_sequence
            mrna_inter_sta_idx = self.rna2_inter_pos[0] - 1
            mrna_inter_end_idx = self.rna2_inter_pos[1]
            mrna_inter_seq = self.rna2_interacting_sequence
        # 2 - infer the extension needed for content measuring
        left_addition_size = len(self.mrna_left_extension)
        right_addition_size = len(self.mrna_right_extension)
        # 3 - extend the interaction seed
        # 3.1 - find the extended indexes on the mRNA sequence
        left_ind = mrna_inter_sta_idx - left_addition_size
        mid_sta = max(0, left_ind)
        right_ind = mrna_inter_end_idx + right_addition_size
        mid_end = min(len(mrna_seq), right_ind)
        # 3.2 - if additional left extension is needed
        if left_ind < 0:
            left_ext_sta_ind = max(0, len(self.mrna_left_extension) - abs(left_ind))
        else:
            left_ext_sta_ind = len(self.mrna_left_extension)
        # 3.3 - if additional right extension is needed
        if right_ind > len(mrna_seq):
            chars_to_add = right_ind - len(mrna_seq)
            right_ext_end_ind = min(len(self.mrna_right_extension), chars_to_add)
        else:
            right_ext_end_ind = 0

        # 4 - define the mRNA relevant region (interacting + flanking)
        _left = self.mrna_left_extension[left_ext_sta_ind:]
        _mid = mrna_seq[mid_sta:mid_end]
        _right = self.mrna_right_extension[:right_ext_end_ind]
        rel_region = f'{_left}{_mid}{_right}'

        assert mrna_inter_seq in rel_region, "logical bug when computing rel_region"

        # 5 - define the mRNA flanking region
        _left_fr = self.mrna_left_extension[left_ext_sta_ind:] + mrna_seq[mid_sta:mrna_inter_sta_idx]
        _mid_inter = mrna_inter_seq
        _right_fr = mrna_seq[mrna_inter_end_idx:mid_end] + self.mrna_right_extension[:right_ext_end_ind]
        rel_region_fr = f'{_left_fr}{_mid_inter}{_right_fr}'
        flanking_region = f'{_left_fr}{_right_fr}'

        assert mrna_inter_seq in rel_region_fr, "logical bug when computing rel_region_fr"
        assert rel_region == rel_region_fr
        assert len(flanking_region) <= left_addition_size + right_addition_size, \
            "length of flanking region is bounded by left and right extensions"

        # 5 - calculate features over the mRNA flanking region
        A_prop, U_prop, G_prop, C_prop = self.get_context_props(context_region=flanking_region)

        res = {
            "mRNA_interacting_and_flanking_region": rel_region,
            "mRNA_left_flanking_region": _left_fr,
            "mRNA_right_flanking_region": _right_fr,
            f"{prfx}_mRNA_flanking_region_A_prop": A_prop,
            f"{prfx}_mRNA_flanking_region_U_prop": U_prop,
            f"{prfx}_mRNA_flanking_region_G_prop": G_prop,
            f"{prfx}_mRNA_flanking_region_C_prop": C_prop,
            f"{prfx}_mRNA_flanking_region_A+U_prop": A_prop + U_prop,
            f"{prfx}_mRNA_flanking_region_G+C_prop": G_prop + C_prop,
        }
        _discrete_features = []

        return res, _discrete_features

    def calc_general_features(self, prev_feat: bool) -> (dict, List[str]):
        prfx = self.default_features_prfx
        if self.rna1_name == 'mRNA':
            mrna_interacting_area_sequence = str(self.rna1_interacting_sequence)
            mrna_interacting_area_len = len(self.rna1_interacting_sequence)

            srna_interacting_area_sequence = str(self.rna2_interacting_sequence)
            srna_interacting_area_len = len(self.rna2_interacting_sequence)
        else:
            mrna_interacting_area_sequence = str(self.rna2_interacting_sequence)
            mrna_interacting_area_len = len(self.rna2_interacting_sequence)

            srna_interacting_area_sequence = str(self.rna1_interacting_sequence)
            srna_interacting_area_len = len(self.rna1_interacting_sequence)

        res = {}
        if prev_feat:
            res.update({
                "mRNA_interacting_area_sequence": mrna_interacting_area_sequence,
                f"{prfx}_mRNA_interacting_area_len": mrna_interacting_area_len,
                "sRNA_interacting_area_sequence": srna_interacting_area_sequence,
                f"{prfx}_sRNA_interacting_area_len": srna_interacting_area_len
            })
        _discrete_features = []
        return res, _discrete_features

    def calc_features(self, mrna_bulges: str, mrna_paired: str, connections: str, srna_paired: str, srna_bulges: str,
                      prev_feat: bool = False):
        prfx = self.default_features_prfx
        # 1 - iterative features (over interaction area)
        _features, _discrete_features_nms, bulges_and_mismatch_str = \
            self.calc_features_iteratively(mrna_bulges=mrna_bulges, mrna_paired=mrna_paired, connections=connections,
                                           srna_paired=srna_paired, srna_bulges=srna_bulges, prfx=prfx,
                                           prev_feat=prev_feat)
        # 2 - consecutive patterns
        _features.update({
            f"{prfx}_mismatches_count": sum(['x' in s for s in bulges_and_mismatch_str.split()]),
            f"{prfx}_bulges_sRNA_count": sum(['s' in s for s in bulges_and_mismatch_str.split()]),
            f"{prfx}_bulges_mRNA_count": sum(['m' in s for s in bulges_and_mismatch_str.split()]),
            f"{prfx}_max_consecutive_bp_prop":
                max(pd.Series(re.findall(r'[|:]+', connections)).apply(len), default=0) / len(bulges_and_mismatch_str),
        })
        _dis = [f"{prfx}_mismatches_count",  f"{prfx}_bulges_sRNA_count", f"{prfx}_bulges_mRNA_count"]
        _discrete_features_nms = _discrete_features_nms + _dis

        if prev_feat:
            _features.update({
                f"{prfx}_max_consecutive_mismatches":
                    max(pd.Series(re.findall(r'[x]+', bulges_and_mismatch_str)).apply(len), default=0),
                f"{prfx}_max_consecutive_bulges_sRNA":
                    max(pd.Series(re.findall(r'[s]+', bulges_and_mismatch_str)).apply(len), default=0),
                f"{prfx}_max_consecutive_bulges_mRNA":
                    max(pd.Series(re.findall(r'[m]+', bulges_and_mismatch_str)).apply(len), default=0),
                f"{prfx}_max_consecutive_bp_including_GU":
                    max(pd.Series(re.findall(r'[|:]+', connections)).apply(len), default=0),
                f"{prfx}_max_consecutive_bp_excluding_GU":
                    max(pd.Series(re.findall(r'[|]+', connections)).apply(len), default=0)
            })

        # 4 - general features
        _general_features, _dis = self.calc_general_features(prev_feat=prev_feat)
        _features.update(_general_features)
        _discrete_features_nms = _discrete_features_nms + _dis

        # 5 - mRNA flanking region composition features
        _mrna_frc_features, _dis = self.calc_mrna_flanking_region_composition_features()
        _features.update(_mrna_frc_features)
        _discrete_features_nms = _discrete_features_nms + _dis

        #  mRNA "context" features (prev)
        if prev_feat:
            _mrna_context_features, _dis = self.calc_mrna_context_features()
            _features.update(_mrna_context_features)

        self.dup_features = _features
        self.discrete_dup_features_names = _discrete_features_nms
        return

    def adjust_prime_sequence(self, prime_seq: str, left_len: int, right_len: int) -> str:
        if len(prime_seq) > self._5_or_3_prime_seq_len:
            # replace middle with dots
            left = prime_seq[0:left_len]
            middle = "..."
            right = prime_seq[len(prime_seq)-right_len:len(prime_seq)]

            prime_seq = left + middle + right
            assert len(prime_seq) == self._5_or_3_prime_seq_len
        return prime_seq

    def add_prime_header(self, seq: str, header: str) -> str:
        seq = header + seq
        seq = " " * (self._5_or_3_prime_seq_len + len(self._5_prime_header) - len(seq)) + seq
        return seq

    def add_prime_footer(self, seq: str, footer: str) -> str:
        seq = seq + footer
        seq = seq + " " * (self._5_or_3_prime_seq_len + len(self._5_prime_footer) - len(seq))
        return seq

    def get_5_and_3_prime_sequences(self, rna_seq: str, end_idx_5_prime: int, start_idx_3_prime: int,
                                    reverse_outputs: bool = False) -> (str, str):
        # 1 - get the complete prime sequences
        seq_5_prime = rna_seq[0:end_idx_5_prime]
        seq_3_prime = rna_seq[start_idx_3_prime - 1:]
        # 2 - adjust large prime sequences
        seq_5_prime = self.adjust_prime_sequence(prime_seq=seq_5_prime, left_len=self._prime_edge_len,
                                                 right_len=self._prime_near_inter_len)
        seq_3_prime = self.adjust_prime_sequence(prime_seq=seq_3_prime, left_len=self._prime_near_inter_len,
                                                 right_len=self._prime_edge_len)
        # 3 - add headers and footers
        if reverse_outputs:
            seq_3_prime = self.add_prime_header(seq=seq_3_prime[::-1], header=self._3_prime_header)
            seq_5_prime = self.add_prime_footer(seq=seq_5_prime[::-1], footer=self._5_prime_footer)
        else:
            seq_5_prime = self.add_prime_header(seq=seq_5_prime, header=self._5_prime_header)
            seq_3_prime = self.add_prime_footer(seq=seq_3_prime, footer=self._3_prime_footer)
        assert len(seq_5_prime) == len(seq_3_prime)

        return seq_5_prime, seq_3_prime

    def generate_non_interacting_area(self) -> (str, str, str, str):
        if self.rna1_name == 'sRNA':
            srna_seq = self.rna1_sequence
            srna_inter_pos = self.rna1_inter_pos
            mrna_seq = self.rna2_sequence
            mrna_inter_pos = self.rna2_inter_pos
        else:
            srna_seq = self.rna2_sequence
            srna_inter_pos = self.rna2_inter_pos
            mrna_seq = self.rna1_sequence
            mrna_inter_pos = self.rna1_inter_pos

        mrna_5_prime, mrna_3_prime = self.get_5_and_3_prime_sequences(rna_seq=mrna_seq,
                                                                      end_idx_5_prime=mrna_inter_pos[0] - 1,
                                                                      start_idx_3_prime=mrna_inter_pos[1] + 1)
        srna_5_prime, srna_3_prime = self.get_5_and_3_prime_sequences(rna_seq=srna_seq,
                                                                      end_idx_5_prime=srna_inter_pos[0] - 1,
                                                                      start_idx_3_prime=srna_inter_pos[1] + 1,
                                                                      reverse_outputs=True)
        return mrna_5_prime, mrna_3_prime, srna_5_prime, srna_3_prime

    def set_interaction_connections_and_calc_features(self):
        # 1 - generate interaction area sequences
        rna1_bulges, rna1_paired, connections, rna2_paired, rna2_bulges = self.generate_interacting_area()
        # 2 - adjust sequences according to their RNA type (mRNA vs sRNA)
        mrna_bulges, mrna_paired, connections, srna_paired, srna_bulges = \
            self.adjust_by_rna_type(rna1_bulges=rna1_bulges, rna1_paired=rna1_paired, connections=connections,
                                    rna2_paired=rna2_paired, rna2_bulges=rna2_bulges)
        # 3 - generate non-interacting area components (consider RNA type)
        mrna_5_prime, mrna_3_prime, srna_5_prime, srna_3_prime = self.generate_non_interacting_area()
        self.update_lines_3_to_7(mrna_5_prime, mrna_3_prime, mrna_bulges, mrna_paired,
                                 srna_5_prime, srna_3_prime, srna_bulges, srna_paired, connections)
        # 4 - calculate features

        return

    def set_coordinates(self):
        """ define lines 1,2 and 8,9 """
        if self.rna1_name == 'sRNA':
            srna_inter_pos = self.rna1_inter_pos
            srna_seq_len = len(self.rna1_sequence)
            mrna_inter_pos = self.rna2_inter_pos
            mrna_seq_len = len(self.rna2_sequence)
        else:
            mrna_inter_pos = self.rna1_inter_pos
            mrna_seq_len = len(self.rna1_sequence)
            srna_inter_pos = self.rna2_inter_pos
            srna_seq_len = len(self.rna2_sequence)

        mrna_left = str(max(mrna_inter_pos[0] - 1, 1))
        mrna_right = str(min(mrna_inter_pos[1] + 1, mrna_seq_len))
        srna_left = str(min(srna_inter_pos[1] + 1, srna_seq_len))
        srna_right = str(max(srna_inter_pos[0] - 1, 1))

        middle1 = " "*(self.interacting_area_len - (len(mrna_left)-1))
        bottom1 = " "*(self.footer_len - len(mrna_right))
        middle9 = " "*(self.interacting_area_len - (len(srna_left)-1))
        bottom9 = " "*(self.footer_len - len(srna_right))

        self.line1 = self.line1 + mrna_left + middle1 + mrna_right + bottom1
        self.line2 = self.line2 + "|" + " "*self.interacting_area_len + "|" + self.line2_footer
        self.line8 = self.line8 + "|" + " "*self.interacting_area_len + "|" + self.line8_footer
        self.line9 = self.line9 + srna_left + middle9 + srna_right + bottom9
        return

    def update_lines_1_2_8_9(self):
        """ define lines 1,2 and 8,9 """
        if self.rna1_name == 'sRNA':
            srna_inter_pos = self.rna1_inter_pos
            srna_seq_len = len(self.rna1_sequence)
            mrna_inter_pos = self.rna2_inter_pos
            mrna_seq_len = len(self.rna2_sequence)
        else:
            mrna_inter_pos = self.rna1_inter_pos
            mrna_seq_len = len(self.rna1_sequence)
            srna_inter_pos = self.rna2_inter_pos
            srna_seq_len = len(self.rna2_sequence)

        mrna_left = str(max(mrna_inter_pos[0] - 1, 1))
        mrna_right = str(min(mrna_inter_pos[1] + 1, mrna_seq_len))
        srna_left = str(min(srna_inter_pos[1] + 1, srna_seq_len))
        srna_right = str(max(srna_inter_pos[0] - 1, 1))

        middle1 = " "*(self.alignment_len - (len(mrna_left)-1))
        bottom1 = " "*(self.footer_len - len(mrna_right))
        middle9 = " "*(self.alignment_len - (len(srna_left)-1))
        bottom9 = " "*(self.footer_len - len(srna_right))

        self.line1 = self.line1 + mrna_left + middle1 + mrna_right + bottom1
        self.line2 = self.line2 + "|" + " "*self.alignment_len + "|" + self.line2_footer
        self.line8 = self.line8 + "|" + " "*self.alignment_len + "|" + self.line8_footer
        self.line9 = self.line9 + srna_left + middle9 + srna_right + bottom9
        return

    def get_duplex_string(self) -> str:
        return f"{self.line1}\n{self.line2}\n{self.line3}\n{self.line4}\n{self.line5}\n{self.line6}\n{self.line7}" \
               f"\n{self.line8}\n{self.line9}"

    def get_dup_features(self, get_id: bool = False, features_prfx: str = 'FE') -> dict:
        res = {"id": self.dup_id} if get_id else {}
        if self.dup_features is None:
            logger.warning("Make sure to run generate_duplex() first, then get_features()")
        _features = {k.replace(f'{self.default_features_prfx}_', f'{features_prfx}_'): v
                     for k, v in self.dup_features.items()}
        res.update(_features)

        return res
        # return self.dup_features

    def get_discrete_features_names(self,  features_prfx: str = 'FE') -> List[str]:
        if self.discrete_dup_features_names is None:
            logger.warning("Make sure to run generate_duplex() first, then get_discrete_dup_features()")
        _dis_features_nms = [k.replace(f'{self.default_features_prfx}_', f'{features_prfx}_') for k in
                             self.discrete_dup_features_names]
        return _dis_features_nms

    # def get_num_of_base_pairs(self) -> int:
    #     return self.rna1_dot_brackets.count("(")
    #
    # def get_interacting_area_len(self) -> int:
    #     return self.interacting_area_len

    def generate_duplex_figure(self) -> (str, str, str, str, str):
        # 1 - create alignment
        rna1_bulges, rna1_paired, connections, rna2_paired, rna2_bulges, alignment_len = self.create_alignment()
        self.alignment_len = alignment_len
        # 2 - adjust alignment sequences according to their RNA type (mRNA vs sRNA)
        mrna_bulges, mrna_paired, connections, srna_paired, srna_bulges = \
            self.adjust_by_rna_type(rna1_bulges=rna1_bulges, rna1_paired=rna1_paired, connections=connections,
                                    rna2_paired=rna2_paired, rna2_bulges=rna2_bulges)
        self.update_lines_1_2_8_9()
        # 3 - generate non-alignment sequences (consider RNA type)
        mrna_5_prime, mrna_3_prime, srna_5_prime, srna_3_prime = self.generate_non_interacting_area()
        self.update_lines_3_to_7(mrna_5_prime, mrna_3_prime, mrna_bulges, mrna_paired,
                                 srna_5_prime, srna_3_prime, srna_bulges, srna_paired, connections)

        return mrna_bulges, mrna_paired, connections, srna_paired, srna_bulges

    # def generate_duplex_prev(self, prev_feat: bool = False):
    #     """
    #     """
    #     # 1 - update lines 1,2 and 8,9
    #     self.set_coordinates()
    #     # 2 - update lines 3-7 + calculate features
    #     self.set_interaction_connections_and_calc_features()
    #     return

    def generate_duplex(self, prev_feat: bool = False):
        """
        """
        mrna_bulges, mrna_paired, connections, srna_paired, srna_bulges = self.generate_duplex_figure()
        self.calc_features(mrna_bulges, mrna_paired, connections, srna_paired, srna_bulges, prev_feat=prev_feat)

        return
