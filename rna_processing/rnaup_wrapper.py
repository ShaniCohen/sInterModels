from typing import Tuple
import logging
import os
from os.path import join
from subprocess import PIPE, run
import time
logger = logging.getLogger(__name__)


class RNAupWrapper(object):
    """
    RNAup wrapper

    Parameters
    ----------

    Attributes
    ----------
    Same as parameters
    """
    def __init__(self, rnaup_dir_path: str, input_file: str, output_file: str):
        super(RNAupWrapper, self).__init__()
        self.rna_up_dir_path = rnaup_dir_path
        self.input_file = input_file
        self.output_file = output_file
        self.rna_up_default_interaction_window = 25

    @staticmethod
    def parse_output(out_to_parse: str) -> (str, Tuple[int, int], Tuple[int, int], float, float, float, float):
        outs = out_to_parse.split()
        dot_brackets = outs[0]
        position_longer_seq = (int(outs[1].split(',')[0]), int(outs[1].split(',')[1]))
        position_shorter_seq = (int(outs[3].split(',')[0]), int(outs[3].split(',')[1]))
        # most favorable interaction energy (dG)
        total_energy_dG = float(outs[4].replace("(", ""))
        # the free energy gained from forming the inter-molecular duplex (dGint)
        hybridization_energy = float(outs[6])
        # opening energy, accessibility - longer seq (dGu_l)
        unfolding_energy_longer_seq = float(outs[8])
        # opening energy, accessibility - shorter seq (dGu_s)
        unfolding_energy_shorter_seq = float(outs[10].replace(")", ""))
        return dot_brackets, position_longer_seq, position_shorter_seq, total_energy_dG, hybridization_energy, \
               unfolding_energy_longer_seq, unfolding_energy_shorter_seq

    def get_interaction_outputs(self, _id: int, sRNA_seq: str, mRNA_seq: str, job_num: int = 0,
                                dump_temp_output_file: bool = False, _debug: bool = False) -> dict:
        if _debug:
            logger.debug(f"interaction - {_id}")
        # if _id == 401:
        #     print()
        # 1- define longer and shorter sequences
        longer_rna = 'mRNA' if len(mRNA_seq) >= len(sRNA_seq) else 'sRNA'
        shorter_rna = 'sRNA' if longer_rna == 'mRNA' else 'mRNA'
        if _debug and (longer_rna == 'sRNA'):
            logger.warning(f"    in interaction {_id}, sRNA seq is longer")
        assert len(sRNA_seq) > 0 and len(mRNA_seq) > 0, "no sequence"

        # 2 - generate (override) input file
        # _input_file = self.input_file
        _input_file = f'_{job_num}_{_id}_{self.input_file}'
        with open(join(self.rna_up_dir_path, _input_file), 'w', encoding='utf8') as f:
            f.write(f">mRNA\n"
                    f"{mRNA_seq}\n"
                    f">sRNA\n"
                    f"{sRNA_seq}")
            f.close()
        # _output_file = self.output_file
        _output_file = f'_{job_num}_{_id}_{self.output_file}'

        # 3 - Run RNAup
        # 3.1 - define command
        out_command = '-o ' if dump_temp_output_file else ''
        command = f'./RNAup {out_command}-b < {_input_file} > {_output_file}'
        # 3.2 - run command
        retries = 3
        empty_res = {}
        for i in range(retries):
            stdout = run(args=command, shell=True, stdout=PIPE, cwd=self.rna_up_dir_path)
            stderr = run(args=command, shell=True, stderr=PIPE, cwd=self.rna_up_dir_path)

            # 4- log errors
            error_desc = stderr.stderr.decode()
            returned_error = bool(error_desc)
            empty_res = {
                'interaction_id': _id,
                'RNAup_interacting_sequence': None,
                'RNAup_dot_brackets': None,
                f'RNAup_position_in_{longer_rna}': None,
                f'RNAup_position_in_{shorter_rna}': None,
                'RNAup_total_energy_dG': None,
                f'RNAup_unfolding_energy_{longer_rna}': None,
                f'RNAup_unfolding_energy_{shorter_rna}': None,
                'RNAup_hybridization_energy': None,
                'RNAup_is_sRNA_longer': longer_rna == 'sRNA',
                'RNAup_returned_error': returned_error,
                'RNAup_error_desc': error_desc
            }

            # 5 - check if command completed
            if stdout.returncode == 0:
                # 4.1 - read data from output file
                with open(join(self.rna_up_dir_path, _output_file), 'r', encoding='utf8') as f:
                    content = f.readlines()
                if len(content) > 3:
                    # 4.2 - parse data
                    interacting_sequence = content[3].replace('\n', '')
                    dot_brackets, position_longer_seq, position_shorter_seq, total_energy_dG, hybridization_energy, \
                    unfolding_energy_longer_seq, unfolding_energy_shorter_seq = self.parse_output(content[2])
                    res = {
                        'interaction_id': _id,
                        'RNAup_interacting_sequence': interacting_sequence,
                        'RNAup_dot_brackets': dot_brackets,
                        f'RNAup_position_in_{longer_rna}': position_longer_seq,
                        f'RNAup_position_in_{shorter_rna}': position_shorter_seq,
                        'RNAup_total_energy_dG': total_energy_dG,
                        f'RNAup_unfolding_energy_{longer_rna}': unfolding_energy_longer_seq,
                        f'RNAup_unfolding_energy_{shorter_rna}': unfolding_energy_shorter_seq,
                        'RNAup_hybridization_energy': hybridization_energy,
                        'RNAup_is_sRNA_longer': longer_rna == 'sRNA',
                        'RNAup_returned_error': returned_error,
                        'RNAup_error_desc': error_desc
                    }
                    os.remove(join(self.rna_up_dir_path, _input_file))
                    os.remove(join(self.rna_up_dir_path, _output_file))
                    return res
                else:
                    logger.error(f"i = {i} -> no content for interaction id: {_id}, content: {content}")
                    time.sleep(10)
            else:
                logger.error(f"i = {i} - > error in command for interaction id: {_id}")
                time.sleep(10)

        os.remove(join(self.rna_up_dir_path, _input_file))
        os.remove(join(self.rna_up_dir_path, _output_file))
        return empty_res
