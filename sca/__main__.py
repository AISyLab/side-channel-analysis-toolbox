import click
from sca.util.file_loader import FileLoader
from sca.attack.mia import Mia
from sca.attack.pia import Pia
from sca.analysis.nicv import NICV
from sca.analysis.pearson import Pearson
from sca.attack.sa import SA
from sca.attack.ta import TA
from sca.attack.cpa import CPA
from sca.attack.dpa import DPA
from sca.analysis.sostd import SOSTD
from sca.util.data_partitioner import DataPartitioner
from sca.analysis.tvla import TVLA
from sca.analysis.snr import SNR
from sca.attack.lra import LRA
from sca.tracealign.aligner import Aligner
import numpy as np

CONST_CPA_ONLINE_DEFAULT_NUM_TRACES = 100
""" The default number of traces to use for cpa attack"""

CONST_DEFAULT_TRACES_FILE = "data/traces.npy"
"""The default traces file"""

CONST_DEFAULT_KEYS_FILE = "data/key.npy"
"""The default keys file"""

CONST_DEFAULT_PLAIN_FILE = "data/plain.npy"
"""The default plain file"""

CONST_MIA_DEFAULT_NUM_ATTACK_TRACES = 1000
"""The default number of attack traces"""

CONST_DPA_DEFAULT_OFFSET = 15
""""The default offset for dpa"""

CONST_DPA_DEFAULT_BIT = 4
"""The default bit coefficient for dpa"""

CONST_MIA_DEFAULT_NUM_TRACES = 1000
"""The default number of attack traces"""

CONST_CPA_DEFAULT_NUM_ATTACK_TRACES = 3000
"""The default number of attack traces"""

CONST_CPA_DEFAULT_NUM_TRACES = 3000
"""The default number of attack traces"""

CONST_MIA_DEFAULT_SUBKEY = 0
"""The default subkey to figure out."""

CONST_CPA_DEFAULT_SUBKEY = 0
"""The default subkey to figure out."""

CONST_PEARSON_DEFAULT_POINTS_OF_INTEREST = 5
"""The default number of points of interest to take."""

CONST_SA_DEFAULT_NUM_ATTACK_TRACES = 30
"""The default number of attack traces"""

CONST_LRA_DEFAULT_NUM_TRACES = 3000
"""The default number of attack traces for lra attack"""

CONST_LRA_NUM_FEATURES = 15
""" The default number of features to use for lra attack"""

CONST_SA_NUM_FEATURES = 15
""" The default number of traces to use for stochastic attack"""

CONST_MIA_NUM_FEATURES = 10
""" The default number of traces to use for mutual information attack"""

CONST_CPA_NUM_FEATURES = 10
""" The default number of traces to use for cpa attack"""

CONST_SA_DEFAULT_NUM_TRACES = 4000
"""The default number of traces for stochastic attack"""

CONST_SA_DEFAULT_ROUND = 1
"""The default round to attack on"""

CONST_TA_DEFAULT_NUM_TRACES_PROFILING = 8000
"""The default number of profiling traces for template attack"""

CONST_TA_DEFAULT_NUM_TRACES_ATTACK = 30
"""The default number of attack traces for template attack"""

CONST_TA_DEFAULT_SPACING = 1
"""The default spacing between points of interest."""

CONST_TA_DEFAULT_SUBKEY = 16
"""The default subkey to figure out. 16 means the whole key, so all subkeys."""

CONST_TA_DEFAULT_POINTS_OF_INTEREST = 5
"""The default number of points of interest to take."""


@click.command()
@click.option('--traces-file', '-tf', type=str,
              default=CONST_DEFAULT_TRACES_FILE,
              help='The path for the traces to be used')
@click.option('--keys-file', '-kf', type=str,
              default=CONST_DEFAULT_KEYS_FILE,
              help='The path for the keys to be used')
@click.option('--plain-file', '-pf', type=str,
              default=CONST_DEFAULT_PLAIN_FILE,
              help='The path for the plain to be used')
@click.option('--leakage-model', '-lm', is_flag=True,
              default=True,
              help='The leakage model used, the default is the hamming weights model, enable the flag to switch to the'
                   'intermediate values model.')
def nicv(traces_file, keys_file, plain_file, leakage_model):
    """
    Performs Normalized Inter-Class Variance for feature selection.

    :param traces_file: the traces file to use
    :param keys_file: the keys file to use
    :param plain_file: the plaintexts file to use
    :param leakage_model: the leakage model to use.
    """
    traces, keys, plain = FileLoader.main(traces_file, keys_file, plain_file)
    NICV.run(traces, keys, plain, leakage_model)


@click.command()
@click.option('--traces-file', '-tf', type=str,
              default=CONST_DEFAULT_TRACES_FILE,
              help='The path for the traces to be used')
@click.option('--keys-file', '-kf', type=str,
              default=CONST_DEFAULT_KEYS_FILE,
              help='The path for the keys to be used')
@click.option('--plain-file', '-pf', type=str,
              default=CONST_DEFAULT_PLAIN_FILE,
              help='The path for the plain to be used')
@click.option('--num-features', '-f', type=int,
              default=CONST_PEARSON_DEFAULT_POINTS_OF_INTEREST, show_default=True,
              help='The number of points of interest.')
@click.option('--round', '-r', 'round_', type=int,
              default=CONST_SA_DEFAULT_ROUND, show_default=True,
              help='The round to attack on.')
@click.option('-o', 'operation', type=int,
              help='Which operation to attack. This might override or be overridden by --op-* flags. '
                   '0 is SubBytes, 1 is ShiftRows, 2 is MixColumns, 3 is AddRoundKey.')
@click.option('--op-substitution', 'operation', flag_value=0, default=True,
              help='Attack the SubBytes operation.')
@click.option('--op-shift-rows', 'operation', flag_value=1,
              help='Attack the ShiftRows operation.')
@click.option('--op-mix-columns', 'operation', flag_value=2,
              help='Attack the MixColumns operation.')
@click.option('--op-add-round-key', 'operation', flag_value=3,
              help='Attack the AddRoundKey operation.')
@click.option('--save-traces', is_flag=True,
              help='Save the interesting traces instead of their indices')
@click.option('--subkey', type=int,
              default=0, show_default=True,
              help='The subkey to find; must be in the range [0-15].')
@click.option('--leakage-model', '-lm', is_flag=True,
              default=True,
              help='The leakage model used, the default is the hamming weights model, enable the flag to switch to the'
                   'intermediate values model.')
def pearson(traces_file: str, keys_file: str, plain_file: str, num_features: int, round_: int,
            operation: int, save_traces: bool, subkey: int, leakage_model: bool):
    """
    Performs Pearson correlation for feature selection. By default, the indices
    of the specified number of points of interest (features) are saved to
    `out/pearson_correlation_selected_indices`. When `--save-traces` is passed,
    the corresponding traces are saved instead.

    :param traces_file: the traces file to use.
    :param keys_file: the keys file to use.
    :param plain_file: the plaintexts file to use.
    :param num_features: the number of points of interest to extract.
    :param round_: the AES round to attack.
    :param operation: the AES operation to 'attack', represented as an integer from 0 to 3.
    :param save_traces: whether to save the traces to a file.
    :param subkey: the subkey index to analyze. Must be in the range [0-15].
    :param leakage_model: the leakage model to use.
    """
    traces, keys, plain = FileLoader.main(traces_file, keys_file, plain_file)
    Pearson.run(traces, keys, plain, num_features, save_traces, subkey, round_, operation, leakage_model)


@click.command()
@click.option('--traces-file', '-tf', type=str,
              default=CONST_DEFAULT_TRACES_FILE,
              help='The path for the traces to be used')
@click.option('--keys-file', '-kf', type=str,
              default=CONST_DEFAULT_KEYS_FILE,
              help='The path for the keys to be used')
@click.option('--plain-file', '-pf', type=str,
              default=CONST_DEFAULT_PLAIN_FILE,
              help='The path for the plain to be used')
@click.option('--num-features', '-f', type=int,
              default=CONST_PEARSON_DEFAULT_POINTS_OF_INTEREST, show_default=True,
              help='The number of points of interest.')
@click.option('--round', '-r', 'round_', type=int,
              default=CONST_SA_DEFAULT_ROUND, show_default=True,
              help='The round to attack on.')
@click.option('-o', 'operation', type=int,
              help='Which operation to attack. This might override or be overridden by --op-* flags. '
                   '0 is SubBytes, 1 is ShiftRows, 2 is MixColumns, 3 is AddRoundKey.')
@click.option('--op-substitution', 'operation', flag_value=0, default=True,
              help='Attack the SubBytes operation.')
@click.option('--op-shift-rows', 'operation', flag_value=1,
              help='Attack the ShiftRows operation.')
@click.option('--op-mix-columns', 'operation', flag_value=2,
              help='Attack the MixColumns operation.')
@click.option('--op-add-round-key', 'operation', flag_value=3,
              help='Attack the AddRoundKey operation.')
@click.option('--save-traces', is_flag=True,
              help='Save the interesting traces instead of their indices')
@click.option('--subkey', type=int,
              default=0, show_default=True,
              help='The subkey to find; must be in the range [0-15].')
@click.option('--leakage-model', '-lm', is_flag=True,
              default=True,
              help='The leakage model used, the default is the hamming weights model, enable the flag to switch to the'
                   'intermediate values model.')
def sost(traces_file: str, keys_file: str, plain_file: str, num_features: int, round_: int,
         operation: int, save_traces: bool, subkey: int, leakage_model: bool):
    """
    Performs SOST. By default, the indices of the specified number of points of
    interest (features) are saved to `out/sost_selected_indices`. When
    `--save-traces` is passed, the corresponding traces are saved instead.

    :param traces_file: the traces file to use.
    :param keys_file: the keys file to use.
    :param plain_file: the plaintexts file to use.
    :param num_features: the number of points of interest to extract.
    :param round_: the AES round to attack.
    :param operation: the AES operation to 'attack', represented as an integer from 0 to 3.
    :param save_traces: whether to save the traces to a file.
    :param subkey: the subkey index to analyze. Must be in the range [0-15].
    :param leakage_model: the leakage model to use.
    """
    traces, keys, plain = FileLoader.main(traces_file, keys_file, plain_file)
    SOSTD.run(traces, keys, plain, num_features, save_traces, subkey, round_, operation, leakage_model, True)


@click.command()
@click.option('--traces-file', '-tf', type=str,
              default=CONST_DEFAULT_TRACES_FILE,
              help='The path for the traces to be used')
@click.option('--keys-file', '-kf', type=str,
              default=CONST_DEFAULT_KEYS_FILE,
              help='The path for the keys to be used')
@click.option('--plain-file', '-pf', type=str,
              default=CONST_DEFAULT_PLAIN_FILE,
              help='The path for the plain to be used')
@click.option('--num-features', '-f', type=int,
              default=CONST_PEARSON_DEFAULT_POINTS_OF_INTEREST, show_default=True,
              help='The number of points of interest.')
@click.option('--round', '-r', 'round_', type=int,
              default=CONST_SA_DEFAULT_ROUND, show_default=True,
              help='The round to attack on.')
@click.option('-o', 'operation', type=int,
              help='Which operation to attack. This might override or be overridden by --op-* flags. '
                   '0 is SubBytes, 1 is ShiftRows, 2 is MixColumns, 3 is AddRoundKey.')
@click.option('--op-substitution', 'operation', flag_value=0, default=True,
              help='Attack the SubBytes operation.')
@click.option('--op-shift-rows', 'operation', flag_value=1,
              help='Attack the ShiftRows operation.')
@click.option('--op-mix-columns', 'operation', flag_value=2,
              help='Attack the MixColumns operation.')
@click.option('--op-add-round-key', 'operation', flag_value=3,
              help='Attack the AddRoundKey operation.')
@click.option('--save-traces', is_flag=True,
              help='Save the interesting traces instead of their indices')
@click.option('--subkey', type=int,
              default=0, show_default=True,
              help='The subkey to find; must be in the range [0-15].')
@click.option('--leakage-model', '-lm', is_flag=True,
              default=True,
              help='The leakage model used, the default is the hamming weights model, enable the flag to switch to the'
                   'intermediate values model.')
def sosd(traces_file: str, keys_file: str, plain_file: str, num_features: int, round_: int,
         operation: int, save_traces: bool, subkey: int, leakage_model: bool):
    """
    Performs SOSD. By default, the indices of the specified number of points of
    interest (features) are saved to `out/sosd_selected_indices`. When
    `--save-traces` is passed, the corresponding traces are saved instead.

    :param traces_file: the traces file to use.
    :param keys_file: the keys file to use.
    :param plain_file: the plaintexts file to use.
    :param num_features: the number of points of interest to extract.
    :param round_: the AES round to attack.
    :param operation: the AES operation to 'attack', represented as an integer from 0 to 3.
    :param save_traces: whether to save the traces to a file.
    :param subkey: the subkey index to analyze. Must be in the range [0-15].
    :param leakage_model: the leakage model to use.
    """
    traces, keys, plain = FileLoader.main(traces_file, keys_file, plain_file)
    SOSTD.run(traces, keys, plain, num_features, save_traces, subkey, round_, operation, leakage_model, False)


@click.command()
@click.option('--traces-file', '-tf', type=str,
              default=CONST_DEFAULT_TRACES_FILE,
              help='The path for the traces to be used')
@click.option('--keys-file', '-kf', type=str,
              default=CONST_DEFAULT_KEYS_FILE,
              help='The path for the keys to be used')
@click.option('--plain-file', '-pf', type=str,
              default=CONST_DEFAULT_PLAIN_FILE,
              help='The path for the plain to be used')
@click.option('--subkey', type=int,
              default=CONST_MIA_DEFAULT_SUBKEY,
              help='The subkey to find; must be in the range [0-15]. '
                   'When leaving this option out, the whole key is calculated.')
@click.option('--traces', '-t', 'num_traces', type=int,
              default=CONST_MIA_DEFAULT_NUM_TRACES, show_default=True,
              help='The number of traces to run the attack with.')
@click.option('--num_attack-traces', '-a', type=int,
              default=CONST_MIA_DEFAULT_NUM_ATTACK_TRACES, show_default=True)
@click.option('--round', '-r', 'round_', type=int,
              default=CONST_SA_DEFAULT_ROUND, show_default=True,
              help='The round to attack on.')
@click.option('--num-features', '-f', type=int,
              default=CONST_MIA_NUM_FEATURES, show_default=True)
@click.option('-o', 'operation', type=int,
              help='Which operation to attack. This might override or be overridden by --op-* flags. '
                   '0 is SubBytes, 1 is ShiftRows, 2 is MixColumns, 3 is AddRoundKey.')
@click.option('--op-substitution', 'operation', flag_value=0, default=True,
              help='Attack the SubBytes operation.')
@click.option('--op-shift-rows', 'operation', flag_value=1,
              help='Attack the ShiftRows operation.')
@click.option('--op-mix-columns', 'operation', flag_value=2,
              help='Attack the MixColumns operation.')
@click.option('--op-add-round-key', 'operation', flag_value=3,
              help='Attack the AddRoundKey operation.')
@click.option('--fs-none', 'feature_select', flag_value=0,
              help='Don\'t use feature selection')
@click.option('--fs-pearson', 'feature_select', flag_value=1,
              help='Use pearson for feature selection')
@click.option('--fs-sost', 'feature_select', flag_value=2, default=True,
              help='Use SOST for feature selection')
@click.option('--fs-nicv', 'feature_select', flag_value=3,
              help='Use Nicv for feature selection')
@click.option('--fs-sosd', 'feature_select', flag_value=4,
              help='Use SOSD for feature selection')
@click.option('--leakage-model', '-lm', is_flag=True,
              default=True,
              help='The leakage model used, the default is the hamming weights model, enable the flag to switch to the'
                   'intermediate values model.')
@click.option('--debug-mode', '-d', is_flag=True, default=False,
              help='Enables debug mode, a mode in which more detailed information about the execution of the attack is'
                   'printed and logged')
def mia(traces_file: str, keys_file: str, plain_file: str, subkey: int, leakage_model: bool,
        num_traces: int, num_attack_traces: int, num_features: int, feature_select: int, round_: int,
        operation: int, debug_mode: bool):
    """Performs MIA (Mutual Information Analysis) attack.

    :param traces_file: the traces file to use
    :param keys_file: the keys file to use
    :param plain_file: the plaintexts file to use
    :param subkey: the subkey index to analyze. Must be in the range [0-15]
    :param leakage_model: the leakage model to use.
    :param round_: the AES round to attack.
    :param num_traces: the amount of traces to analyze starting from trace 0.
    :param num_attack_traces: the amount of attack traces to analyze starting from trace 0.
    :param num_features: the number of features
    :param operation: the AES operation to 'attack', represented as an integer from 0 to 3.
    :param feature_select: which feature select to use
    :param debug_mode: whether to enable debug mode
    """
    traces, keys, plain = FileLoader.main(traces_file, keys_file, plain_file)
    profiling_traces, profiling_keys, profiling_plaintext, attack_traces, attack_keys, attack_plaintext = \
        DataPartitioner.get_traces(traces, keys, plain, num_traces, subkey,
                                   0, num_attack_traces, num_features, round_, operation, leakage_model)
    Mia.run(profiling_traces, profiling_keys, profiling_plaintext, attack_traces, attack_keys, attack_plaintext,
            round_, operation, subkey, feature_select,
            num_features, leakage_model, debug_mode_enabled=debug_mode)


@click.command()
@click.option('--traces-file', '-tf', type=str,
              default=CONST_DEFAULT_TRACES_FILE,
              help='The path for the traces to be used')
@click.option('--keys-file', '-kf', type=str,
              default=CONST_DEFAULT_KEYS_FILE,
              help='The path for the keys to be used')
@click.option('--plain-file', '-pf', type=str,
              default=CONST_DEFAULT_PLAIN_FILE,
              help='The path for the plain to be used')
@click.option('--round', '-r', 'round_', type=int,
              default=CONST_SA_DEFAULT_ROUND, show_default=True,
              help='The round to attack on.')
@click.option('-o', 'operation', type=int,
              help='Which operation to attack. This might override or be overridden by --op-* flags. '
                   '0 is SubBytes, 1 is ShiftRows, 2 is MixColumns, 3 is AddRoundKey.')
@click.option('--op-substitution', 'operation', flag_value=0, default=True,
              help='Attack the SubBytes operation.')
@click.option('--op-shift-rows', 'operation', flag_value=1,
              help='Attack the ShiftRows operation.')
@click.option('--op-mix-columns', 'operation', flag_value=2,
              help='Attack the MixColumns operation.')
@click.option('--op-add-round-key', 'operation', flag_value=3,
              help='Attack the AddRoundKey operation.')
@click.option('--traces', '-t', 'num_traces', type=int,
              default=CONST_SA_DEFAULT_NUM_TRACES, show_default=True,
              help='The number of traces to run the attack with.')
@click.option('--num_attack-traces', '-a', type=int,
              default=CONST_SA_DEFAULT_NUM_ATTACK_TRACES, show_default=True)
@click.option('--num-features', '-f', type=int,
              default=CONST_SA_NUM_FEATURES, show_default=True)
@click.option('--subkey', type=int,
              default=CONST_TA_DEFAULT_SUBKEY,
              help='The subkey to find; must be in the range [0-16], 16 signaling the whole key to be found out.')
@click.option('--gpu', is_flag=True, help='Enable GPU acceleration')
@click.option('--fs-none', 'feature_select', flag_value=0,
              help='Don\'t use feature selection')
@click.option('--fs-pearson', 'feature_select', flag_value=1, default=True,
              help='Use pearson for feature selection')
@click.option('--fs-sost', 'feature_select', flag_value=2,
              help='Use SOST for feature selection')
@click.option('--fs-nicv', 'feature_select', flag_value=3,
              help='Use Nicv for feature selection')
@click.option('--fs-sosd', 'feature_select', flag_value=4,
              help='Use SOSD for feature selection')
@click.option('--leakage-model', '-lm', is_flag=True,
              default=False,
              help='The leakage model used, the default is the intermediate values model, enable the flag to switch to '
                   'the hamming weights model.')
@click.option('--debug-mode', '-d', is_flag=True, default=False,
              help='Enables debug mode, a mode in which more detailed information about the execution of the attack is'
                   'printed and logged')
def sa(traces_file: str, keys_file: str, plain_file: str, round_: int, operation: int, num_traces: int,
       num_attack_traces: int, subkey: int, gpu: bool, leakage_model: bool, num_features: int, feature_select: int,
       debug_mode: bool):
    """
    Performs stochastic attack.

    :param traces_file: the traces file to use.
    :param keys_file: the keys file to use.
    :param plain_file: the plaintexts file to use.
    :param round_: the AES round to attack.
    :param operation: the AES operation to 'attack', represented as an integer from 0 to 3.
    :param num_traces: the amount of traces to analyze starting from trace 0.
    :param num_attack_traces: the amount of attack traces to analyze starting from trace 0.
    :param subkey: the subkey index to analyze. Must be in the range [0-16], 16 signaling the whole key to be found out.
    :param gpu: enables gpu acceleration if set to true.
    :param num_features: number of features to select for.
    :param feature_select: which feature select to use, default is pearson.
    :param leakage_model: the leakage model to use.
    :param debug_mode: whether to enable debug mode
    """

    traces, keys, plain = FileLoader.main(traces_file, keys_file, plain_file)
    profiling_traces, profiling_keys, profiling_plaintext, attack_traces, attack_keys, attack_plaintext = \
        DataPartitioner.get_traces(traces, keys, plain, num_traces, subkey,
                                   0, num_attack_traces, num_features, round_, operation, leakage_model)
    SA.run(profiling_traces, profiling_keys, profiling_plaintext, attack_traces, attack_keys, attack_plaintext,
           round_, operation, len(profiling_traces), num_attack_traces, subkey, feature_select, num_features, gpu,
           leakage_model, debug_mode_enabled=debug_mode)


@click.command()
@click.option('--traces-file', '-tf', type=str,
              default=CONST_DEFAULT_TRACES_FILE,
              help='The path for the traces to be used')
@click.option('--keys-file', '-kf', type=str,
              default=CONST_DEFAULT_KEYS_FILE,
              help='The path for the keys to be used')
@click.option('--plain-file', '-pf', type=str,
              default=CONST_DEFAULT_PLAIN_FILE,
              help='The path for the plain to be used')
@click.option('--points-of-interest', '-i', type=int,
              default=CONST_TA_DEFAULT_POINTS_OF_INTEREST, show_default=True,
              help='The number of points of interest.')
@click.option('--pooled', '-p', is_flag=True,
              help='Perform pooled template attack instead of normal template attack.')
@click.option('--spacing', '-s', type=int,
              default=CONST_TA_DEFAULT_SPACING, show_default=True,
              help='The spacing between the points of interest.')
@click.option('--subkey', type=int,
              default=CONST_TA_DEFAULT_SUBKEY,
              help='The subkey to find; must be in the range [0-16], 16 signaling the whole key to be found out.'
                   'When leaving this option out, the whole key is calculated.')
@click.option('--traces', '-t', 'num_traces', type=int,
              default=CONST_TA_DEFAULT_NUM_TRACES_PROFILING, show_default=True,
              help='Number of traces the template will be build with.')
@click.option('--num_attack-traces', '-a', type=int,
              default=CONST_TA_DEFAULT_NUM_TRACES_ATTACK, show_default=True,
              help='Number of traces the attack will be performed with.')
@click.option('--gpu', is_flag=True, help='Enable GPU acceleration')
@click.option('--fs-none', 'feature_select', flag_value=0, default=True,
              help='Don\'t use feature selection')
@click.option('--fs-pearson', 'feature_select', flag_value=1,
              help='Use pearson for feature selection')
@click.option('--fs-sost', 'feature_select', flag_value=2,
              help='Use SOST for feature selection')
@click.option('--fs-nicv', 'feature_select', flag_value=3,
              help='Use Nicv for feature selection')
@click.option('--fs-sosd', 'feature_select', flag_value=4,
              help='Use SOSD for feature selection')
@click.option('--leakage-model', '-lm', is_flag=True,
              default=True,
              help='The leakage model used, the default is the hamming weights model, enable the flag to switch to the'
                   'intermediate values model.')
@click.option('--debug-mode', '-d', is_flag=True,
              default=False,
              help='Enables debug mode, a mode in which more detailed information about the execution of the attack is'
                   'printed and logged')
def ta(traces_file: str, keys_file: str, plain_file: str, points_of_interest: int, pooled: bool, spacing: int,
       subkey: int, num_traces: int, gpu: bool, leakage_model: bool, debug_mode: bool, num_attack_traces: int,
       feature_select: int):
    """
    Performs (pooled) template attack.

    :param traces_file: the traces file to use.
    :param keys_file: the keys file to use.
    :param plain_file: the plaintexts file to use.
    :param points_of_interest: the number of points of interest to extract.
    :param pooled: whether to use pooled attack.
    :param spacing: spacing between the points of interest.
    :param subkey: the subkey index to analyze. Must be in the range [0-16], 16 signaling the whole key to be found out.
    :param num_traces: the amount of traces to analyze starting from trace 0
    :param gpu: whether or not to use gpu for this attack
    :param leakage_model: the leakage model to use
    :param debug_mode: whether to enable debug mode
    :param num_attack_traces: the number of attack traces.
    :param feature_select: which feature select to use, default is none.
    """
    traces, keys, plain = FileLoader.main(traces_file, keys_file, plain_file)

    profiling_traces, profiling_keys, profiling_plaintext, attack_traces, attack_keys, attack_plaintext = \
        DataPartitioner.get_traces(traces, keys, plain, num_traces, subkey,
                                   0, num_attack_traces, 0, 1, 0, leakage_model)

    if not pooled and profiling_traces.shape[0] < profiling_traces.shape[1]:
        print("ERROR: profiling traces are smaller than features, please run Pooled Template Attack instead of"
              " normal Template attack for a more accurate result.")
        exit(1)

    TA.run(profiling_traces, profiling_keys, profiling_plaintext, attack_traces, attack_keys, attack_plaintext,
           pooled, points_of_interest, spacing, subkey, gpu, leakage_model, debug_mode, feature_select)


@click.command()
@click.option('--traces-file', '-tf', type=str,
              default=CONST_DEFAULT_TRACES_FILE,
              help='The path for the traces to be used')
@click.option('--output-file', '-of', type=str,
              default=None,
              help='The name of the desired output file. By default it does not save the processed traces.')
@click.option('--threshold', '-t', 'threshold', type=int,
              default=4.5,
              help='The threshold used in the t-test to determine if a feature leaks')
def tvla(traces_file, output_file, threshold) -> None:
    """
    Performs TVLA
    """
    traces = FileLoader.load_traces(traces_file)
    TVLA.run(traces, output_file=output_file, threshold=threshold)


@click.command()
@click.option('--traces-file', '-tf', type=str,
              default=CONST_DEFAULT_TRACES_FILE,
              help='The path for the traces to be used')
@click.option('--keys-file', '-kf', type=str,
              default=CONST_DEFAULT_KEYS_FILE,
              help='The path for the keys to be used')
@click.option('--plain-file', '-pf', type=str,
              default=CONST_DEFAULT_PLAIN_FILE,
              help='The path for the plain to be used')
@click.option('--subkey', type=int,
              default=CONST_CPA_DEFAULT_SUBKEY,
              help='The subkey to find; must be in the range [0-15]. '
                   'When leaving this option out, the whole key is calculated.')
@click.option('--traces', '-t', 'num_traces', type=int,
              default=CONST_CPA_DEFAULT_NUM_TRACES, show_default=True,
              help='The number of traces to run the attack with.')
@click.option('--num_attack-traces', '-a', type=int,
              default=CONST_CPA_DEFAULT_NUM_ATTACK_TRACES, show_default=True)
@click.option('--round', '-r', 'round_', type=int,
              default=CONST_SA_DEFAULT_ROUND, show_default=True,
              help='The round to attack on.')
@click.option('--num-features', '-f', type=int,
              default=CONST_CPA_NUM_FEATURES, show_default=True)
@click.option('-o', 'operation', type=int,
              help='Which operation to attack. This might override or be overridden by --op-* flags. '
                   '0 is SubBytes, 1 is ShiftRows, 2 is MixColumns, 3 is AddRoundKey.')
@click.option('--op-substitution', 'operation', flag_value=0, default=True,
              help='Attack the SubBytes operation.')
@click.option('--op-shift-rows', 'operation', flag_value=1,
              help='Attack the ShiftRows operation.')
@click.option('--op-mix-columns', 'operation', flag_value=2,
              help='Attack the MixColumns operation.')
@click.option('--op-add-round-key', 'operation', flag_value=3,
              help='Attack the AddRoundKey operation.')
@click.option('--fs-none', 'feature_select', flag_value=0,
              help='Don\'t use feature selection')
@click.option('--fs-pearson', 'feature_select', flag_value=1, default=True,
              help='Use pearson for feature selection')
@click.option('--fs-sost', 'feature_select', flag_value=2,
              help='Use SOST for feature selection')
@click.option('--fs-nicv', 'feature_select', flag_value=3,
              help='Use Nicv for feature selection')
@click.option('--fs-sosd', 'feature_select', flag_value=4,
              help='Use SOSD for feature selection')
@click.option('--leakage-model', '-lm', is_flag=True,
              default=True,
              help='The leakage model used, the default is the hamming weights model, enable the flag to switch to the'
                   'intermediate values model.')
@click.option('--debug-mode', '-d', is_flag=True, default=False,
              help='Enables debug mode, a mode in which more detailed information about the execution of the attack is'
                   'printed and logged')
@click.option('--offline', '-off', 'version', flag_value=0, default=True,
              help='Enables offline CCC (correlation coefficient calculation).')
@click.option('--online', '-on', 'version', flag_value=1, default=False,
              help='Enables online CCC (correlation coefficient calculation).')
@click.option('--conditional-averaging', '-ca', 'version', flag_value=2, default=False,
              help='Enables conditional-averaging.')
def cpa(traces_file: str, keys_file: str, plain_file: str, subkey: int, leakage_model: bool,
        num_traces: int, num_attack_traces: int, num_features: int, feature_select: int, round_: int,
        operation: int, debug_mode: bool, version: int):
    """
    Performs different CPA attacks.

    :param traces_file: the traces file to use
    :param keys_file: the keys file to use
    :param plain_file: the plaintexts file to use
    :param subkey: the subkey index to analyze. Must be in the range [0-15]
    :param leakage_model: the leakage model to use.
    :param round_: the AES round to attack.
    :param num_traces: the amount of traces to analyze starting from trace 0.
    :param num_attack_traces: the amount of attack traces to analyze starting from trace 0.
    :param num_features: the number of features
    :param operation: the AES operation to 'attack', represented as an integer from 0 to 3.
    :param feature_select: which feature select to use
    :param debug_mode: whether to enable debug mode
    :param version: pick which version of CPA you want, the default is offline CPA.
    """

    online, conditional_averaging = False, False
    if version == 1:
        online = True
    elif version == 2:
        conditional_averaging = True

    traces, keys, plain = FileLoader.main(traces_file, keys_file, plain_file)
    profiling_traces, profiling_keys, profiling_plaintext, attack_traces, attack_keys, attack_plaintext = \
        DataPartitioner.get_traces(traces, keys, plain, num_traces, subkey,
                                   0, num_attack_traces, num_features, round_, operation, leakage_model)
    CPA.run(profiling_traces, profiling_keys, profiling_plaintext, attack_traces, attack_keys, attack_plaintext,
            round_, operation, subkey, feature_select, num_features, leakage_model,
            debug_mode_enabled=debug_mode, online=online, conditional_averaging=conditional_averaging)


@click.command()
@click.option('--traces-file', '-tf', type=str,
              default=CONST_DEFAULT_TRACES_FILE,
              help='The path for the traces to be used')
@click.option('--num-features', '-f', type=int,
              default=CONST_PEARSON_DEFAULT_POINTS_OF_INTEREST, show_default=True,
              help='The number of points of interest.')
def snr(traces_file: str, num_features: int):
    """
    Performs SNR

    :param traces_file: the file from which to load the traces
    :param num_features: the number of points of interest to extract.
    :return: the signal to noise ratio of the traces
    """

    traces = FileLoader.load_traces(traces_file)
    SNR.run(traces, num_features)


@click.command()
@click.option('--traces-file', '-tf', type=str,
              default=CONST_DEFAULT_TRACES_FILE,
              help='The path for the traces to be used')
@click.option('--keys-file', '-kf', type=str,
              default=CONST_DEFAULT_KEYS_FILE,
              help='The path for the keys to be used')
@click.option('--plain-file', '-pf', type=str,
              default=CONST_DEFAULT_PLAIN_FILE,
              help='The path for the plain to be used')
@click.option('--subkey', type=int,
              default=CONST_CPA_DEFAULT_SUBKEY,
              help='The subkey to find; must be in the range [0-15]. '
                   'When leaving this option out, the whole key is calculated.')
@click.option('--num_attack-traces', '-a', type=int,
              default=CONST_CPA_DEFAULT_NUM_ATTACK_TRACES, show_default=True)
@click.option('--round', '-r', 'round_', type=int,
              default=CONST_SA_DEFAULT_ROUND, show_default=True,
              help='The round to attack on.')
@click.option('-o', 'operation', type=int,
              help='Which operation to attack. This might override or be overridden by --op-* flags. '
                   '0 is SubBytes, 1 is ShiftRows, 2 is MixColumns, 3 is AddRoundKey.')
@click.option('--op-substitution', 'operation', flag_value=0, default=True,
              help='Attack the SubBytes operation.')
@click.option('--op-shift-rows', 'operation', flag_value=1,
              help='Attack the ShiftRows operation.')
@click.option('--op-mix-columns', 'operation', flag_value=2,
              help='Attack the MixColumns operation.')
@click.option('--op-add-round-key', 'operation', flag_value=3,
              help='Attack the AddRoundKey operation.')
@click.option('--bit', '-b', 'bit', type=int,
              default=CONST_DPA_DEFAULT_BIT,
              help='Bit to run the attack on')
@click.option('--offset', '-of', 'offset', type=int,
              default=CONST_DPA_DEFAULT_OFFSET,
              help='Offset used for second order dpa')
@click.option('--debug-mode', '-d', is_flag=True, default=False,
              help='Enables debug mode, a mode in which more detailed information about the execution of the attack is'
                   'printed and logged')
@click.option('--single-order', '-so', is_flag=True, default=False,
              help='Use single order dpa')
def dpa(traces_file: str, keys_file: str, plain_file: str, subkey: int, num_attack_traces: int,
        round_: int, operation: int, debug_mode: bool, bit: int, offset: int, single_order: bool):
    """Performs different DPA attacks.

    :param traces_file: the traces file to use
    :param keys_file: the keys file to use
    :param plain_file: the plaintexts file to use
    :param subkey: the subkey index to analyze. Must be in the range [0-15]
    :param round_: the AES round to attack.
    :param num_attack_traces: the amount of attack traces to analyze starting from trace 0.
    :param operation: the AES operation to 'attack', represented as an integer from 0 to 3.
    :param debug_mode: whether to enable debug mode
    :param bit: which bit to attack
    :param offset: the offset to use
    :param single_order: enables single order DPA
    """
    traces, keys, plain = FileLoader.main(traces_file, keys_file, plain_file)
    profiling_traces, profiling_keys, profiling_plaintext, attack_traces, attack_keys, attack_plaintext = \
        DataPartitioner.get_traces(traces, keys, plain, 0, subkey,
                                   0, num_attack_traces, 0, round_, operation, False)
    DPA.run(attack_traces, attack_keys, attack_plaintext,
            round_, operation, subkey, offset=offset, single_order=single_order,
            debug_mode_enabled=debug_mode, bit=bit)


@click.command()
@click.option('--traces-file', '-tf', type=str,
              default=CONST_DEFAULT_TRACES_FILE,
              help='The path for the traces to be used')
@click.option('--keys-file', '-kf', type=str,
              default=CONST_DEFAULT_KEYS_FILE,
              help='The path for the keys to be used')
@click.option('--plain-file', '-pf', type=str,
              default=CONST_DEFAULT_PLAIN_FILE,
              help='The path for the plain to be used')
@click.option('--subkey', type=int,
              default=CONST_MIA_DEFAULT_SUBKEY,
              help='The subkey to find; must be in the range [0-15]. '
                   'When leaving this option out, the whole key is calculated.')
@click.option('--traces', '-t', 'num_traces', type=int,
              default=CONST_MIA_DEFAULT_NUM_TRACES, show_default=True,
              help='The number of traces to run the attack with.')
@click.option('--num_attack-traces', '-a', type=int,
              default=CONST_MIA_DEFAULT_NUM_ATTACK_TRACES, show_default=True)
@click.option('--round', '-r', 'round_', type=int,
              default=CONST_SA_DEFAULT_ROUND, show_default=True,
              help='The round to attack on.')
@click.option('--debug-mode', '-d', is_flag=True, default=False,
              help='Enables debug mode, a mode in which more detailed information about the execution of the attack is'
                   'printed and logged')
def pia(traces_file: str, keys_file: str, plain_file: str, subkey: int, num_traces: int, num_attack_traces: int,
        round_: int,
        debug_mode: bool):
    """Performs PIA (Perceived Information Analysis).

    :param traces_file: the traces file to use
    :param keys_file: the keys file to use
    :param plain_file: the plaintexts file to use
    :param subkey: the subkey index to analyze. Must be in the range [0-15]
    :param round_: the AES round to attack.
    :param num_traces: the amount of traces to analyze starting from trace 0.
    :param num_attack_traces: the amount of attack traces to analyze starting from trace 0.
    :param num_features: the number of features
    :param feature_select: which feature select to use
    :param debug_mode: whether to enable debug mode
    """
    traces, keys, plain = FileLoader.main(traces_file, keys_file, plain_file)
    profiling_traces, profiling_keys, profiling_plaintext, attack_traces, attack_keys, attack_plaintext = \
        DataPartitioner.get_traces(traces, keys, plain, num_traces, subkey,
                                   0, num_attack_traces, 5000, round_, 0, False)

    Pia.run(profiling_traces, profiling_plaintext, profiling_keys, attack_traces, subkey, debug_mode_enabled=debug_mode)


@click.command()
@click.option('--aligned', '-al', 'aligned', type=str,
              default='none',
              help='File to align with.')
@click.option('--unaligned', '-u', 'unaligned', type=str,
              help='Unaligned file.')
@click.option('-a', 'algorithm', type=int,
              help='Which algorithm to use for trace alignment')
@click.option('-o', 'output', type=str,
              help='File to output to')
@click.option('--alg-dtw', 'algorithm', flag_value=0, default=True,
              help='Use the fastdtw algorithm')
@click.option('--alg-fft', 'algorithm', flag_value=1,
              help='Use the FFT algorithm')
@click.option('--debug-mode', '-d', is_flag=True, default=False,
              help='Enables debug mode, a mode in which more detailed information about the execution of the attack is'
                   'printed and logged')
def align(aligned: str, unaligned: str, algorithm: int, output: str, debug_mode: bool):
    """Aligns traces with specified algorithm and outputs an aligned file.

    :param aligned: File of 1 or more aligned traces.
    :param unaligned: File of unaligned traces
    :param algorithm: algorithm to use
    :param output: filename to output to.
    :param debug_mode: debug mode flag.
    :return:
    """

    unaligned_file = FileLoader.load_traces(unaligned)
    if aligned == 'none':
        aligned_file = np.zeros((1, len(unaligned[0])))
        aligned_file[0] = unaligned[0]
    else:
        aligned_file = FileLoader.load_traces(aligned)
    Aligner.run(aligned_file, unaligned_file, algorithm, output, debug_mode)


@click.command()
@click.option('--traces-file', '-tf', type=str,
              default=CONST_DEFAULT_TRACES_FILE,
              help='The path for the traces to be used')
@click.option('--keys-file', '-kf', type=str,
              default=CONST_DEFAULT_KEYS_FILE,
              help='The path for the keys to be used for feature selection')
@click.option('--plain-file', '-pf', type=str,
              default=CONST_DEFAULT_PLAIN_FILE,
              help='The path for the plain to be used')
@click.option('--traces', '-t', 'num_traces', type=int,
              default=CONST_LRA_DEFAULT_NUM_TRACES, show_default=True,
              help='The number of traces to run the attack with.')
@click.option('--subkey', type=int,
              default=CONST_MIA_DEFAULT_SUBKEY,
              help='The subkey to find; must be in the range [0-16], 16 signaling the whole key to be found out.')
@click.option('--num-features', '-f', type=int,
              default=CONST_LRA_NUM_FEATURES, show_default=True)
@click.option('--fs-none', 'feature_select', flag_value=0,
              help='Don\'t use feature selection')
@click.option('--fs-pearson', 'feature_select', flag_value=1, default=True,
              help='Use pearson for feature selection')
@click.option('--fs-sost', 'feature_select', flag_value=2,
              help='Use SOST for feature selection')
@click.option('--fs-nicv', 'feature_select', flag_value=3,
              help='Use Nicv for feature selection')
@click.option('--fs-sosd', 'feature_select', flag_value=4,
              help='Use SOSD for feature selection')
def lra(traces_file: str, keys_file: str, plain_file: str, num_traces: int, subkey: int, num_features: int,
        feature_select: int):
    """
    Performs non-linear regression analysis.

    :param traces_file: the traces file to use.
    :param keys_file: the keys file to use.
    :param plain_file: the plaintexts file to use.
    :param num_traces: the amount of traces to use
    :param subkey: the subkey index to analyze. Must be in the range [0-16], 16 signaling the whole key to be found out.
    :param num_features: number of features to select for.
    :param feature_select: which feature select to use, default is pearson.
    """

    traces, keys, plain = FileLoader.main(traces_file, keys_file, plain_file)
    traces = traces[:num_traces, :]
    keys = keys[:num_traces, :]
    plain = plain[:num_traces, :]

    lra_obj = LRA(traces, plain)
    lra_obj.run(traces, keys, plain, subkey, num_features, feature_select)


@click.group()
def cli():
    pass


if __name__ == "__main__":
    cli.add_command(mia)
    cli.add_command(nicv)
    cli.add_command(pearson)
    cli.add_command(sa)
    cli.add_command(ta)
    cli.add_command(sost)
    cli.add_command(sosd)
    cli.add_command(tvla)
    cli.add_command(cpa)
    cli.add_command(snr)
    cli.add_command(dpa)
    cli.add_command(pia)
    cli.add_command(align)
    cli.add_command(lra)
    cli()
