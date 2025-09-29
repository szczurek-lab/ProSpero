import sys
import os
from evodiff.pretrained import OA_DM_38M
from prospero.experiments_config import ALPHABETS, WT_SEQUENCES

from prospero.utils import set_seed, get_new_starting_seq
from prospero.experiment_tracker import ExperimentTracker
from prospero.inference import ProteinSampler

from prospero.dataset import RegressionDataset
from prospero.landscapes import get_landscape, NoisyLandscape

import argparse
from argparse import ArgumentDefaultsHelpFormatter
import numpy as np
from copy import deepcopy

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Unified Argument Parser for Oracle, Dataset, and Proxy Arguments",
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    # Experiment arguments
    parser.add_argument("--results_dirpath", type=str)
    parser.add_argument("--n_iters", type=int, default=10)
    parser.add_argument("--n_queries", type=int, default=128)
    parser.add_argument("--seed", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--snr", type=int)
    parser.add_argument("--full_deterministic", action="store_true", default=False)

    # Sampler arguments
    parser.add_argument("--resampling_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--alphabet", type=str, default="CHARGE")
    parser.add_argument("--kappa_scan", type=float, default=1.0)
    parser.add_argument("--kappa_guidance", type=float, default=0.1)
    parser.add_argument("--n_checks_multiplier", type=int, default=16)

    # proxy arguments
    parser.add_argument("--ensemble_size", type=int, default=3)

    return parser



def run_iter(args, logger):
    seed = args.seed
    set_seed(seed, args.full_deterministic)
    args.task = "AAV"
    logger.info(f"Starting seed {seed}")

    save_path = os.path.join(args.results_dirpath, f"noise_level_{args.snr}", f"seed_{seed}.pkl")
    wt_sequence = WT_SEQUENCES[args.task]
    oracle = get_landscape(args.task)
    dataset = RegressionDataset(args.task)
    signal_variance = np.var(np.concatenate([dataset.train_scores, dataset.valid_scores]))
    proxy = NoisyLandscape(args.ensemble_size, args.snr, signal_variance)

    alphabet = ALPHABETS[args.alphabet]

    model, _, tokenizer_oadm, _ = OA_DM_38M()
    model = model.cuda()
    exp_tracker = ExperimentTracker(logger, deepcopy(dataset), wt_sequence, best_percentile=0.95)

    starting_sequence = WT_SEQUENCES[args.task]
    min_corruptions = 3
    max_corruptions = 10

    for e in range(args.n_iters):
        sampler = ProteinSampler(model, tokenizer_oadm, alphabet)
        sequences = list()
        ref_sequences = dataset.train.tolist() + dataset.valid.tolist()
        while len(sequences) < args.n_queries:
            sampler.generate_raa_from_alanine_scan(
                proxy, starting_sequence, args.batch_size, args.resampling_steps, min_corruptions, 
                max_corruptions, args.kappa_scan, args.n_checks_multiplier, args.kappa_guidance
            )
            sequences += sampler.get_top_sequences(args.n_queries, ref_sequences)
            ref_sequences += sequences

        sequences = sequences[:args.n_queries]
        assert len(sequences) == args.n_queries


        scores = oracle.get_fitness(np.array(sequences)).tolist()
        dataset.add((sequences, scores))
        exp_tracker.calculate_top_n_metrics((sequences, scores), e + 1, n=100)
        starting_sequence = get_new_starting_seq(dataset)
        exp_tracker.save_results(save_path)


def main():
    parser = get_parser()
    args = parser.parse_args()
    run_iter(args, logger)


if __name__ == "__main__":
    main()
