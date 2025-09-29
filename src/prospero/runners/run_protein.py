import sys
import os
from evodiff.pretrained import OA_DM_38M
from prospero.experiments_config import ALPHABETS, WT_SEQUENCES


from prospero.utils import set_seed, get_new_starting_seq, get_new_starting_seq_dshift
from prospero.experiment_tracker import ExperimentTracker
from prospero.inference import ProteinSampler

from prospero.surrogate import Ensemble, ConvolutionalNetworkModel
from prospero.dataset import RegressionDataset
from prospero.landscapes import get_landscape

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
    parser.add_argument("--n_queries", type=int, default=128)
    parser.add_argument("--seed", type=int, choices=[1, 2, 3, 4, 5], default=1)
    parser.add_argument("--task", type=str, choices=list(WT_SEQUENCES))
    parser.add_argument("--full_deterministic", action="store_true", default=False)
    parser.add_argument("--n_iters", type=int, default=10)

    # Sampler arguments
    parser.add_argument("--resampling_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--alphabet", type=str, default="CHARGE")
    parser.add_argument("--kappa_scan", type=float, default=1.0)
    parser.add_argument("--kappa_guidance", type=float, default=0.1)
    parser.add_argument("--n_checks_multiplier", type=int, default=16)
    parser.add_argument("--min_corruptions", type=int, default=3)
    parser.add_argument("--max_corruptions", type=int, default=10)

    # Proxy arguments
    parser.add_argument("--num_model_max_epochs", type=int, default=3000)
    parser.add_argument("--ensemble_size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs_per_valid", type=int, default=1)
    parser.add_argument("--proxy_batch_size", type=int, default=256)

    return parser



def run_iter(args, logger):
    seed = args.seed
    set_seed(seed, args.full_deterministic)
    logger.info(f"Starting seed {seed}")

    save_dir = os.path.join(args.results_dirpath, args.task)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"seed_{seed}.pkl")

    wt_sequence = WT_SEQUENCES[args.task]
    oracle = get_landscape(args.task)
    dataset = RegressionDataset(args.task)

    proxy = Ensemble(
        [ConvolutionalNetworkModel(len(wt_sequence), args) for _ in range(args.ensemble_size)]
    )
    logger.info("Training started")
    proxy.train(dataset)
    logger.info("Training finished")
    
    alphabet = ALPHABETS[args.alphabet]
    
    model, _, tokenizer_oadm, _ = OA_DM_38M()
    model = model.cuda()
    exp_tracker = ExperimentTracker(logger, deepcopy(dataset), wt_sequence, best_percentile=0.95)

    starting_sequence = WT_SEQUENCES[args.task]

    for e in range(args.n_iters):
        sampler = ProteinSampler(model, tokenizer_oadm, alphabet)
        sequences = list()
        ref_sequences = dataset.train.tolist() + dataset.valid.tolist()
        # generate new sequences
        while len(sequences) < args.n_queries:
            sampler.generate_raa_from_alanine_scan(
                proxy, starting_sequence, args.batch_size, args.resampling_steps, args.min_corruptions, 
                args.max_corruptions, args.kappa_scan, args.n_checks_multiplier, args.kappa_guidance, 
            )
            sequences += sampler.get_top_sequences(args.n_queries, ref_sequences)
            ref_sequences += sequences

        sequences = sequences[:args.n_queries]
        assert len(sequences) == args.n_queries

        # eval candidate sequences
        if not args.task.startswith("D_SHIFT"):
            scores = oracle.get_fitness(np.array(sequences)).tolist()
        else:
            scores = oracle.get_fitness(sequences).tolist()
        
        # append dataset and retrain the surrogate
        dataset.add((sequences, scores))
        exp_tracker.calculate_top_n_metrics((sequences, scores), e + 1, n=100)
        starting_sequence = get_new_starting_seq(dataset) if not args.task.startswith("D_SHIFT") else get_new_starting_seq_dshift(dataset, args.task)

        proxy = Ensemble(
            [ConvolutionalNetworkModel(len(wt_sequence), args) for _ in range(args.ensemble_size)]
        )
        exp_tracker.save_results(save_path)
        if e + 1 < args.n_iters:
            proxy.train(dataset)


def main():
    parser = get_parser()
    args = parser.parse_args()
    run_iter(args, logger)


if __name__ == "__main__":
    main()
