import pickle
import numpy as np
import itertools


def mean_novelty(seqs, ref_seqs):
    novelties = []
    for seq in seqs:
        novelties.append(np.min([edit_dist(*pair) for pair in itertools.product([seq], ref_seqs)]))
    return np.mean(novelties)

def mean_pairwise_distances(seqs):
    dists = []
    for pair in itertools.combinations(seqs, 2):
        dists.append(edit_dist(*pair))
    return np.mean(dists)

def edit_dist(seq1, seq2):
    return sum([x!=y for x, y in zip(seq1, seq2)])
    


class ExperimentTracker:
    def __init__(self, logger, dataset, wt_sequence=None, best_percentile=0.95):
        self.logger = logger
        self.train_sequences = dataset.train if wt_sequence is None else self.get_best_sequences(dataset, best_percentile)
        self.wt_sequence = wt_sequence
        self.collected_sequences = list()
        self.collected_scores = list()
        self.exp_results = dict()

    def get_best_sequences(self, dataset, percentile):
        train_sequences = dataset.train
        train_scores = dataset.train_scores
        thr = np.percentile(train_scores, percentile)
        return train_sequences[train_scores >= thr].tolist()

    def add_records(self, new_records):
        self.collected_sequences += new_records[0]
        self.collected_scores += new_records[1]
        return new_records[0], new_records[1]

    def calculate_top_n_metrics(self, new_records, n_iter, n=100):
        iter_sequences, iter_scores = self.add_records(new_records)
        indices = np.argsort(self.collected_scores)[::-1][:n]
        top_sequences = np.array(self.collected_sequences)[indices]
        top_scores = np.array(self.collected_scores)[indices]
        top_performances = np.mean(top_scores)
        top_distances = mean_pairwise_distances(top_sequences)
        top_novelties = mean_novelty(top_sequences, self.train_sequences)
        if self.wt_sequence is not None:
            wt_novelty = mean_novelty(top_sequences, [self.wt_sequence])
        top_performances_median = np.percentile(top_scores, 50)
        best_score = np.max(self.collected_scores)
        
        self.logger.info(
            f"Top {n} collected scores after iteration {n_iter}: | Performance: {top_performances} | Diversity: {top_distances} | Novelty: {top_novelties} | WT Novelty: {wt_novelty if self.wt_sequence is not None else None} | Best Score: {best_score} | Performance median: {top_performances_median}"
        )

        self.exp_results[n_iter] = {
            "Sequences": top_sequences.tolist(),
            "Scores": top_scores.tolist(),
            "Performance": top_performances,
            "Diversity": top_distances,
            "Novelty": top_novelties,
            "WT Novelty": wt_novelty if self.wt_sequence is not None else "",
            "Best score": best_score,
            "Median performance": top_performances_median,
            "Iter sequences": iter_sequences,
            "Iter scores": iter_scores
        }

    def save_results(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.exp_results, f)
            