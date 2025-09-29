import numpy as np
import argparse
import os
import json
import pickle


def extract_data(data, n_iter):
    data = [i[n_iter] for i in data]
    seqs = [i["Sequences"] for i in data]
    max_scores = [i["Best score"] for i in data]
    diversity = [i["Diversity"] for i in data]
    novelty = [i["WT Novelty"] for i in data]
    mean_scores = [i["Performance"] for i in data]
    median_scores = [i["Median performance"] for i in data]

    mean_performance = np.mean(mean_scores).round(3)
    mean_diversity = np.mean(diversity).round(3)
    mean_novelty_score = np.mean(novelty).round(3)
    mean_max_score = np.mean(max_scores).round(3)
    mean_median_score = np.mean(median_scores).round(3)
    
    std_performance = np.std(mean_scores).round(3)
    std_diversity = np.std(diversity).round(3)
    std_novelty = np.std(novelty).round(3)
    std_max_score = np.std(max_scores).round(3)
    std_median_score = np.std(median_scores).round(3)

    iter_data = {
        "Sequences": np.concatenate(seqs).tolist(),
        "Mean max score": mean_max_score,
        "Std max score": std_max_score,
        "Mean performance": mean_performance,
        "Std performance": std_performance,
        "Mean diversity": mean_diversity,
        "Std diversity": std_diversity,
        "Mean novelty": mean_novelty_score,
        "Std novelty": std_novelty,
        "Mean median performance": mean_median_score,
        "Std median performance": std_median_score,
    }
    return iter_data    


def etl_scores(data, path, n_iters):
    avg_data = {n_iter: extract_data(data, n_iter) for n_iter in range(1, n_iters + 1)}
    with open(path, "w") as f:
        json.dump(avg_data, f)
    return avg_data


def get_seed_data(path):
    seed_data = []
    for item in ["seed_1.pkl", "seed_2.pkl", "seed_3.pkl", "seed_4.pkl", "seed_5.pkl"]:
        with open(os.path.join(path, item), "rb") as f:
            data = pickle.load(f)
            seed_data.append(data)
    return seed_data



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dirpath", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--n_iters", type=int, default=10)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    task_path = os.path.join(args.results_dirpath, args.task)
    seed_data = get_seed_data(task_path)
    etl_scores(seed_data, os.path.join(task_path, "transformed_results.json"), args.n_iters)


if __name__ == "__main__":
    main()