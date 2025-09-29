import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import sys

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)

class Sampler:
    def __init__(self, model, tokenizer, alphabet):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unrolled_scores = dict()
        self.alphabet = alphabet
        self.token_to_cluster = self.get_token_clusters()
        self.PAD = -1

    def get_token_clusters(self):
        aa_to_token = {i: c for c, i in enumerate(self.tokenizer.all_aas[:20])}
        tokens_alphabet = {aa_to_token[k]: self.tokenizer.tokenize([v]) for k, v in self.alphabet.items()}
        return tokens_alphabet

    def get_top_sequences(self, k_best, ref_sequences):
        ref_sequences = set(ref_sequences)
        sorted_seqs = sorted(self.unrolled_scores, key=lambda x: self.unrolled_scores[x], reverse=True)
        filtered_seqs = [seq for seq in sorted_seqs if seq not in ref_sequences]
        self.unrolled_scores = dict()
        return filtered_seqs[:k_best]


    def is_resampling_step(self, step, resampling_steps):
        if isinstance(resampling_steps, list):
            return step in resampling_steps
        else:
            return not step % resampling_steps

    def sample_lin_indices(self, raw_scores):
        raw_scores = raw_scores.cpu()
        raw_scores -= raw_scores.min()
        raw_scores += 1e-8
        weights = raw_scores / raw_scores.sum()
        indices = torch.multinomial(weights, len(weights), replacement=True)
        return indices.to(self.device)

    def sample_n_corruptions_uniform(self, min_corruptions, max_corruptions):
        return np.random.randint(min_corruptions, max_corruptions + 1)    


class ProteinSampler(Sampler):
    def __init__(self, model, tokenizer, alphabet):
        super().__init__(model, tokenizer, alphabet)
                
    def shotgun_alanine_scan(self, sequence, proxy, min_corruptions, max_corruptions, batch_size, n_checks_multiplier, k):
        """
        Targeted Masking (Algorithm 2)
        """
        tokenized_seq = self.tokenizer.tokenize([sequence])
        maskable_tokens = np.array(list(self.token_to_cluster))
        maskable_ids = np.nonzero(np.isin(tokenized_seq, maskable_tokens))[0]

        batch_substituted = np.empty((batch_size * n_checks_multiplier, len(sequence)), dtype="U1")
        sampled_ids_list = list()
        for n in range(batch_size * n_checks_multiplier):
            n_corruptions = self.sample_n_corruptions_uniform(min_corruptions, max_corruptions) # line 2
            try:
                sampled_ids = np.random.choice(maskable_ids, n_corruptions, replace=False) # line 3
            except ValueError:
                sampled_ids = maskable_ids
            split_sequence = np.array(list(sequence))
            split_sequence[sampled_ids] = "A" # line 4
            batch_substituted[n] = split_sequence
            sampled_ids_list.append(sampled_ids)
            
        substituted_seqs = ["".join(s) for s in batch_substituted]
        mean, std = proxy.forward_with_uncertainty(substituted_seqs)
        ucb = (mean + k * std).detach().cpu().numpy() # line 5
        best_ids = np.argsort(ucb)[::-1][:batch_size]
        
        batch = np.tile(tokenized_seq, (batch_size, 1))
        ids = [sampled_ids_list[idx] for idx in best_ids]
        
        locs = list()
        og_tokens_to_clusters = list()
        for row, row_ids in zip(batch, ids):
            loc = np.array(sorted(row_ids, key=lambda x: len(self.token_to_cluster[row[x]])))
            og_tokens_to_clusters.append({i: self.token_to_cluster[row[i]] for i in loc})
            row[loc] = self.tokenizer.mask_id # line 7
            locs.append(loc)

        max_loc_length = np.max([len(loc) for loc in locs])
        locs = np.array([np.pad(loc, (0, max_loc_length - len(loc)), constant_values=self.PAD) for loc in locs])
        sample = torch.tensor(batch, device=self.device)
        og_tokens_to_clusters = np.array([{k: torch.tensor(v, device=self.device) for k, v in item.items()} for item in og_tokens_to_clusters])

        return sample, locs, og_tokens_to_clusters
  
    @torch.no_grad
    def generate_raa_from_alanine_scan(
        self, guide, starting_sequence, batch_size, resampling_steps, min_corruptions, max_corruptions, kappa_scan, n_checks_multiplier, kappa_guidance
    ):
        """
        Biologically-constrained SMC (Algorithm 3)
        """
        sample, locs, og_tokens_to_clusters = self.shotgun_alanine_scan(
            starting_sequence, guide, min_corruptions, max_corruptions, batch_size, n_checks_multiplier, kappa_scan
        )
        
        steps = len(locs[0])
        batch_ll = torch.zeros(batch_size, device=self.device)
        n_predictions = torch.tensor((locs != self.PAD).sum(axis=1), device=self.device)
        for i in tqdm(range(steps)):
            steps_left = abs(i - steps)
            samples_left = np.nonzero(locs[:, i] != self.PAD)[0]
            if not len(samples_left):
                break
            timestep = torch.tensor([0] * len(samples_left)).to(self.device)
            prediction = self.model(sample, timestep)
            p = prediction[samples_left, locs[samples_left, i]]

            sampled_aas = list()
            for logits, og_token_to_cluster, og_idx in zip(p, og_tokens_to_clusters[samples_left], locs[samples_left, i]):
                probs = torch.nn.functional.softmax(logits[og_token_to_cluster[og_idx]], dim=0) # line 10
                aa_idx = torch.multinomial(probs, num_samples=1)
                sampled_aas.append(og_token_to_cluster[og_idx][aa_idx])
            sampled_aas = torch.cat(sampled_aas)
            sample[samples_left, locs[samples_left, i]] = sampled_aas

            log_probs = F.log_softmax(p[:, :20], dim=1)
            ll = log_probs[torch.arange(p.shape[0]), sampled_aas] # line 11
            batch_ll[samples_left] += ll
            
            if self.is_resampling_step(steps_left, resampling_steps):
                unrolled_sample, unrolled_ll = self.unroll_from_alanine_scan(sample, locs[:, i+1:], og_tokens_to_clusters, batch_ll) # line 12
                inv_perplexity = 1 / torch.exp(-unrolled_ll / n_predictions) # line 14
                scores = guide.get_ucb(unrolled_sample, kappa_guidance) # line 13
                if steps_left < 10:
                    self.unrolled_scores |= dict(zip(unrolled_sample, scores))
            
                # resample
                ids = self.sample_lin_indices(scores * inv_perplexity)
                sample = sample[ids]
                batch_ll = batch_ll[ids]
                n_predictions = n_predictions[ids]
                ids = ids.cpu().numpy()
                locs = locs[ids]
                og_tokens_to_clusters = og_tokens_to_clusters[ids]
    

    @torch.no_grad
    def unroll_from_alanine_scan(self, sample, remaining_locs, og_tokens_to_clusters, batch_ll):
        """
        Rollout (Algorithm 4)
        """
        batch_ll = deepcopy(batch_ll)
        sample = deepcopy(sample)
        for i in range(len(remaining_locs[0])):
            samples_left = np.nonzero(remaining_locs[:, i] != self.PAD)[0]
            if not len(samples_left):
                break
            timestep = torch.tensor([0] * len(samples_left)).to(self.device)
            prediction = self.model(sample, timestep)
            p = prediction[samples_left, remaining_locs[samples_left, i]]

            sampled_aas = list()
            for logits, og_token_to_cluster, og_idx in zip(p, og_tokens_to_clusters[samples_left], remaining_locs[samples_left, i]):
                probs = torch.nn.functional.softmax(logits[og_token_to_cluster[og_idx]], dim=0) # line 5
                aa_idx = torch.multinomial(probs, num_samples=1)
                sampled_aas.append(og_token_to_cluster[og_idx][aa_idx])
            sampled_aas = torch.cat(sampled_aas)
            sample[samples_left, remaining_locs[samples_left, i]] = sampled_aas

            log_probs = F.log_softmax(p[:, :20], dim=1)
            ll = log_probs[torch.arange(p.shape[0]), sampled_aas] # line 6
            batch_ll[samples_left] += ll


        untokenized = [self.tokenizer.untokenize(s) for s in sample]
        return untokenized, batch_ll
    