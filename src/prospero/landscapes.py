import numpy as np
import torch
import os
import tape
import json
from prospero.experiments_config import ORACLES_PATH
from transformers import AutoTokenizer, EsmForProteinFolding
from tqdm import tqdm


class AAVLandscape:

    def __init__(
        self,
        oracle_path: str,
        phenotype: str = "liver",
        minimum_fitness_multiplier: float = 1,
        start: int = 450,
        end: int = 540,
        noise: int = 0,
    ):
        self.sequences = {}
        self.phenotype = f"log2_{phenotype}_v_wt"

        self.mfm = minimum_fitness_multiplier
        self.start = start
        self.end = end
        self.noise = noise
        self.wild_type = 'PSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVL'

        with open(oracle_path) as f:
            self.data = {
                int(pos): val
                for pos, val in json.load(f).items()
                if self.start <= int(pos) < self.end
            }

        self.top_seq, self.max_possible = self.compute_max_possible()

    def compute_max_possible(self):
        """Compute max possible fitness of any sequence (used for normalization)."""
        best_seq = ""
        max_fitness = 0
        for pos in self.data:
            current_max = -10
            current_best = "M"
            for aa in self.data[pos]:
                current_fit = self.data[pos][aa][self.phenotype]
                if (
                    current_fit > current_max
                    and self.data[pos][aa]["log2_packaging_v_wt"] > -6
                ):
                    current_best = aa
                    current_max = current_fit

            best_seq += current_best
            max_fitness += current_max
        return best_seq, max_fitness

    def _get_raw_fitness(self, seq):
        total_fitness = 0
        for i, s in enumerate(seq):
            if s in self.data[self.start + i]:
                total_fitness += self.data[self.start + i][s][self.phenotype]

        return total_fitness + self.mfm * self.max_possible

    def get_fitness(self, sequences):
        fitnesses = []
        for seq in sequences:
            normed_fitness = self._get_raw_fitness(seq) / (
                self.max_possible * (self.mfm + 1)
            )
            fitness_with_noise = normed_fitness + np.random.normal(scale=self.noise)
            fitnesses.append(max(0, fitness_with_noise))

        return np.array(fitnesses)


class TAPELandscape:
    """
        A TAPE-based oracle model to simulate protein fitness landscape.
    """
    
    def __init__(self, task):
        task_dir_path = os.path.join(ORACLES_PATH, task)
        print(task_dir_path)
        assert os.path.exists(os.path.join(task_dir_path, 'pytorch_model.bin'))
        self.model = tape.ProteinBertForValuePrediction.from_pretrained(task_dir_path)        
        self.tokenizer = tape.TAPETokenizer(vocab='iupac')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def get_fitness(self, sequences):
        sequences = np.array(sequences)
        scores = []

        # Score sequences in batches of size 32
        for subset in np.array_split(sequences, max(1, len(sequences) // 32)):
            encoded_seqs = torch.tensor(
                [self.tokenizer.encode(seq) for seq in subset]
            ).to(self.device)

            scores.append(
                self.model(encoded_seqs)[0].detach().cpu().numpy().astype(float).reshape(-1)
            )

        return np.concatenate(scores)
    

class NoisyLandscape:
    def __init__(self, ensemble_size, snr, signal_variance):
        noise_std = self.calculate_noise_std(snr, signal_variance)
        self.oracle = [
            AAVLandscape(os.path.join(ORACLES_PATH, "AAV2_single_subs-2.json"), noise=noise_std) for _ in range(ensemble_size)
        ]

    def calculate_noise_std(self, snr, signal_variance):
        noise_var = signal_variance * 10 ** (-snr / 10)
        return np.sqrt(noise_var)

    def _call_models(self, sequences):
        return torch.stack([torch.Tensor(o.get_fitness(sequences)) for o in self.oracle])

    def get_fitness(self, sequences):
        outputs = self._call_models(sequences)
        return outputs.mean(dim=0)

    def get_scores(self, sequences):
        return self._call_models(sequences).mean(dim=0)

    def forward_with_uncertainty(self, sequences):
        outputs = self._call_models(sequences)
        return outputs.mean(dim=0), outputs.std(dim=0)

    def get_ucb(self, sequences, k=1.0):
        outputs = self._call_models(sequences).cuda()
        return outputs.mean(dim=0) + k * outputs.std(dim=0)


def get_landscape(task):
    if task == "AAV":
        return AAVLandscape(os.path.join(ORACLES_PATH, "AAV2_single_subs-2.json"))
    elif task in ["D_SHIFT", "D_SHIFT_SMALL", "D_SHIFT_HARD"]:
        return ESMFoldLandscape()
    else:
        return TAPELandscape(task)
    


class ESMFoldLandscape:
    def __init__(self):
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        try:
            model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True, use_safetensors=True)
        except OSError:
            model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True, use_safetensors=False)
        model = model.cuda()
        return model, tokenizer
    
    @torch.no_grad()
    def get_fitness(self, sequences):
        outputs = []
        seqs_tokenized = self.tokenizer(sequences, padding=False, add_special_tokens=False)["input_ids"]
        for input_ids in tqdm(seqs_tokenized):
            input_ids = torch.tensor(input_ids, device='cuda').unsqueeze(0)
            output = self.model(input_ids)
            outputs.append({key: val.cpu() for key, val in output.items()})
        ptms = [pred["ptm"].item() for pred in outputs]
        return np.array(ptms)
    