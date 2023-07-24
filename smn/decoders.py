import math
import random
from abc import ABC, abstractmethod
from typing import Iterable, List

from smn.markov import MarkovChain
from smn.utils import argsort


__all__ = [
    "Decoder",
    "TopKDecoder"
]


class Decoder(ABC):
    def __init__(self, chain: MarkovChain):
        self.chain = chain

    @abstractmethod
    def decode(self, prompt: Iterable[int]) -> Iterable[int]:
        raise NotImplementedError
    

class TopKDecoder(Decoder):
    """
    Top-K decoding.
    Naively generates token after token randomly choosing k-most probable tokens.
    """

    def __init__(
            self, 
            chain: MarkovChain, 
            k: int = 3, 
            max_len: int = 10, 
            len_penalization: float = 0.2,
            reps_penalization: float = 0.2,
            initial_eos: float = 0.0,
            reduce_k: bool = False
        ):

        """
        Args:
            chain (MarkovChain): Trained Markov chain.
            k (int): Number of most probable candidates to sample. Default: 3.
            max_len (int): Expected resulting sequence length. Uses for EOS-token probability update.
                           Does not necessarily result in expected sequence length. Default: 10.
            len_penalization (float): Penalization factor for sequence length. 
                                      Takes values from 0 to 1, where 0 - no penalization, 1 - vice versa. 
                                      Uses in pair with ```max_len```. Default: 0.2.
            reps_penalization (float): Penalization factor for new token repetitions. 
                                       Lowers probability of previous token to be sampled twice 
                                       by the factor of ```reps_penalization```. Default: 0.2.
            initial_eos (float): Initial probability for EOS-token. Smaller values result in longer sequences. 
                                 Default: 0.0.
            reduce_k (bool): Reduce k over generation process. Longer sequence -> lower k. Default: False.
        """

        super().__init__(chain)

        self._k = k
        self._max_len = max_len
        self._len_penalization = len_penalization
        self._reps_penalization = reps_penalization
        self._initial_eos = initial_eos
        self._eos_token_id = self.chain.dictionary.token2id["<eos>"]
        self._reduce_k = reduce_k

    def _penalize_for_length(
            self, 
            probs: Iterable[float], 
            sequence: Iterable[int]
        ) -> List[float]:
        updated_prob = self._initial_eos + self._len_penalization * len(sequence) / self._max_len
        probs[self._eos_token_id] = updated_prob

        return probs
    
    def _penalize_for_repetitions(
            self,
            probs: Iterable[float], 
            sequence: Iterable[int]
        ) -> List[float]:
        if sequence:
            probs[sequence[-1]] *= (1 - self._reps_penalization)

        return probs

    def decode(self, sequence: Iterable[int]) -> Iterable[int]:
        idx = -1
        while len(sequence) != self._max_len:
            probs = self.chain.generate_next_token_distribution(sequence)

            # Penalize
            probs = self._penalize_for_length(probs, sequence)
            probs = self._penalize_for_repetitions(probs, sequence)

            # Select k most probable
            if self._reduce_k:
                k = min(self._k * 1 / (len(sequence) / 2 + 1e-10), self._k)
                k = int(max(2, k))
            else:
                k = self._k
            top_k_ids = argsort(probs)[-k:]
            idx = random.choice(top_k_ids)

            if idx == self._eos_token_id:
                break

            # Cocnat to sequence
            sequence += [idx]

        generated = "".join([self.chain.dictionary.id2token[id_] for id_ in sequence])

        return generated