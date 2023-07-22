import pickle
from typing import Iterable, List, Dict

from collections import defaultdict, Counter

from smn.tokenizers import Tokenizer
from smn.dictionary import Dictionary

    

class MarkovChain:
    _states: Dict[List[int], int] = defaultdict(list)

    def __init__(self, tokenizer: Tokenizer, order: int = 1):
        self.order = order
        self._tokenizer = tokenizer

        self._dict = self._tokenizer.dictionary

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer
    
    @property
    def dictionary(self) -> Dictionary:
        return self._tokenizer.dictionary

    def _preprocess_sequence(self, sequence: Iterable[int]) -> List[int]:
        """Adds start and end tokens. Pads to left using specified self._order"""

        sequence = [self._dict.token2id["<sos>"]] + sequence + [self._dict.token2id["<eos>"]]
        sequence = [self._dict.token2id["<pad>"]] * (self.order - 1) + sequence

        return sequence
    
    def fit(self, sequences: Iterable[Iterable[int]]):

        sequences = [self._preprocess_sequence(seq) for seq in sequences]

        # Memorize transitions
        for seq in sequences:
            for i in range(self.order, len(seq)):
                condition = tuple(seq[i - self.order:i])
                self._states[condition].append(seq[i])

        # Compute transition probabilities
        for condition, outcomes in self._states.items():
            probs = [0.0] * len(self._dict)
            for outcome, counts in Counter(outcomes).items():
                probs[outcome] = counts / len(outcomes)

            self._states[condition] = probs

    def generate_next_token_distribution(self, sequence: Iterable[int]) -> List[int]:

        if len(sequence) < self.order:
            # Drop last <eos> token
            sequence = self._preprocess_sequence(sequence)[:-1]

        condition = tuple(sequence[-self.order:])
        if self._states.get(condition, None) is None:
            # Init uniformely if condition was not in train
            n = len(self.dictionary)
            probs = [1 / n] * n
        else:
            probs = self._states[tuple(sequence[-self.order:])].copy()

        return probs
