from typing import Generic, TypeVar, Iterable
from functools import cached_property


T = TypeVar("T")


class Dictionary(Generic[T]):

    SPECIAL_TOKENS = ["<sos>", "<eos>", "<pad>"]

    def __init__(self):
        self._dictionary = []

    def _fit_check(self):
        assert self._dictionary, "Dictionary didn't fitted yet."

    @cached_property
    def token2id(self):
        self._fit_check()
        
        return {t: i for i, t in enumerate(self._dictionary)}
    
    @property
    def id2token(self):
        self._fit_check()

        return self._dictionary

    def fit(self, sequences: Iterable[Iterable[T]]):
        """Fill dictionary with tokens from input sequences"""

        unique_tokens = set([tok for seq in sequences for tok in seq])
        sorted_tokens = list(sorted(list(set(unique_tokens))))

        self._dictionary = self.SPECIAL_TOKENS + sorted_tokens

    def transform(self, sequences: Iterable[Iterable[T]]) -> Iterable[Iterable[int]]:
        """Transforms list of original sequences to list of int sequences"""

        transformed = []
        for seq in sequences:
            tr_seq = [self.token2id[tok] for tok in seq]
            transformed.append(tr_seq)

        return transformed

    def __len__(self) -> int:
        self._fit_check()
        
        return len(self._dictionary)