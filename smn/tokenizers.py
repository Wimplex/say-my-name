import re
from abc import ABC, abstractmethod
from typing import Iterable, List

from smn.dictionary import Dictionary


__all__ = [
    "Tokenizer",
    "CharTokenizer"
]


class Tokenizer(ABC):
    _dictionary: Dictionary = None

    @property
    def dictionary(self) -> Dictionary:
        return self._dictionary
    
    def _fit_dictionary(self, tokenized_sequences: Iterable[Iterable[str]]):
        self._dictionary = Dictionary()
        self._dictionary.fit(tokenized_sequences)

    @abstractmethod
    def fit(self, sequences: Iterable[str]):
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, sequences: Iterable[str]) -> List[List[int]]:
        raise NotImplementedError


class CharTokenizer(Tokenizer):
    _punct_to_delete = '!"#$%&()*+,./:;<=>?@[\]^_`{|}~'
    _punct_delete_regex = re.compile("[%s]" % re.escape(_punct_to_delete))

    def fit(self, sequences: Iterable[str]):
        tokenized = [self._preprocess_string(st) for st in sequences]
        self._fit_dictionary(tokenized)

    def _preprocess_string(self, string: str) -> str:
        string = string.lower()
        tokenized = re.sub(self._punct_delete_regex, "", string)

        return tokenized

    def tokenize(self, sequences: Iterable[str]) -> List[List[int]]:
        tokenized = [self._preprocess_string(st) for st in sequences]
        transformed = self.dictionary.transform(tokenized)

        return transformed
    

class BPETokenizer(CharTokenizer):
    def __init__(vocab_size: int):
        self.vocab_size = vocab_size

    def fit(self, sequences: Iterable[str]):
        pass