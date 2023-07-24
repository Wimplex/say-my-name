import json
import argparse
from typing import Tuple, Dict, Any, List, Iterable

from smn.utils import CONFIGS_DIR
from smn.markov import MarkovChain
from smn.decoders import *
from smn.tokenizers import *


# TODO:
# 1. sequence probability computation
# 2. More decoding (beam search for example)
# 3. ppl evaluation
# 4. more data (scrap)
# 5. additional factors conditioning (sex, country, etc...)
# 6. Saving and loading


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--config", type=str, default=None)

    return parser.parse_args()


def load_config(config_name: str) -> Dict[str, Any]:
    with open(CONFIGS_DIR / f"{config_name}.json", "r") as file:
        config = json.load(file)

    return config


def read_train_set(name: str) -> List[str]:
    with open(f"data/{name}", "r", encoding="utf-8") as file:
        lines = file.readlines()
        data = [line.strip() for line in lines]

    return data


class Pipeline:

    chain: MarkovChain
    tokenizer: Tokenizer
    decoder: Decoder

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config

        self._trained = False

    def _init_tokenizer(self):
        if self.config:
            tokenizer_type = self.config.get("tokenizer", None)

            # Here could be more tokenizer variants in future
            if tokenizer_type == "char":
                self.tokenizer = CharTokenizer()
            elif tokenizer_type is None:
                self.tokenizer = CharTokenizer()
        else:
            self.tokenizer = CharTokenizer()

    def _init_chain(self):
        if self.config:
            if self.config.get("chain", None) is not None:
                self.chain = MarkovChain(self.tokenizer, **self.config["chain"])
            else:
                self.chain = MarkovChain(self.tokenizer)
        else:
            self.chain = MarkovChain(self.tokenizer)

    def _init_decoder(self):
        if self.config:
            decoder_cfg = self.config.get("decoder", None)

            if decoder_cfg is not None:
                self.decoder = TopKDecoder(self.chain, **decoder_cfg)
            else:
                self.decoder = TopKDecoder(self.chain)
        else:
            self.decoder = TopKDecoder(self.chain)

    def train(self, data: Iterable[str]) -> Tuple[MarkovChain, Tokenizer]:

        # Tokenize data
        self._init_tokenizer()
        self.tokenizer.fit(data)
        tokenized = self.tokenizer.tokenize(data)

        # Train MC model
        self._init_chain()
        self.chain.fit(tokenized)

        self._trained = True

    def generate(self, prompt: str):
        assert self._trained, "Chain did not trained"

        test_tokenized = self.tokenizer.tokenize([prompt])

        if getattr(self, "decoder", None) is None:
            self._init_decoder()

        generated = self.decoder.decode(test_tokenized[0])
        generated = generated.capitalize()
        
        return generated


def main():
    args = parse_args()
    config = load_config(args.config) if args.config is not None else None

    train_data = read_train_set(config['data'])

    pipe = Pipeline(config)
    pipe.train(train_data)
    
    for _ in range(args.count):
        res = pipe.generate(args.prompt)
        print(res)


if __name__ == "__main__":
    main()