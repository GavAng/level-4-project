from collections.abc import Callable
from typing import TypeVar
import numpy as np
import torch


T = TypeVar("T", bound=torch.Tensor)
V = TypeVar("V", bound=torch.Tensor)


class Data:
    def __init__(self, data: list[tuple[T, V]], seed: int = 42):
        n_examples = len(data)
        inputs, labels = zip(*data)

        # standard machine learning train, validation, test split
        split = [0.6, 0.2, 0.2]
        split_sizes = [int(n_examples * frac) for frac in split]

        g = torch.Generator()
        g.manual_seed(seed)
        permutation = torch.randperm(n_examples, generator=g)

        permuted_inputs = torch.stack(inputs)[permutation]
        self.train_inputs, self.val_inputs, self.test_inputs = torch.split(
            permuted_inputs, split_sizes
        )
        permuted_labels = torch.stack(labels)[permutation]
        self.train_labels, self.val_labels, self.test_labels = torch.split(
            permuted_labels, split_sizes
        )

    @classmethod
    def from_inputs(cls, inputs: list[T], label_function: Callable[[T], V]):
        return cls([(input, label_function(input)) for input in inputs])

    @classmethod
    def from_natural(
        cls,
        n_examples: int,
        n_range: int,
        input_function: Callable[[int], T],
        label_function: Callable[[T], V],
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        example_indices = rng.choice(
            n_range, size=n_examples, replace=False, shuffle=False
        )

        return cls.from_inputs(
            [input_function(i) for i in example_indices], label_function
        )
