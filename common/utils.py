from collections.abc import Callable
from typing import TypeVar
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


T = TypeVar("T", bound=torch.Tensor)
V = TypeVar("V", bound=torch.Tensor)


class Data:
    def __init__(
        self,
        data: list[tuple[T, V]],
        *,
        batch_size: int = 1,
        seed: int = 42,
    ) -> None:
        n_examples = len(data)
        inputs, labels = map(torch.stack, zip(*data))

        # standard machine learning train, validation, test split
        split = [0.6, 0.2, 0.2]
        temp_sizes = [int(n_examples * frac) for frac in split[:-1]]
        split_sizes = temp_sizes + [n_examples - sum(temp_sizes)]

        train_inputs, val_inputs, test_inputs = torch.split(inputs, split_sizes)
        train_labels, val_labels, test_labels = torch.split(labels, split_sizes)

        g = torch.Generator()
        g.manual_seed(seed)

        train_dataset = TensorDataset(train_inputs, train_labels)
        train_sampler = RandomSampler(train_dataset, generator=g)
        self.train_loader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=batch_size
        )

        val_dataset = TensorDataset(val_inputs, val_labels)
        val_sampler = RandomSampler(val_dataset, generator=g)
        self.val_loader = DataLoader(
            val_dataset, sampler=val_sampler, batch_size=batch_size
        )

        test_dataset = TensorDataset(test_inputs, test_labels)
        test_sampler = RandomSampler(test_dataset, generator=g)
        self.test_loader = DataLoader(
            test_dataset, sampler=test_sampler, batch_size=batch_size
        )

    @classmethod
    def from_inputs(
        cls,
        inputs: list[T],
        label_function: Callable[[T], V],
        *,
        batch_size: int = 1,
    ):
        return cls(
            [(input, label_function(input)) for input in inputs],
            batch_size=batch_size,
        )

    @classmethod
    def from_natural(
        cls,
        n_examples: int,
        n_range: int,
        input_function: Callable[[int], T],
        label_function: Callable[[T], V],
        *,
        batch_size: int = 1,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        example_indices = rng.choice(
            n_range, size=n_examples, replace=False, shuffle=False
        )

        return cls.from_inputs(
            [input_function(i) for i in example_indices],
            label_function,
            batch_size=batch_size,
        )
