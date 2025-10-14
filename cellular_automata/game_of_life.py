from typing import Literal
import numpy as np
import numpy.typing as npt
from itertools import product


class GameOfLife:
    def __init__(
        self, initial_state: npt.NDArray[np.int8] | list[list[Literal[0, 1]]]
    ) -> None:
        self.state = np.array(initial_state, dtype=np.int8)

    @classmethod
    def by_natural(cls, natural: int, shape: tuple[int, int]):
        n_rows, n_cols = shape
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError("Invalid shape.")
        n_cells = n_rows * n_cols
        if natural < 0 or natural >= 2**n_cells:
            raise ValueError(
                "natural must be between 0 and 2 ** (n_rows * n_cols) - 1 inclusive."
            )
        binary = bin(natural)[2:]
        code = np.array(list(map(int, reversed(binary))))
        code.resize(shape)
        initial_state = code.reshape(shape)
        return cls(initial_state)

    def neighbours(self, i: int, j: int) -> npt.NDArray[np.int8]:
        n_rows: int
        n_cols: int
        n_rows, n_cols = self.state.shape

        row_indices = np.arange(i - 1, i + 2) % n_rows
        col_indices = np.arange(j - 1, j + 2) % n_cols
        row_indices, col_indices = zip(
            *filter(lambda index: index != (i, j), product(row_indices, col_indices))
        )
        return self.state[row_indices, col_indices].copy()

    def update(self, n_updates: int = 1) -> None:
        for _ in range(n_updates):
            new_state = self.state.copy()
            for (i, j), live in np.ndenumerate(self.state):
                n_live_neighbours = np.sum(self.neighbours(i, j))
                if live and (n_live_neighbours <= 1 or n_live_neighbours >= 4):
                    new_state[i, j] = 0
                elif not live and n_live_neighbours == 3:
                    new_state[i, j] = 1
            self.state = new_state

    def __repr__(self) -> str:
        return "\n".join("".join(row) for row in np.where(self.state == 1, "X", " "))
