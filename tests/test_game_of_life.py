from typing import Literal
import numpy as np
import pytest

from cellular_automata import GameOfLife


@pytest.mark.parametrize(
    "initial_state, index, neighbours",
    [
        (
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            (0, 0),
            [0, 0, 1, 0, 1, 1, 1, 1],
        ),
        (
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            (1, 1),
            [0, 1, 0, 1, 1, 0, 1, 0],
        ),
        (
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            (2, 2),
            [1, 1, 1, 1, 0, 1, 0, 0],
        ),
    ],
)
def test_neighbours(
    initial_state: list[list[Literal[0, 1]]],
    index: tuple[int, int],
    neighbours: list[Literal[0, 1]],
):
    game = GameOfLife(initial_state)
    assert (game.neighbours(*index) == np.array(neighbours)).all()


@pytest.mark.parametrize(
    "initial_state, expected_state",
    [
        (
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ],
        ),
        (
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
        ),
        (
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ],
        ),
    ],
)
def test_update(
    initial_state: list[list[Literal[0, 1]]],
    expected_state: list[list[Literal[0, 1]]],
):
    game = GameOfLife(initial_state)
    game.update()
    assert (game.state == np.array(expected_state)).all()
