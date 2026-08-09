import pytest

from experiments.run_alfworld_sharded import partition


def test_partition_is_balanced_and_disjoint() -> None:
    assert partition(10, 4) == [(0, 3), (3, 3), (6, 2), (8, 2)]
    assert partition(3, 10) == [(0, 1), (1, 1), (2, 1)]


def test_partition_validates_arguments() -> None:
    with pytest.raises(ValueError):
        partition(0, 1)
