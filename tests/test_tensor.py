import pytest
from src.tensor import Tensor
from src.utils import nested_get, nested_len, random_list_tensor
from itertools import product
import random


@pytest.mark.parametrize("data", [[[1.0, 2.0], [3.0, 4.0]]])
def test_tensor_basic(data: list):
    tensor = Tensor.from_list(data)
    for p in [(0,0), (0,1), (1,0), (1,1)]:
        v = nested_get(data, p)
        assert isinstance(v, float)
        assert v == tensor.get_item(p)


@pytest.mark.parametrize("iteration", range(100))
def test_tensor_auto(iteration: int):
    n_dims = random.randint(1, 4)
    dims = [random.randint(1, 4) for _ in range(n_dims)]
    data = random_list_tensor(dims)
    tensor = Tensor.from_list(data)
    for p in product(*[range(d) for d in dims]):
        v = nested_get(data, p)
        assert isinstance(v, float)
        assert v == tensor.get_item(p)