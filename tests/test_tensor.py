import pytest
from src.tensor import Tensor
from src.utils import nested_get, nested_len, random_list_tensor
from itertools import product
import random
import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvxyz"


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
    data = random_list_tensor(tuple(dims))
    tensor = Tensor.from_list(data)
    for p in product(*[range(d) for d in dims]):
        v = nested_get(data, p)
        assert isinstance(v, float)
        assert v == tensor.get_item(p)


@pytest.mark.parametrize("iteration", range(100))
def test_einsum(iteration: int):
    n_dims1 = random.randint(0, 4)
    n_dims2 = random.randint(0, n_dims1)
    output_n_dims = random.randint(0, n_dims1 + n_dims2)
    #n_dims1, n_dims2, output_n_dims = 3, 2, 1
    output_letters = [ALPHABET[k] for k in range(output_n_dims)]
    output_str = "".join(output_letters)
    input_letters_not_summed = output_letters + ["|"]
    random.shuffle(input_letters_not_summed)
    input_letters_not_summed_str = "".join(input_letters_not_summed)
    input_letters1 = input_letters_not_summed_str.split("|")[0]
    input_letters2 = input_letters_not_summed_str.split("|")[1]
    if len(input_letters1) > len(input_letters2):
        input_letters1, input_letters2 = input_letters2, input_letters1
    input_letters1 = input_letters1 + "".join([ALPHABET[~k] for k in range(n_dims1 - len(input_letters1))])
    input_letters2 = input_letters2 + "".join([ALPHABET[~k] for k in range(n_dims2 - len(input_letters2))])
    command = f"{input_letters1},{input_letters2}->{output_str}"
    used_letters = set(input_letters1 + input_letters2 + output_str)
    dims = {k: random.randint(1, 4) for k in used_letters}
    
    rl1 = random_list_tensor(tuple([dims[k] for k in input_letters1]))
    rl2 = random_list_tensor(tuple([dims[k] for k in input_letters2]))
    input_tensor_1 = Tensor.from_list(rl1)
    input_tensor_2 = Tensor.from_list(rl2)
    np1 = np.array(rl1)
    np2 = np.array(rl2)
    
    output_tensor = Tensor.einsum(command, input_tensor_1, input_tensor_2)
    output_np = np.einsum(command, np1, np2)
    assert output_tensor.dims == tuple([dims[k] for k in output_letters])
    for np_val, my_val in zip(output_np.flatten().tolist(), output_tensor._data):
        assert abs(np_val - my_val) < 1e-12

@pytest.mark.parametrize("iteration", range(100))
def test_sum(iteration: int):
    n_dims = random.randint(0, 5)
    dims = [random.randint(1, 5) for _ in range(n_dims)]
    rl1 = random_list_tensor(tuple(dims))
    rl2 = random_list_tensor(tuple(dims))
    input_tensor_1 = Tensor.from_list(rl1)
    input_tensor_2 = Tensor.from_list(rl2)
    np1 = np.array(rl1)
    np2 = np.array(rl2)
    output_tensor = Tensor.add(input_tensor_1, input_tensor_2)
    output_np = np1 + np2
    for np_val, my_val in zip(output_np.flatten().tolist(), output_tensor._data):
        assert abs(np_val - my_val) < 1e-12


def test_trace():
    tensor = Tensor([1.0, 5.0, 3.0, 4.0], dims=(2, 2))
    trace = Tensor.einsum("ii->", tensor)
    assert trace.size == 1.0
    assert trace.get_item(()) == 5.0


def test_transpose():
    tensor = Tensor([1.0, 5.0, 3.0, 4.0], dims=(2, 2))
    transpose = Tensor.einsum("ij->ji", tensor)
    assert transpose.size == 4.0
    assert transpose._data == [1.0, 3.0, 5.0, 4.0]
