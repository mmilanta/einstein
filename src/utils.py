
import random
import copy


def nested_get(data, keys):
    n_data = copy.deepcopy(data)
    for k in keys:
        n_data = n_data[k]
    return n_data


def nested_len(data: list | float) -> list[int]:
    if isinstance(data, float):
        return []
    return [len(data)] + nested_len(data[0])


def random_list_tensor(dims: tuple[int, ...]):
    if len(dims) == 0:
        return random.random()
    return [random_list_tensor(dims[1:]) for _ in range(dims[0])]


def prod(data: list[int]) -> int:
    out = 1
    for d in data:
        out *= d
    return out
