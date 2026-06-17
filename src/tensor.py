from src.utils import nested_len, prod

class InconsistentSize(Exception):
    pass


class Tensor:
    def __init__(self, data: list[float], dims: tuple[int]):
        self.dims = dims
        self._data = data

    @property
    def size(self) -> int:
        size = 1
        for d in self.dims:
            size *= d
        return size

    def _validate(self):
        if self.size != len(self._data):
            raise InconsistentSize()

    def get_item(self, key: tuple[int]):
        if len(key) != len(self.dims):
            raise ValueError(f"This {len(self.dims)}-dimensional tensor was indexed with a {len(key)}-dimensional key.")
        idx = 0
        d_prod = 1
        for k, d in zip(key[::-1], self.dims[::-1]):
            idx += d_prod*k
            d_prod *= d
        return self._data[idx]

    @classmethod
    def from_list(cls, list: list) -> Tensor:
        data: list[float] = flatten_list_of_list(list)
        dims = nested_len(list)

        return Tensor(data, dims)


    @classmethod
    def einsum(cls, einsum_command: str, *args: Tensor) -> Tensor:
        inputs_str, output = einsum_command.split("->")
        inputs = inputs_str.split(",")
        assert len(inputs) == len(args), "input segments sizes must match number of args."
        # check lens
        letter_to_vector_axis: dict[str, list[tuple[int, int]]] = {}
        for i, (input, arg) in enumerate(zip(inputs, args)):
            assert len(input) == len(arg.dims), f"the {i}-th vector in the input has {len(input)} dimensions, but the einesum string assumes it has {len(arg.dims)}."
            for j, k in enumerate(input):
                letter_to_vector_axis.setdefault(k, []).append((i, j))
        for j, k in enumerate(output):
            letter_to_vector_axis.setdefault(k, []).append((-1, j)) # we reserve -1 fro outputs

        letter_to_dim_size: dict[str, int] = {}

        # check dims matching
        for k, vals in letter_to_vector_axis.items():
            dims = [args[v[0]].dims[v[1]] for v in vals if v[0] >= 0]
            assert len(set(dims)) == 1, f"The letter {k} correspond to axis of different dimensions, namely {dims}."
            letter_to_dim_size[k] = next(dims)

        # check that the output is always connected
        for k, vals in letter_to_vector_axis.items():
            if next([v for v in vals if v[0] == -1]):
                assert len(vals) > 1, f"The letter {k} was found only in the output. No references in the inputs."

        output_dims = [letter_to_dim_size[k] for k in output]
        output_size = prod(output_dims)
        output_data = [0 for _ in range(output_size)]
        for i in range(output_size): # this could happen in parallel
            output_data[i] 


def flatten_list_of_list(list: list | float) -> list[float]:
    if isinstance(list, float):
        return [list]
    return sum([flatten_list_of_list(l) for l in list], [])


def rollup