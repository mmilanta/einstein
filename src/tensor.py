from src.utils import nested_len, prod
from itertools import product

class InconsistentSize(Exception):
    pass


class Tensor:
    def __init__(self, data: list[float], dims: tuple[int, ...]):
        self.dims = dims
        self._data = data
        self._validate()

    @property
    def size(self) -> int:
        size = 1
        for d in self.dims:
            size *= d
        return size

    def _validate(self):
        if self.size != len(self._data):
            raise InconsistentSize()

    def get_item(self, key: tuple[int, ...]):
        if len(key) != len(self.dims):
            raise ValueError(f"This {len(self.dims)}-dimensional tensor was indexed with a {len(key)}-dimensional key.")
        idx = 0
        d_prod = 1
        for k, d in zip(key[::-1], self.dims[::-1]):
            idx += d_prod*k
            d_prod *= d
        return self._data[idx]

    @classmethod
    def from_list(cls, list) -> Tensor:
        data: list[float] = flatten_list_of_list(list)
        dims = nested_len(list)
        x = tuple(dims)
        return Tensor(data, tuple(dims))


    @classmethod
    def einsum(cls, einsum_command: str, *args: Tensor, dims: dict[str, int] | None = None) -> Tensor:
        inputs_str, output = einsum_command.split("->")
        inputs = inputs_str.split(",")
        assert len(inputs) == len(args), f"input segments sizes must match number, found {len(inputs)}, {len(args)}"
        # check lens
        # print(f"einsum {einsum_command} {[arg.dims for arg in args]}")
        letter_to_vector_axis: dict[str, list[tuple[int, int]]] = {}
        for i, (input, arg) in enumerate(zip(inputs, args)):
            assert len(input) == len(arg.dims), f"the {i}-th vector in the input has {len(input)} dimensions, but the einsum string assumes it has {len(arg.dims)}. {einsum_command} {[a.dims for a in args]}"
            for j, k in enumerate(input):
                letter_to_vector_axis.setdefault(k, []).append((i, j))
        for j, k in enumerate(output):
            letter_to_vector_axis.setdefault(k, []).append((-1, j)) # we reserve -1 fro outputs


        # check dims matching
        letter_to_dim_size: dict[str, int] = dims if dims is not None else {}
        for k, vals in letter_to_vector_axis.items():
            state_dims = [args[v[0]].dims[v[1]] for v in vals if v[0] >= 0]
            assert len(set(state_dims)) == 1, f"The letter {k} correspond to axis of different dimensions, namely {dims}."
            if k in letter_to_dim_size:
                assert letter_to_dim_size[k] == state_dims[0], f"The letter {k} correspond to dimensions NA, but the dimension {letter_to_dim_size[k]} was passed as parameter."
            letter_to_dim_size[k] = state_dims[0]

        # check that the output is always connected
        for k, vals in letter_to_vector_axis.items():
            if [v for v in vals if v[0] == -1]:
                assert len(vals) > 1, f"The letter {k} was found only in the output. No references in the inputs."

        output_dims = [letter_to_dim_size[k] for k in output]
        remaining_letters = output
        disappearing_letters = list(set("".join(inputs)).symmetric_difference(set(output)))
        output_size = prod(output_dims)
        output_data = [0.0 for _ in range(output_size)]
        overall_index = 0
        for output_keys in product(* [range(dim) for dim in output_dims]): # this could happen in parallel
            o = 0.0
            letter_to_index = {
                letter: val for letter, val in zip(remaining_letters, output_keys)
            }
            for sum_keys in product(* [range(letter_to_dim_size[letter]) for letter in disappearing_letters]):
                letter_to_index.update(
                    {
                        letter: val for letter, val in zip(disappearing_letters, sum_keys)
                    }
                )
                t = 1.0
                for arg, input in zip(args, inputs):
                    key = tuple([letter_to_index[k] for k in input])
                    t *= arg.get_item(key)
                o += t
            output_data[overall_index] = o
            overall_index += 1
        return Tensor(data=output_data, dims=tuple(output_dims))

    @classmethod
    def add(cls, *args: Tensor) -> Tensor:
        for a in args:
            assert a.dims == args[0].dims, f"Found: {[a.dims for a in args]}"
        data = [
            sum(a._data[i] for a in args)
            for i in range(args[0].size)
        ]
        return Tensor(data=data, dims=args[0].dims)

    @classmethod
    def time_constant(cls, arg: Tensor, constant: float) -> Tensor:
        return Tensor(data=[x * constant for x in arg._data], dims=arg.dims)

def flatten_list_of_list(list: list | float) -> list[float]:
    if isinstance(list, float):
        return [list]
    return sum([flatten_list_of_list(l) for l in list], [])
