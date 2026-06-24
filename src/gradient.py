from tensor import Tensor

ALPHABET = "abcdefghijklmnopqrstuvxyz"

class GTensor:
    def __init__(self, data: Tensor):
        self._data = data
        self._grad: list[Tensor] | None = None
        self._partial_grad: list[Tensor] | None = None
        self._partial_grad_operator: list[str] | None = None
        self._references: list[GTensor] | None = None

    @classmethod
    def einsum(cls, einsum_command: str, *args: GTensor) -> GTensor:
        pass

    @classmethod
    def add(cls, *args: GTensor) -> GTensor:
        res = Tensor.add([arg._data for arg in args])
        partial_grad = Tensor([1 for _ in range(args[0]._data.size)], dims=res.dims)
        out = GTensor(
            data=res,
        )
        x = ALPHABET[:len(res.dims)]
        out._partial_grad=[partial_grad for _ in range(len(args))]

        out._partial_grad_operator=[f"{x},{x}->{x}" for _ in range(len(args))]
        out._references=list(args)

