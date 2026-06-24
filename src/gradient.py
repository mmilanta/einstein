from tensor import Tensor

class GTensor:
    def __init__(self, data: Tensor):
        self._data = data
        self._grad: list[Tensor] | None = None
        self._grad_ref: list[GTensor] | None = None

    @classmethod
    def einsum(cls, einsum_command: str, *args: GTensor) -> GTensor:
        

    @classmethod
    def add(cls, *args: GTensor) -> GTensor:
        res = Tensor.add([arg._data for arg in args])
        grad = Tensor([1 for _ in range(args[0]._data.size)])
        out = GTensor(
            data=res,
        )
        out._grad=[grad for _ in range(len(args))]
        out._grad_ref=list(args)

