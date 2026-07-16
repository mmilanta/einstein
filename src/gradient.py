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
        assert len(args) == 2, "Only support 2 vector einsum for gradient flow"
        output_tensor = GTensor(Tensor.einsum(einsum_command, args))
        input_command, output_command = einsum_command.split("->")
        input_commands = input_command.split(",")
        output_tensor._partial_grad = [args[0]._data, args[1]._data]
        output_tensor._partial_grad_operator = [
            f"{input_commands[1]},{output_command}->{input_commands[0]}",
            f"{input_commands[0]},{output_command}->{input_commands[1]}",
        ]
        output_tensor._references=list(args)

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


def ReLU(input: GTensor) -> GTensor:
    output = GTensor(
        data=Tensor(
            data=[max(x, 0) for x in input._data._data],
            dims=input._data.dims
        )
    )
    output._partial_grad = [
        Tensor(
            data=[1 if input._data._data[i] > 0 else 0 for i in range(input._data.size)],
            dims=input._data.dims
        )
    ]
    x = ALPHABET[:len(input._data.dims)]
    output._partial_grad_operator=[f"{x},{x}->{x}"]
    output._references=[input]
