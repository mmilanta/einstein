from src.tensor import Tensor

ALPHABET = "abcdefghijklmnopqrstuvxyz"

class GTensor:
    def __init__(self, data: Tensor):
        self._data = data
        self._grad: Tensor = Tensor(data=[0.0 for _ in data._data], dims=data.dims)
        self._partial_grad: list[Tensor] = []
        self._partial_grad_operator: list[str] = []
        self._references: list[GTensor] = []
        self._forward_count: int = 0

    @classmethod
    def einsum(cls, einsum_command: str, *args: GTensor) -> GTensor:
        assert len(args) == 2, "Only support 2 vector einsum for gradient flow"
        output_tensor = GTensor(Tensor.einsum(einsum_command, *[arg._data for arg in args]))
        input_command, output_command = einsum_command.split("->")
        input_commands = input_command.split(",")
        output_tensor._partial_grad = [args[1]._data, args[0]._data]
        output_tensor._partial_grad_operator = [
            f"{output_command},{input_commands[1]}->{input_commands[0]}",
            f"{output_command},{input_commands[0]}->{input_commands[1]}",
        ]
        output_tensor._references=list(args)
        for arg in args:
            arg._forward_count += 1
        return output_tensor

    @classmethod
    def time_constant(cls, tensor: GTensor, constant: float) -> GTensor:
        out = GTensor(data=Tensor.time_constant(tensor._data, constant))
        x = ALPHABET[:len(out._data.dims)]
        out._partial_grad=[Tensor([constant for _ in range(tensor._data.size)], dims=tensor._data.dims)]
        out._partial_grad_operator=[f"{x},{x}->{x}"]
        out._references=[tensor]
        tensor._forward_count += 1
        return out

    @classmethod
    def add(cls, *args: GTensor) -> GTensor:
        res = Tensor.add(*[arg._data for arg in args])
        partial_grad = Tensor([1 for _ in range(args[0]._data.size)], dims=res.dims)
        out = GTensor(
            data=res,
        )
        x = ALPHABET[:len(res.dims)]
        out._partial_grad=[partial_grad for _ in range(len(args))]

        out._partial_grad_operator=[f"{x},{x}->{x}" for _ in range(len(args))]
        out._references=list(args)
        for arg in args:
            arg._forward_count += 1
        return out

    def backward(self: GTensor, grad: Tensor):
        self._grad = Tensor.add(grad, self._grad)
        self._forward_count -= 1
        if self._forward_count > 0:
            return
        assert self._references is not None, "References is None"
        
        for ref, partial_grad, partial_grad_operator in zip(self._references, self._partial_grad, self._partial_grad_operator):
            ref.backward(Tensor.einsum(partial_grad_operator, self._grad, partial_grad))

    def reset(self):
        self._grad = Tensor(data=[0.0 for _ in self._data._data], dims=self._data.dims)
        self._forward_count = 0

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

    input._forward_count += 1
    return output
