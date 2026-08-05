from src.gradient import GTensor
from src.tensor import Tensor
from src.utils import random_list_tensor
import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvxyz"
def rt(tensor: Tensor | GTensor, n_digits: int = 4, inline: bool = True):
    data = tensor._data if isinstance(tensor, Tensor) else tensor._data._data
    data_str = [f"{d:.4f}" for d in data]
    return f"[{', '.join(data_str)}]" if len(data_str) > 1 else data_str[0]


def test_basic():
    n_dims = 2
    n_epochs = 100
    x = GTensor(Tensor.from_list(random_list_tensor(dims=(n_dims,))))
    y = GTensor(Tensor.from_list([1.0, 1.0]))

    lr = 0.1
    for epoch in range(n_epochs):
        v = GTensor.einsum("i,i->", x, y)
        v.backward(Tensor([1.0], dims=()))
        x._data = Tensor.add(x._data, Tensor.time_constant(x._grad, -lr))

        x._data = Tensor.add(x._data, Tensor.einsum("i,->i", x._grad, Tensor(data=[-lr], dims=())))
        print(f"LOSS: {v._data._data}")
        x.reset()
    assert sum(abs(x_) for x_ in x._data._data) < 1e-4, "Gradient descent should converge to 0"

def test_multiply_constant():
    x = GTensor(Tensor(data=[1.0], dims=()))
    y = GTensor.time_constant(x, 2)
    y.backward(Tensor([1.0], dims=()))
    print(rt(x))
    print(rt(x._grad))
    print(rt(y))
    print(rt(y._grad))

def test_grad_mat_vect():
    A = GTensor(Tensor(data=[1.0, 2.0, 3.0, 4.0], dims=(2, 2)))
    x1 = GTensor(Tensor(data=[5.0, 6.0], dims=(2,)))
    x2 = GTensor(Tensor(data=[7.0, 8.0], dims=(2,)))
    Ax = GTensor.einsum("ij,j->i", A, x1)
    y = GTensor.einsum("j,j->", Ax, x2)
    y.backward(Tensor([1.0], dims=()))
    assert A._grad._data == [35.0, 42.0, 40.0, 48.0], "Wrong gradient"
    assert x1._grad._data == [31.0, 46.0], "Wrong gradient"
    assert x2._grad._data == [17.0, 39.0], "Wrong gradient"
    assert y._data._data == [431.0], "Wrong forward"

def test_linear_system_via_gradien():
    n_dims = 7
    n_epochs = 100000
    b_np = np.random.random(size=(n_dims))
    Z_np = np.random.random(size=(n_dims, n_dims))
    A_np = Z_np @ Z_np.T + (0.1 * np.eye(n_dims))

    #A_np = np.array([[2.0]])
    #b_np = np.array([1.0])


    b = GTensor(Tensor.from_list(b_np.tolist()))
    A = GTensor(Tensor.from_list(A_np.tolist()))

    x_np = np.linalg.inv(A_np) @ b_np
    print(f"b_np: {b_np}")
    print(f"A_np: {A_np}")
    print(f"x_np: {x_np}")

    x = GTensor(Tensor(data=[0.0 for _ in range(n_dims)], dims=(n_dims,)))

    lr = 0.01
    for epoch in range(n_epochs):
        v0 = GTensor.einsum("i,->i", x, GTensor(data=Tensor(data=[1.0], dims=())))
        v01 = GTensor.einsum("i,->i", x, GTensor(data=Tensor(data=[1.0], dims=())))
        v1 = GTensor.einsum("ij,j->i", A, v0)
        v2 = GTensor.einsum("i,i->", v0, v1)
        v3 = GTensor.time_constant(v2, 0.5)
        v4 = GTensor.einsum("i,i->", b, v01)
        v5 = GTensor.time_constant(v4, -1.0)
        v6 = GTensor.add(v3, v5)
        #print(f"RUINNIG BACKWARD")
        x_cur_np = np.array(x._data._data)
        # print(f"x: {id(x)}")
        # print(f"v0: {id(v0)} {rt(v0)} ∇{rt(v0._grad)}")
        # print(f"v01: {id(v01)} {rt(v01)} ∇{rt(v01._grad)}")
        # print(f"v1: {id(v1)} {rt(v1)} ∇{rt(v1._grad)}")
        # print(f"v2: {id(v2)} {rt(v2)} ∇{rt(v2._grad)}")
        # print(f"v3: {id(v3)} {rt(v3)} ∇{rt(v3._grad)}")
        # print(f"v4: {id(v4)} {rt(v4)} ∇{rt(v4._grad)}")
        # print(f"v5: {id(v5)} {rt(v5)} ∇{rt(v5._grad)}")
        # print(f"v6: {id(v6)} {rt(v6)} ∇{rt(v6._grad)}")
        # print(f"v4focus: {v4._partial_grad_operator} {[rt(t) for t in v4._partial_grad]} {[id(t) for t in v4._references]} count: {v4._forward_count}")
        # print(f"X: {rt(x)}-{rt(x._grad)} v0: {rt(v0)}-{rt(v0._grad)} v01: {rt(v01)}-{rt(v01._grad)} v1: {rt(v1)}-{rt(v1._grad)} v2: {rt(v2)}-{rt(v2._grad)} v3: {rt(v3)}-{rt(v3._grad)} v4: {rt(v4)}-{rt(v4._grad)} v5: {rt(v5)}-{rt(v5._grad)} v5: {rt(v5)}-{rt(v5._grad)} v6: {rt(v6)}-{rt(v6._grad)}")
        x_np_pred = np.array(x._data._data)
        distance = float(np.sum(x_np_pred - x_np) ** 2)
        print(f"LOSS: {rt(v6)}. DISTANCE: {distance:.4f} ground_truth LOSS: {float(.5 * x_np.T @ A_np @ x_np - x_np.dot(b_np)):.4f}]")
        if distance < 1e-4:
            break
        v6.backward(Tensor([1.0], dims=()))

        x._data = Tensor.add(x._data, Tensor.time_constant(x._grad, -lr))
        
        x_np_pred = np.array(x._data._data)
        A.reset()
        b.reset()
        x.reset()

        #print(f"X: {rt(x)}-{rt(x._grad)} v0: {rt(v0)}-{rt(v0._grad)} v01: {rt(v01)}-{rt(v01._grad)} v1: {rt(v1)}-{rt(v1._grad)} v2: {rt(v2)}-{rt(v2._grad)} v3: {rt(v3)}-{rt(v3._grad)} v4: {rt(v4)}-{rt(v4._grad)} v5: {rt(v5)}-{rt(v5._grad)} v5: {rt(v5)}-{rt(v5._grad)} v6: {rt(v6)}-{rt(v6._grad)}")
    else:
        raise AssertionError("Method did not converge")
