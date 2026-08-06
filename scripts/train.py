from scripts.data import load_data, get_image, get_label, get_size
from scripts.model import MLP
import tqdm

from src.gradient import GTensor, CrossEntropy
from src.tensor import Tensor

def main():
    train_imgs, train_labels = load_data("./data", "train")
    batch_size = 16
    lr = 0.001
    epochs = 10
    model = MLP(hidden_sizes=[128], start_size=28*28, end_size=10)
    pbr = tqdm.tqdm()
    for epoch in range(epochs):
        print(f"epoch {epoch}")
        n_batches = get_size(train_imgs) // batch_size
        for batch_indx in tqdm.tqdm(range(n_batches), total=n_batches):
            batch_images = [
                get_image(train_imgs, (batch_indx * batch_size) + i) for i in range(batch_size)
            ]
            batch_labels = [
                get_label(train_labels, (batch_indx * batch_size) + i) for i in range(batch_size)
            ]
            inputs = GTensor(data=Tensor.from_list(batch_images))

            batch_predictions = model.forward(inputs)
            binary_labels = Tensor.from_list([
                [
                    1.0 if j == val else 0.0 for j in range(10)
                ] for val in batch_labels
            ])
            loss_vect = CrossEntropy(batch_predictions, binary_labels)
            loss_vect.backward(grad=Tensor(data=[1.0 for _ in range(len(loss_vect._data._data))], dims=loss_vect._data.dims))
            model.apply_gradient(-lr)
            model.reset()
            print(f"LOSS: {sum(loss_vect._data._data)}")

main()
