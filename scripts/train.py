from scripts.data import load_data, get_image, get_label, get_size
from scripts.model import MLP
import tqdm
import datetime
import json

from src.gradient import GTensor, CrossEntropy
from src.tensor import Tensor

def log(line: str, run_id: str):
    with open(f"scripts/logs/{run_id}.txt", "a") as f:
        f.write(line + "\n")
    

def main():
    run_id = str(datetime.datetime.now())
    train_imgs, train_labels = load_data("./data", "train")
    batch_size = 64
    lr = 0.005
    epochs = 10
    model = MLP(hidden_sizes=[128], start_size=28*28, end_size=10)
    for epoch in range(epochs):
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
            loss = sum(loss_vect._data._data) / batch_size
            n_correct = 0
            for label, pred in zip(batch_labels, batch_predictions._data.to_list(), strict=True):
                for p in pred:
                    if p > pred[label]:
                        break
                else:
                    n_correct += 1
            with open(f"scripts/logs/{run_id}.txt", "a") as f:
                f.write(f"{loss}-{n_correct/batch_size}\n")
        with open(f"scripts/checkpoints/{run_id}-{epoch}.json", "w") as f:
            f.write(
                json.dumps(
                    {
                        "model_weights": [
                            w._data.to_list() for w in model.weights
                        ]
                    }
                )
            )

main()
