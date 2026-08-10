from scripts.data import load_data, get_image, get_label, get_size
from scripts.model import MLP
import tqdm
import datetime
import json
import random


from src.gradient import GTensor, CrossEntropy
from src.tensor import Tensor

def log(line: str, run_id: str):
    with open(f"scripts/logs/{run_id}.txt", "a") as f:
        f.write(line + "\n")
    

def main():
    run_id = str(datetime.datetime.now())
    random.seed(42)
    train_imgs, train_labels = load_data("./data", "train")
    idxs = list(range(get_size(train_imgs)))
    random.shuffle(idxs)
    batch_size = 16
    lr = 0.001
    epochs = 10
    model = MLP(hidden_sizes=[256], start_size=28*28, end_size=10)
    n_steps_checkpoint = 100
    for epoch in range(epochs):
        n_batches = get_size(train_imgs) // batch_size
        for step in tqdm.tqdm(range(n_batches), total=n_batches):
            batch_images = [
                get_image(train_imgs, idxs[(step * batch_size) + i]) for i in range(batch_size)
            ]
            batch_labels = [
                get_label(train_labels, idxs[(step * batch_size) + i]) for i in range(batch_size)
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
            for label, pred, losses in zip(batch_labels, batch_predictions._data.to_list(), loss_vect._data.to_list(), strict=True):
                this_correct = False
                for i, p in enumerate(pred):
                    if p >= pred[label] and i != label:
                        break
                else:
                    this_correct = True
                    n_correct += 1
                
            #     print("---")
            #     print(f"   label {label}: {"V" if this_correct else "X"}")
            #     print(f"   logits [{', '.join([f'{l:.4f}' for l in pred])}]")
            #     print(f"   losses [{', '.join([f'{l:.4f}' for l in losses])}]")
            # print(f"LOSS: {loss}")
            with open(f"scripts/logs/{run_id}.txt", "a") as f:
                f.write(f"{loss}-{n_correct/batch_size}\n")

            if step % n_steps_checkpoint == 0:
                with open(f"scripts/checkpoints/{run_id}-{epoch}-{step}.json", "w") as f:
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
