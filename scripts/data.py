import gzip
def load_data(data_path, split: str):
    if split == "train":
        img_path = f"{data_path}/train-images-idx3-ubyte.gz"
        lbl_path = f"{data_path}/train-labels-idx1-ubyte.gz"
    elif split == "test":
        raise NotImplementedError()
    else:
        raise ValueError("Only train or test supported")
    
    with gzip.open(img_path,'r') as fin:
        data = fin.read()
    with gzip.open(lbl_path,'r') as fin:
        labels = fin.read()
    return data, labels

def get_label(data: list[bytes], index: int) -> int:
    magic_number = int.from_bytes(data[0:4])
    assert magic_number == 2049
    n_images = int.from_bytes(data[4:8])
    assert index >= 0, f"dataset has index from {0} to {n_images - 1}. {index} given"
    assert index < n_images, f"dataset has index from {0} to {n_images - 1}. {index} given"
    return int(data[8 + index])

def get_image(data: list[bytes], index: int) -> list[list[float]]:
    magic_number = int.from_bytes(data[0:4])
    assert magic_number == 2051, "File is not an image dataset file"
    n_images = int.from_bytes(data[4:8])
    assert index >= 0, f"dataset has index from {0} to {n_images - 1}. {index} given"
    assert index < n_images, f"dataset has index from {0} to {n_images - 1}. {index} given"
    n_rows = int.from_bytes(data[8:12])
    n_cols = int.from_bytes(data[12:16])
    image_size = n_rows * n_cols
    img = data[((index * image_size) + 16):(16 + image_size + (index * image_size))]
    img_float = [int(pixel)/255  for pixel in img]
    return img_float

def get_size(data: list[bytes]):
    return int.from_bytes(data[4:8])