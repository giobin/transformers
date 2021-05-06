import os, sys
import socket


name = socket.gethostname()
print(f'running on {name}')

def create_dummy_data(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    contents = {"source": "What is love ?", "target": "life"}
    n_lines = {"train": 12, "val": 2, "test": 2}
    for split in ["train", "test", "val"]:
        for field in ["source", "target"]:
            content = "\n".join([contents[field]] * n_lines[split])
            with open(os.path.join(data_dir, f"{split}.{field}"), "w") as f:
                f.write(content)

data_dir = './prova_data/'
create_dummy_data(data_dir)

print(f'created the tmp dir in prova_data')