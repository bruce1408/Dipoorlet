import os
import numpy as np

file_path = os.path.dirname(os.path.abspath(__file__))

print(file_path)
mock_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
os.makedirs(f"{file_path}/input", exist_ok=True)

mock_data.tofile(f"{file_path}/input/0.bin")
mock_data.tofile(f"{file_path}/input/1.bin")  