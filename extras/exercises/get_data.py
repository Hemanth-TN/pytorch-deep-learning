
import requests
import zipfile
from pathlib import Path
import os

url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"

data_path = Path("./exercise_data")
zip_file_name = "pizza_steak_sushi.zip"
image_path = data_path / "pizza_steak_sushi"


if not image_path.is_dir():
    image_path.mkdir(parents=True,
                     exist_ok=True)

    with open(data_path/zip_file_name, 'wb') as f:
        r = requests.get(url)
        print(f"Downloading the data zip file")
        f.write(r.content)
else:
    print("data already exists.. Skipping download")

if not os.listdir(image_path):
    with zipfile.ZipFile(data_path/zip_file_name, 'r') as z_ref:
        print(f"extracting the image files")
        z_ref.extractall(path=image_path)
else:
    print('Images are extracted. Skipping extraction')

