
These files are too big to be included in Git. After `git clone`, one can use following code to recreate them.


```python
import os
if not os.path.exists('data'): os.mkdir('data')
```

# torchvision

```python
import torchvision
torchvision.datasets.MNIST('data', download = True)
torchvision.datasets.CIFAR10('data/cifar-10', download = True)
```

# torchtext

```python
import torchtext
torchtext.datasets.IMDB.splits(torchtext.data.Field(), torchtext.data.Field(), root = "data")
torchtext.datasets.WikiText2.splits(torchtext.data.Field(), root = 'data')
torchtext.vocab.GloVe(name = '6B', dim = 300, cache = "data/vocab")
```

# Dogs vs. Cats

- download data from [kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)
- extract to `data/dogs-vs-cats/`
- `train/` contains 25000 labled images, whereas `test1` contains 12500 unlabled images
- I split `train/` by 4:1 into `data/` using following code

```python
import os, shutil, glob, numpy as np
np.random.seed(0)
path = "data/dogs-vs-cats"
files = glob.glob(os.path.join(path, "train", "*.jpg"))
assert len(files) != 0, "Aborted! Can't find any images under `" + path + "`"

def mk_dir(path):
    """make directory if not exist"""
    if not os.path.exists(path): os.mkdir(path)    

shutil.rmtree(os.path.join(path, "data"), ignore_errors = True)
os.mkdir(os.path.join(path, "data"))
for data_set in ["train", "test"]:
    os.mkdir(os.path.join(path, "data", data_set))
    for animal in ["dog", "cat"]:
        os.mkdir(os.path.join(path, "data", data_set, animal))

shuffle = np.random.permutation(len(files))
test_size = int(len(files) * 0.2)    # arg test ratio
for phase, file_indexs in {"test": shuffle[:test_size], "train": shuffle[test_size:]}.items():
    for i in file_indexs:
        animal = files[i].split('/')[-1].split('.')[0]
        image = files[i].split('/')[-1]
        src = os.path.join("..", "..", "..", "train", image)
        dest = os.path.join(path, "data", phase, animal, image)
        if not os.path.exists(dest): os.symlink(src, dest)
```
