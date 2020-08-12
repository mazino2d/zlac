## Lossless audio checker model


### Install dependencies

```bash
pip install -r requirements.txt
# or
pip install pandas numpy librosa tensorflow
pip install git+https://github.com/keunwoochoi/kapre.git
```

**Note**: You must use [Python3](https://www.python.org/downloads/)

### How to plot model

- Install [graphviz](https://graphviz.org/): `sudo apt-get install graphviz`

- Example code:

```python
from utils import *

if __name__ == "__main__":
    LIST_POOL_SIZE = [(3, 4), (2, 4), (2, 2), (2, 2), (2, 2), (2, 2)]
    model = gen_model(LIST_POOL_SIZE, rate_dropout=0.05, is_plot_mode=True)
```

### Project infomation

- Author: khoidd @ Zalo Group
- Email: khoidd@vng.com.vn