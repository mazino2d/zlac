from utils import *

if __name__ == "__main__":
    LIST_POOL_SIZE = [(3, 4), (2, 4), (2, 2), (2, 2), (2, 2), (2, 2)]
    model = gen_model(LIST_POOL_SIZE, rate_dropout=0.05, is_plot_mode=True)
