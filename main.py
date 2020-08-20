from utils import *

if __name__ == "__main__":
    # # Create data
    # X, y = load_dataset()

    # X_train, X_val, X_test, y_train, y_val, y_test = data_split(X, y, 0.6)

    # inpt0_train = np.expand_dims(X_train[:, 0, :], axis=1)
    # inpt1_train = np.expand_dims(X_train[:, 1, :], axis=1)

    # inpt0_val = np.expand_dims(X_val[:, 0, :], axis=1)
    # inpt1_val = np.expand_dims(X_val[:, 1, :], axis=1)

    # Create model
    LIST_POOL_SIZE = [(3, 4), (2, 4), (2, 2), (2, 2), (2, 2), (2, 2)]
    model = gen_model(LIST_POOL_SIZE, rate_dropout=0.05, is_plot_mode=True)

    # # Start train
    # history = model.fit(
    #     x=[inpt0_train, inpt1_train], y=y_train, batch_size=16, 
    #     epochs=2, validation_data=([inpt0_val, inpt1_val], y_val),
    # )

    # model.save_weights('weights/')


