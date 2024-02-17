import numpy as np


def compute_gradient_mse(y, tx, w):
    """Compute the gradient."""
    N = len(y)
    error = y - tx.dot(w)
    gradient = -1 / N * tx.T.dot(error)
    return gradient


def compute_loss_mse(y, tx, w):
    """Calculate the loss using mse."""
    N = len(y)
    squared_error = (y - tx.dot(w)) ** 2
    loss = 1 / (2 * N) * np.sum(squared_error)
    return loss


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Calculate the loss using mse"""
    if max_iters == 0:
        return (initial_w, compute_loss_mse(y, tx, initial_w))
    w = initial_w
    # loss = compute_loss_mse(y, tx, w)
    for iter in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)
        w = w - gamma * gradient
        if iter % 100 == 0:
            print(
                "Current iteration={i}, loss={l}".format(
                    i=iter, l=compute_loss_mse(y, tx, w)
                )
            )
    loss = compute_loss_mse(y, tx, w)
    return (w, loss)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Calculate the loss using mse"""
    if max_iters == 0:
        return (initial_w, compute_loss_mse(y, tx, initial_w))
    w = initial_w
    for iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            gradient = compute_gradient_mse(minibatch_y, minibatch_tx, w)
            w = w - gamma * gradient
        if iter % 100 == 0:
            print(
                "Current iteration={i}, loss={l}".format(
                    i=iter, l=compute_loss_mse(y, tx, w)
                )
            )

    loss = compute_loss_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_=0.5):
    """implement ridge regression."""
    N = tx.shape[0]
    D = tx.shape[1]
    w = np.linalg.solve(tx.T.dot(tx) + lambda_ * 2 * N * np.identity(D), tx.T.dot(y))
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def sigmoid(t):
    """apply sigmoid function on t."""
    t = np.where(t > 500, 500, t)
    t = np.where(t < -500, -500, t)
    return 1.0 / (1.0 + np.exp(-t))


def compute_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    y_pred = sigmoid(np.dot(tx, w))
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def compute_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    gradient = tx.T.dot(pred - y)
    return gradient / len(y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """implement logistic regression.
    Used Gradient Descent as optimization method"""
    if max_iters == 0:
        return (initial_w, compute_logistic_loss(y, tx, initial_w))

    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        gradient = compute_gradient_logistic(y, tx, w)
        w = w - gamma * gradient
        if iter % 100 == 0:
            print(
                "Current iteration={i}, loss={l}".format(
                    i=iter, l=compute_logistic_loss(y, tx, w)
                )
            )
    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """implement regularized logistic regression
    Used Gradient Descent as optimization method"""
    if max_iters == 0:
        return (initial_w, compute_logistic_loss(y, tx, initial_w))

    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        # loss = compute_logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        gradient = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
        if iter % 100 == 0:
            loss = compute_logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
            print(
                "Current iteration={i}, loss={l} (with regularization)".format(
                    i=iter, l=loss
                )
            )
    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def compute_weighted_logistic_loss(y, tx, w, w_vec):
    """compute the cost by negative log likelihood."""
    y_pred = sigmoid(np.dot(tx, w))
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)

    return -np.mean(
        w_vec[0] * y * np.log(y_pred) + w_vec[1] * (1 - y) * np.log(1 - y_pred)
    )


def compute_weighted_gradient_logistic(y, tx, w, w_vec):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    gradient1 = tx.T.dot(-w_vec[0] * y + w_vec[1] * pred)

    gradient2 = (w_vec[0] - w_vec[1]) * (y * pred).T @ tx
    return (gradient1 + gradient2) / len(y)


def weighted_reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iters, gamma, weight
):
    """implement regularized logistic regression
    Used Gradient Descent as optimization method"""
    if max_iters == 0:
        return (initial_w, compute_weighted_logistic_loss(y, tx, initial_w, weight))

    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        # loss = compute_logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        gradient = (
            compute_weighted_gradient_logistic(y, tx, w, weight) + 2 * lambda_ * w
        )
        w = w - gamma * gradient
        if iter % 100 == 0:
            loss = compute_weighted_logistic_loss(
                y, tx, w, weight
            ) + lambda_ * np.squeeze(w.T.dot(w))
            print(
                "Current iteration={i}, loss={l} (with regularization)".format(
                    i=iter, l=loss
                )
            )
    loss = compute_weighted_logistic_loss(y, tx, w, weight)
    return w, loss
