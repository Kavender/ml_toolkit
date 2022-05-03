from random import shuffle
import numpy as np

# decay rate applied to past velocity
# gamma = .9
# learning rate
# alpha=0.1
# n_iter = 100

## TODO: define get_minibatch_grad

def get_minibatch(X, y, minibatch_size):
    minibatches = []

    X, y = shuffle(X, y)

    for i in range(0, X.shape[0], minibatch_size):
        X_mini = X[i:i + minibatch_size]
        y_mini = y[i:i + minibatch_size]

        minibatches.append((X_mini, y_mini))

    return minibatches


def iter_mini_batch_and_get_gradient(X, y, minibatch_size):
    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        yield get_minibatch_grad(model, X_mini, y_mini)


def sgd(model, X_train, y_train, minibatch_size):

    for grad in iter_mini_batch_and_get_gradient(X_train, y_train, minibatch_size):
        for layer in grad:
            model[layer] += alpha * grad[layer]

    return model


def momentum(model, X_train, y_train, minibatch_size, gamma=0.9, alpha=0.1):
    # to store our momentum for every parameter
    velocity = {k: np.zeros_like(v) for k, v in model.items()}

    for grad in iter_mini_batch_and_get_gradient(X_train, y_train, minibatch_size):
        for layer in grad:
            velocity[layer] = gamma * velocity[layer] + alpha * grad[layer]
            model[layer] += velocity[layer]

    return model


def nesterov(model, X_train, y_train, minibatch_size, gamma=0.9, alpha=0.1):
    velocity = {k: np.zeros_like(v) for k, v in model.items()}

    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        # approximated next state of model params that  is calculated by adding the momentum to the current params
        model_ahead = {k: v + gamma * velocity[k] for k, v in model.items()}
        grad = get_minibatch_grad(model_ahead, X_mini, y_mini)

        for layer in grad:
            velocity[layer] = gamma * velocity[layer] + alpha * grad[layer]
            model[layer] += velocity[layer]

    return model


def adagrad(model, X_train, y_train, minibatch_size, eps=1e-8):
    # adaptive learning rate, use the accumulated sum of squared of all params to normalize the learning rate
    # param used to updated a lot will be slowed down, vs. params with little update will aacelerate
    cache = {k: np.zeros_like(v) for k, v in model.items()}

    for grad in iter_mini_batch_and_get_gradient(X_train, y_train, minibatch_size):
        # parameters update is pointwise operation, hence the learning rate is adaptive per-paramete
        for k in grad:
            cache[k] += grad[k]**2
            # eps: Smoothing to avoid division by zero
            model[k] += alpha * grad[k] / (np.sqrt(cache[k]) + eps)

    return model


def rmsprop(model, X_train, y_train, minibatch_size, gamma=.9):
    cache = {k: np.zeros_like(v) for k, v in model.items()}

    for grad in iter_mini_batch_and_get_gradient(X_train, y_train, minibatch_size):
        for k in grad:
            # instead of considering all of the past gradients, RMSprop behaves like moving average, applied with decay rate gamma
            cache[k] = gamma * cache[k] + (1 - gamma) * (grad[k]**2)
            model[k] += alpha * grad[k] / (np.sqrt(cache[k]) + eps)

    return model


def adam(model, X_train, y_train, minibatch_size, beta1 = .9, beta2 = .999, alpha=1e-3, eps=1e-8):
    # Adam is RMSprop with momentum, M & R is initialized with zero, as biased towards zero until warmed up.
    M = {k: np.zeros_like(v) for k, v in model.items()}
    R = {k: np.zeros_like(v) for k, v in model.items()}


    for grad in iter_mini_batch_and_get_gradient(X_train, y_train, minibatch_size):
        for k in grad:
            M[k] = beta1 * M[k] + (1. - beta1) * grad[k]
            R[k] = beta2 * R[k] + (1. - beta2) * grad[k]**2

            # bias correction mechansim to make convergence faster
            m_k_hat = M[k] / (1. - beta1**(t))
            r_k_hat = R[k] / (1. - beta2**(t))

            model[k] += alpha * m_k_hat / (np.sqrt(r_k_hat) + eps)

    return model
