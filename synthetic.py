import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.integrate import odeint

def make_var_stationary(beta, radius=0.97):
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta

def causal_structure(p, beta_value, sparsity, seed=0):
    np.random.seed(seed)
    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * beta_value

    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1
    return GC, beta

def linear_ts_data(T, lag, beta, GC, seed=0, sd=0.1, interv=False, anomaly=200, strength=0.1):
    np.random.seed(seed)

    p = np.shape(GC)[0]
    beta = make_var_stationary(beta)

    interv_target = np.random.randint(0, 2, (p, 1))
    interv_matrix = np.tile(interv_target, (1, p)) * GC * strength
    interv_beta = beta + interv_matrix

    # Generate data.
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        if interv and t > anomaly:
            X[:, t] = np.dot(interv_beta, X[:, (t - lag):t].flatten(order='F'))
        else:
            X[:, t] = np.dot(beta, X[:, (t - lag):t].flatten(order='F'))
        X[:, t] += + errors[:, t - 1]

    return X.T[burn_in:], interv_target, interv_beta


def nonlinear_ts_data(T, lag, beta, GC, seed=0, sd=0.1, interv=False, anomaly=200, strength=0.1):
    np.random.seed(seed)

    p = np.shape(GC)[0]
    beta = make_var_stationary(beta)

    interv_target = np.random.randint(0, 2, (p, 1))
    interv_matrix = np.tile(interv_target, (1, p)) * GC * strength
    interv_beta = beta + interv_matrix

    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        if interv and t > anomaly:
            X[:, t] = np.dot(interv_beta, X[:, (t - lag):t].flatten(order='F'))
            alpha = 0.1  # Leaky ReLU parameter
            X[:, t] = np.where(X[:, t] > 0, X[:, t], alpha * X[:, t])
        else:
            X[:, t] = np.dot(beta, X[:, (t - lag):t].flatten(order='F'))
            alpha = 0.1  # Leaky ReLU parameter
            X[:, t] = np.where(X[:, t] > 0, X[:, t], alpha * X[:, t])
        X[:, t] += + errors[:, t - 1]

    return X.T[burn_in:], interv_target, interv_beta

def calculate_metrics(graph1, graph2):
    flat_graph1 = graph1.flatten()
    flat_graph2 = graph2.flatten()


    TP = np.sum((flat_graph1 == 1) & (flat_graph2 == 1))
    FP = np.sum((flat_graph1 == 0) & (flat_graph2 == 1))
    FN = np.sum((flat_graph1 == 1) & (flat_graph2 == 0))
    TN = np.sum((flat_graph1 == 0) & (flat_graph2 == 0))


    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    shd = FP + FN
    return precision, accuracy, recall, f1, shd


def calculate_aucroc(predicted_probabilities, true_labels):
    true_labels_flat = np.array(true_labels).flatten()
    predicted_probabilities_flat = np.array(predicted_probabilities).flatten()

    auc_roc = roc_auc_score(true_labels_flat, predicted_probabilities_flat)
    return auc_roc


def lorenz(x, t, F):
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F

    return dxdt


def simulate_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000,seed=0):

    np.random.seed(seed)

    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC

def calculate_aucpr(y_pred, y_true):
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_pr_value = auc(recall, precision)
    return auc_pr_value