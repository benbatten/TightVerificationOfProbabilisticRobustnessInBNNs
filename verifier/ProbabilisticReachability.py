import numpy as np
import pickle

import tensorflow as tf

import keras

from pathos.pools import ProcessPool
import multiprocessing

from utils import interval_matrix_product
from utils import relu
from utils import bayes_interval_matrix_product
from utils import interval_conv_prop
from utils import get_conv_coords
from utils import get_conv_coord_from_index

from scipy.special import erf
import math



def tf_model_builder(params):
    input_shape = (params[0][0].shape[0],)
    layers = [keras.layers.Flatten(input_shape=input_shape)]
    for i, param in enumerate(params):
        size = param[1].shape[0]
        layers.append(keras.layers.Dense(size, activation='relu'))

    model = keras.Sequential([layer for layer in layers])
    model.build()
    for i, param in enumerate(params):
        model.layers[i + 1].set_weights(param)
    return model


def compute_erf_prob(intervals, mean, stddev):
    prob = 0.0
    for interval in intervals:
        # val1 = erf((mean - interval[0]) / (math.sqrt(2) * (stddev)))
        # val2 = erf((mean - interval[1]) / (math.sqrt(2) * (stddev)))
        # prob += 0.5 * (val1 - val2)  # Check why halved
        val1 = (1 + erf((interval[1] - mean)/(math.sqrt(2) * stddev)))/ 2
        val2 = (1 + erf((interval[0] - mean)/(math.sqrt(2) * stddev))) / 2
        assert val1 >= val2, "Issue with ERF probability"
        prob += val1 - val2
    # if(prob < 0.99):
    #    print intervals
    #    print mean
    #    print stddev
    return prob


'''Compute gradient of deterministic NN'''
def compute_gradient(input, param_samps, logit, target_logit, mod_type, reverse=False):
    tf_model = tf_model_builder(param_samps)
    if mod_type == 'cnn':
        logit = logit[0]
    x = tf.constant(input)
    with tf.GradientTape() as tape:
        tape.watch(tf_model.trainable_variables)
        output = tf_model(x)
        loss_output = output[0][logit] - output[0][target_logit]

    param_grads = tape.gradient(loss_output, tf_model.trainable_variables)

    if reverse:
        grads = []
        assert len(param_grads) % 2 == 0, "Odd number of gradient arrays."
        param_len = int(len(param_grads) / 2)
        for i in range(param_len):
            grads.append((np.sign(-param_grads[2 * i].numpy()), np.sign(-param_grads[2 * i + 1].numpy())))
    else:
        grads = []
        assert len(param_grads) % 2 == 0, "Odd number of gradient arrays."
        param_len = int(len(param_grads) / 2)
        for i in range(param_len):
            grads.append([(param_grads[2 * i].numpy()) == 0.0, (param_grads[2 * i + 1].numpy()) == 0.0])

    return grads


def bayesian_interval_bound_pass(inp, params1, params2):
    # assert 0.0 <= np.min(inp) and np.max(inp) <= 1.0, "Input incorrectly scaled."
    layer_sizes = [param[0].shape[1] for param in params1]
    layer_bounds = {}
    layer_bounds[0] = inp
    for i, layer_size in enumerate(layer_sizes):
        mid_output = bayes_interval_matrix_product(inp, params1[i][0], params2[i][0])
        for j in range(layer_size):
            candidates = (mid_output[j, 0] + params1[i][1][j],
                          mid_output[j, 0] + params2[i][1][j],
                          mid_output[j, 1] + params1[i][1][j],
                          mid_output[j, 1] + params2[i][1][j])
            mid_output[j, 0] = min(candidates)
            mid_output[j, 1] = max(candidates)
        if i != len(layer_sizes) - 1:
            mid_output = relu(mid_output)
            layer_bounds[i + 1] = mid_output
        else:
            mid_output = mid_output
            layer_bounds[i + 1] = mid_output
        inp = mid_output
    return layer_bounds


def interval_bound_pass(inp, params):
    assert 0.0 <= np.min(inp) and np.max(inp) <= 1.0, "Input incorrectly scaled."
    layer_sizes = [param[0].shape[1] for param in params]
    layer_bounds = {}
    layer_bounds[0] = inp
    for i, layer_size in enumerate(layer_sizes):
        mid_output = interval_matrix_product(params[i][0], inp)
        for j in range(layer_size):
            mid_output[j, :] = mid_output[j, :] + params[i][1][j]
        for j in range(layer_size):
            mid_output[j, :] = relu(mid_output[j, :])
        layer_bounds[i + 1] = mid_output
        inp = mid_output
    return layer_bounds


def conv_interval_bounds(input_lb, input_ub, kernels, strides):
    layer_activations = [[np.zeros((input_lb.shape)) != 0, np.zeros((input_lb.shape)) != 0]]
    layer_bounds = [[input_lb, input_ub]]
    for kern, stride in zip(kernels, strides):
        output_lb, output_ub = interval_conv_prop(input_lb, input_ub, kern[1], kern[0], stride)
        layer_activations.append([output_ub <= 0.,
                                  output_ub > 0.])  # First array gives true for values dead, second for values active.
        output_lb = relu(output_lb)
        output_ub = relu(output_ub)
        layer_bounds.append([output_lb, output_ub])
        input_lb = output_lb
        input_ub = output_ub
    return layer_activations, layer_bounds



'''Below is implementation for BNN LBP from Wicker et al.'''
def get_alphas_betas(zeta_l, zeta_u, activation="relu"):
    alpha_L, alpha_U = list([]), list([])
    beta_L, beta_U = list([]), list([])
    for i in range(len(zeta_l)):
        if (zeta_u[i] <= 0):
            alpha_U.append(0)
            alpha_L.append(0)
            beta_L.append(0)
            beta_U.append(0)
        elif (zeta_l[i] >= 0):
            alpha_U.append(1)
            alpha_L.append(1)
            beta_L.append(0)
            beta_U.append(0)
        else:
            # For relu I have the points (zeta_l, 0) and (zeta_u, zeta_u)
            a_U = zeta_u[i] / (zeta_u[i] - zeta_l[i])
            b_U = -1 * (a_U * zeta_l[i])

            # a_L = a_U ; b_L = 0
            if (zeta_u[i] + zeta_l[i]) >= 0:
               a_L = 1 ;   b_L = 0
            else:
                a_L = 0
                b_L = 0
            alpha_U.append(a_U)
            alpha_L.append(a_L)
            beta_L.append(b_L)
            beta_U.append(b_U)
    return alpha_U, beta_U, alpha_L, beta_L


def get_bar_lower(linear_bound_coef, mu_l, mu_u,
                  nu_l, nu_u, lam_l, lam_u):
    mu_l = np.squeeze(mu_l)
    mu_u = np.squeeze(mu_u)
    mu_bar, nu_bar, lam_bar = [], [], []

    nu_bar = nu_l

    # coef of the form - alpha_U, beta_U, alpha_L, beta_L
    for i in range(len(linear_bound_coef)):
        if (linear_bound_coef[i, 2] >= 0):
            mu_bar.append(linear_bound_coef[i, 2] * mu_l[i])
            for k in range(len(nu_bar)):
                try:
                    nu_bar[k][i] = linear_bound_coef[i, 2] * np.asarray(nu_l[k][i])
                except:
                    print("Error")
            lam_bar.append(linear_bound_coef[i, 2] * lam_l[i] + linear_bound_coef[i, 3])
        else:
            mu_bar.append(linear_bound_coef[i, 2] * mu_u[i])
            for k in range(len(nu_bar)):
                nu_bar[k][i] = linear_bound_coef[i, 2] * nu_u[k][i]
            lam_bar.append(linear_bound_coef[i, 2] * lam_u[i] + linear_bound_coef[i, 3])
    return np.asarray(mu_bar), nu_bar, np.asarray(lam_bar)


def get_bar_upper(linear_bound_coef, mu_l, mu_u,
                  nu_l, nu_u, lam_l, lam_u):
    mu_l = np.squeeze(mu_l)
    mu_u = np.squeeze(mu_u)
    mu_bar, nu_bar, lam_bar = [], [], []
    nu_bar = nu_u
    for i in range(len(linear_bound_coef)):
        if (linear_bound_coef[i, 0] >= 0):
            mu_bar.append(linear_bound_coef[i, 0] * mu_u[i])
            for k in range(len(nu_bar)):
                nu_bar[k][i] = linear_bound_coef[i, 0] * np.asarray(nu_u[k][i])
            lam_bar.append(linear_bound_coef[i, 0] * lam_u[i] + linear_bound_coef[i, 1])
        else:
            mu_bar.append(linear_bound_coef[i, 0] * mu_l[i])
            for k in range(len(nu_bar)):
                nu_bar[k][i] = linear_bound_coef[i, 0] * nu_l[k][i]
            lam_bar.append(linear_bound_coef[i, 0] * lam_l[i] + linear_bound_coef[i, 1])
    return np.asarray(mu_bar), nu_bar, np.asarray(lam_bar)


def get_abc_lower(w, mu_l_bar, nu_l_bar, la_l_bar,
                  mu_u_bar, nu_u_bar, la_u_bar):
    a, b, c = [], [], []
    for i in range(len(w)):
        curr_a = []
        # curr_b = []
        curr_c = []
        for j in range(len(w[i])):
            if (w[i][j] >= 0):
                curr_a.append(w[i][j] * mu_l_bar[i])
                curr_c.append(w[i][j] * la_l_bar[i])
            else:
                curr_a.append(w[i][j] * mu_u_bar[i])
                curr_c.append(w[i][j] * la_u_bar[i])
        a.append(curr_a)

        c.append(curr_c)
    for k in range(len(nu_l_bar)):
        curr_b = []
        # for i in range(len(w)):
        for j in range(len(w[i])):
            curr_curr_b = []
            # for j in range(len(w[i])):
            for i in range(len(w)):
                if (w[i][j] >= 0):
                    curr_curr_b.append(w[i][j] * nu_l_bar[k][i])
                else:
                    curr_curr_b.append(w[i][j] * nu_u_bar[k][i])
            curr_b.append(curr_curr_b)
        b.append(curr_b)

    return np.asarray(a), b, np.asarray(c)


def get_abc_upper(w, mu_l_bar, nu_l_bar, la_l_bar,
                  mu_u_bar, nu_u_bar, la_u_bar):
    # This is anarchy
    return get_abc_lower(w, mu_u_bar, nu_u_bar, la_u_bar,
                         mu_l_bar, nu_l_bar, la_l_bar)


def min_of_linear_fun(coef_vec, uppers, lowers):
    # getting the minimum
    val_min = 0
    for i in range(len(coef_vec)):
        if coef_vec[i] >= 0:
            val_min = val_min + coef_vec[i] * lowers[i]
        else:
            val_min = val_min + coef_vec[i] * uppers[i]
    return val_min


def max_of_linear_fun(coef_vec, uppers, lowers):
    val_max = - min_of_linear_fun(-coef_vec, uppers, lowers)
    return val_max


def propogate_lines(in_reg, risk_params, safe_params, init_layer):
    x_l, x_u = in_reg[:, 0], in_reg[:, 1]


    widths = [r[1].shape[0] for r in risk_params[init_layer:]]

    n_hidden_layers = len(widths) - 1


    # Code adaptation end. From now on it's the standard code

    # Actual code from now on

    # Step 1: Inputn layers -> Pre-activation function
    # W_0_L, W_0_U, b_0_L, b_0_U = (sWs[0][0] - dWs[0] * w_margin, sWs[0][0] + dWs[0] * w_margin,
    #                               sbs[0][0] - dbs[0] * w_margin, sbs[0][0] + dbs[0] * w_margin)

    W_0_L, W_0_U, b_0_L, b_0_U = risk_params[init_layer][0], safe_params[init_layer][0], risk_params[init_layer][1], safe_params[init_layer][1]

    W_0_L = W_0_L.T
    W_0_U = W_0_U.T

    mu_0_L = W_0_L
    mu_0_U = W_0_U

    n_hidden_1 = W_0_L.shape[0]

    nu_0_L = np.asarray([x_l for i in range(n_hidden_1)])
    nu_0_U = np.asarray([x_l for i in range(n_hidden_1)])
    la_0_L = - np.dot(x_l, W_0_L.T) + b_0_L
    la_0_U = - np.dot(x_l, W_0_U.T) + b_0_U

    # getting bounds on pre-activation fucntion
    zeta_0_L = [(min_of_linear_fun(np.concatenate((mu_0_L[i].flatten(), nu_0_L[i].flatten())),
                                   np.concatenate((np.asarray(x_u).flatten(), W_0_U[i].flatten())),
                                   np.concatenate((np.asarray(x_l).flatten(), W_0_L[i].flatten())))) for i in
                range(n_hidden_1)]

    zeta_0_L = np.asarray(zeta_0_L) + la_0_L

    zeta_0_U = [(max_of_linear_fun(np.concatenate((mu_0_U[i].flatten(), nu_0_U[i].flatten())),
                                   np.concatenate((np.asarray(x_u).flatten(), W_0_U[i].flatten())),
                                   np.concatenate((np.asarray(x_l).flatten(), W_0_L[i].flatten())))) for i in
                range(n_hidden_1)]

    zeta_0_U = np.asarray(zeta_0_U) + la_0_U

    # Initialising variable for main loop
    curr_zeta_L = zeta_0_L
    curr_zeta_U = zeta_0_U
    curr_mu_L = mu_0_L
    curr_mu_U = mu_0_U
    curr_nu_L = [nu_0_L]
    curr_nu_U = [nu_0_U]
    curr_la_L = la_0_L
    curr_la_U = la_0_U

    W_Ls = W_0_L.flatten()
    W_Us = W_0_U.flatten()
    # loop over the hidden layers
    for l in range(1, n_hidden_layers + 1):
        if l < n_hidden_layers:
            curr_n_hidden = widths[l]
        else:
            curr_n_hidden = 10

        LUB = np.asarray(get_alphas_betas(curr_zeta_L, curr_zeta_U))
        LUB = np.asmatrix(LUB).transpose()
        # Now evaluate eq (*) conditions:
        curr_mu_L_bar, curr_nu_L_bar, curr_la_L_bar = get_bar_lower(LUB, curr_mu_L, curr_mu_U,
                                                                    curr_nu_L, curr_nu_U,
                                                                    curr_la_L, curr_la_U)

        curr_mu_U_bar, curr_nu_U_bar, curr_la_U_bar = get_bar_upper(LUB, curr_mu_L, curr_mu_U,
                                                                    curr_nu_L, curr_nu_U,
                                                                    curr_la_L, curr_la_U)

        curr_z_L = [min_of_linear_fun([LUB[i, 2]], [curr_zeta_U[i]], [curr_zeta_L[i]]) + LUB[i, 3]
                    for i in range(len(curr_zeta_U))]

        # SUpper and lower bounds for weights and biases of current hidden layer
        # curr_W_L, curr_W_U, curr_b_L, curr_b_U = (sWs[l][0] - dWs[l] * w_margin, sWs[l][0] + dWs[l] * w_margin,
        #                                           sbs[l][0] - dbs[l] * w_margin, sbs[l][0] + dbs[l] * w_margin)

        curr_W_L, curr_W_U, curr_b_L, curr_b_U = risk_params[l][0], safe_params[l][0], risk_params[l][1], safe_params[l][1]

        a_L, b_L, c_L = get_abc_lower(curr_W_L, curr_mu_L_bar, curr_nu_L_bar, curr_la_L_bar,
                                      curr_mu_U_bar, curr_nu_U_bar, curr_la_U_bar)

        a_U, b_U, c_U = get_abc_upper(curr_W_U, curr_mu_L_bar, curr_nu_L_bar, curr_la_L_bar,
                                      curr_mu_U_bar, curr_nu_U_bar, curr_la_U_bar)

        curr_mu_L = np.sum(a_L, axis=0)
        curr_mu_U = np.sum(a_U, axis=0)
        curr_nu_L = []
        curr_nu_U = []
        for k in range(l - 1):
            curr_nu_L.append(np.sum(b_L[k], axis=1))
            curr_nu_U.append(np.sum(b_U[k], axis=1))

        curr_nu_L.append(b_L[l - 1])
        curr_nu_U.append(b_U[l - 1])

        curr_nu_L.append(np.asarray([curr_z_L for i in range(curr_n_hidden)]))
        curr_nu_U.append(np.asarray([curr_z_L for i in range(curr_n_hidden)]))

        curr_la_L = np.sum(c_L, axis=0) - np.dot(curr_z_L, curr_W_L) + curr_b_L
        curr_la_U = np.sum(c_U, axis=0) - np.dot(curr_z_L, curr_W_U) + curr_b_U

        curr_zeta_L = []
        curr_zeta_U = []

        for i in range(curr_n_hidden):
            ith_mu_L = curr_mu_L[i]
            ith_mu_U = curr_mu_U[i]

            ith_W_Ls = np.concatenate((W_Ls, curr_W_L.T[i]))
            ith_W_Us = np.concatenate((W_Us, curr_W_U.T[i]))
            ith_nu_L = []
            ith_nu_U = []
            for k in range(len(curr_nu_L)):
                ith_nu_L = np.concatenate((ith_nu_L, np.asarray(curr_nu_L[k][i]).flatten()))
                ith_nu_U = np.concatenate((ith_nu_U, np.asarray(curr_nu_U[k][i]).flatten()))

            curr_zeta_L.append(min_of_linear_fun(np.concatenate((ith_mu_L, ith_nu_L)),
                                                 np.concatenate((x_u, ith_W_Us)),
                                                 np.concatenate((x_l, ith_W_Ls))
                                                 ))

            curr_zeta_U.append(max_of_linear_fun(np.concatenate((ith_mu_U, ith_nu_U)),
                                                 np.concatenate((x_u, ith_W_Us)),
                                                 np.concatenate((x_l, ith_W_Ls))
                                                 ))
        curr_zeta_L = curr_zeta_L + curr_la_L
        curr_zeta_U = curr_zeta_U + curr_la_U

        W_Ls = np.concatenate((W_Ls, curr_W_L.T.flatten()))
        W_Us = np.concatenate((W_Us, curr_W_U.T.flatten()))
    return [curr_zeta_L, curr_zeta_U]
