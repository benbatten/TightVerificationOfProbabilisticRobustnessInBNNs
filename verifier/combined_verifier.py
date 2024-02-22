import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dynamic_grad_ratio", default=0.2)
parser.add_argument("--net_name", default='wick_MNIST_1_128')
parser.add_argument("--epsilon", default=0.025)
parser.add_argument("--im_number", default=50)
parser.add_argument("--grad_stepsize", default=0.5)
parser.add_argument("--implementation", default='ours')
parser.add_argument("--samp_num", default=50)

args = parser.parse_args()
dynamic_grad_ratio = float(args.dynamic_grad_ratio) + 1
epsilon = float(args.epsilon)
im_number = int(args.im_number)
grad_stepsize = float(args.grad_stepsize)
implementation = args.implementation
net_name = args.net_name
sample_num = int(args.samp_num)

import copy
import sys
import time
import os
import pickle
import math
import itertools
import numpy as np
from ProbabilisticReachability import compute_gradient
from ProbabilisticReachability import bayesian_interval_bound_pass
from ProbabilisticReachability import interval_bound_pass
from ProbabilisticReachability import conv_interval_bounds
from ProbabilisticReachability import compute_erf_prob
from ProbabilisticReachability import propogate_lines
from utils import single_conv_layer
from utils import relu
from utils import bound_checker
from utils import list_sort
from utils import interval_conv_prop
from utils import single_erf
from utils import merger
from utils import overlap_interval
from pathos.pools import ProcessPool
import multiprocessing
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100
import keras
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

use_lin = True

if net_name.startswith('CIFAR'):
    mod_type = 'cnn'
    dset = 'CIFAR10'
    layers = 2
else:
    mod_type = 'mlp'
    dset = 'MNIST'
    if net_name.startswith('wick_MNIST_1'):
        layers = 1
    else:
        layers = 2


def main(image):
    '''***1: import the data and define input set bounds.***'''
    assert dset in ['MNIST', 'CIFAR10', 'CIFAR100'], "INVALID DATASET"
    if dset == 'MNIST':
        with open('xs.h5', 'rb') as f:
            x_test = pickle.load(f)
        with open('ys.h5', 'rb') as f:
            y_test = pickle.load(f)
        x = x_test[image].flatten().astype('float64')
    elif dset == 'CIFAR10':
        (X_train, Y_train), (x_test, y_test) = cifar10.load_data()
        x = x_test[image].astype('float64') / 255.
    elif dset == 'CIFAR100':
        (X_train, Y_train), (x_test, y_test) = cifar100.load_data()
        x = x_test[image].flatten().astype('float64') / 255.

    def propagate_conv(input, kernels, strides, padding='SAME'):
        if not input.shape == x_test[0].shape:
            input = np.reshape(input, x_test[0].shape)
        for i, kernel in enumerate(kernels):
            args = [input, kernel[0], kernel[1],
                    strides[i], padding]
            input = single_conv_layer(args)
        return input

    def load_model():
        global strides
        path = "Networks/{0}/{1}.h5".format(dset, net_name)
        with open("Networks/{0}/{1}.h5".format(dset, net_name), 'rb') as f:  # Loads list [params, std_devs]
            net = pickle.load(f)
        net_params = {}
        if mod_type == 'cnn':
            net_params['kernels'] = net[0]
            params = net[1]
            std_devs = net[2]
            strides = net[3]
        else:
            params = net[0]
            std_devs = net[1]
        for i in range(len(net[1])):
            net_params["mW_{}".format(i)] = params[i][0]
            net_params["mb_{}".format(i)] = params[i][1]
            net_params["dW_{}".format(i)] = std_devs[i][0]
            net_params["db_{}".format(i)] = std_devs[i][1]
        return net_params

    '''Return dictionary of parameter samples from posterior dist.'''

    def sample_net(samp_num, net_params):
        samples = {}
        if mod_type == 'cnn':
            lay_num = int((len(net_params) - 1) / 4)
        else:
            lay_num = layers
        for i in range(layers + 1):
            means = ['mW_{}'.format(i), 'mb_{}'.format(i)]
            std_devs = ['dW_{}'.format(i), 'db_{}'.format(i)]
            mat_shape = net_params[means[0]].shape
            bias_shape = net_params[means[1]].shape
            samples['sW_{}'.format(i)] = np.random.normal(net_params[means[0]], net_params[std_devs[0]],
                                                          (samp_num, mat_shape[0], mat_shape[1]))
            samples['sb_{}'.format(i)] = np.random.normal(net_params[means[1]], net_params[std_devs[1]],
                                                          (samp_num, bias_shape[0]))
        return samples

    '''Propagates a single input through the layers of an MLP'''

    def propagate_MLP(input, samples):
        samp_num = list(samples.values())[0].shape[0]
        for i in range(layers + 1):
            tmp_input = 0
            for s in range(samp_num):
                tmp_input += np.matmul(input, samples['sW_{}'.format(i)][s]) + samples['sb_{}'.format(i)][s]
            input = tmp_input / samp_num
            if i < layers:
                input = relu(input)
        return input

    true_class = y_test[image]

    net_params = load_model()
    if mod_type == 'cnn':
        std_devs = [[net_params['dW_{}'.format(i)], net_params['db_{}'.format(i)]] for i in
                    range(int((len(net_params) - 1) / 4))]
        means = [[net_params['mW_{}'.format(i)], net_params['mb_{}'.format(i)]] for i in
                 range(int((len(net_params) - 1) / 4))]
    else:
        std_devs = [[net_params['dW_{}'.format(i)], net_params['db_{}'.format(i)]] for i in
                    range(layers + 1)]
        means = [[net_params['mW_{}'.format(i)], net_params['mb_{}'.format(i)]] for i in
                 range(layers + 1)]

    input_lb = np.clip(x - epsilon, 0, 1)
    input_ub = np.clip(x + epsilon, 0, 1)

    if mod_type == 'cnn':
        int_states, int_bounds = conv_interval_bounds(input_lb, input_ub, net_params['kernels'], strides)

        states = int_states[-1]
        bounds = int_bounds[-1]
        inp = np.array([list(bounds[0][0, 0]), list(bounds[1][0, 0])]).T
    else:
        inp = np.array([input_lb.flatten(), input_ub.flatten()]).T

    s = sample_net(50, net_params)
    if mod_type == 'cnn':
        our_prediction = propagate_conv(x, net_params['kernels'], strides).flatten()
        class_out = np.argmax(propagate_MLP(our_prediction, s))
        if class_out != y_test[image]:
            return -1
    else:
        class_out = np.argmax(propagate_MLP((x).flatten(), s))
        if class_out != y_test[image]:
            return -1

    '''Define function for our method (both VIE and GIE)'''
    def main_verifier_ours(args):
        l, risk_param_gradients, safe_param_gradients, samples, std_devs = args  # samples should be list of lists [[weight_1, bias_1], [etc]]

        risk_halt = False
        safe_halt = False

        risk_params = copy.deepcopy(samples)
        safe_params = copy.deepcopy(samples)

        for layer in range(len(risk_param_gradients)):
            risk_param_gradients[layer][0] = -1 * (
                        (risk_param_gradients[layer][0].astype('float') * (dynamic_grad_ratio - 1)) + 1)
            risk_param_gradients[layer][1] = -1 * (
                        (risk_param_gradients[layer][1].astype('float') * (dynamic_grad_ratio - 1)) + 1)
            safe_param_gradients[layer][0] = (safe_param_gradients[layer][0].astype('float') * (
                        dynamic_grad_ratio - 1)) + 1
            safe_param_gradients[layer][1] = (safe_param_gradients[layer][1].astype('float') * (
                        dynamic_grad_ratio - 1)) + 1

        halter = False
        index = 0
        iter_counter = 0
        while not halter:
            iter_counter += 1

            if not risk_halt:
                store_risk = copy.deepcopy(risk_params)
                for i, param in enumerate(risk_params):
                    risk_params[i] = [param[0] + (std_devs[i][0] * risk_param_gradients[i][0] * grad_stepsize),
                                      param[1] + (std_devs[i][1] * risk_param_gradients[i][1] * grad_stepsize)]

            if not safe_halt:
                store_safe = copy.deepcopy(safe_params)
                for i, param in enumerate(safe_params):
                    safe_params[i] = [param[0] + (std_devs[i][0] * safe_param_gradients[i][0] * grad_stepsize),
                                      param[1] + (std_devs[i][1] * safe_param_gradients[i][1] * grad_stepsize)]
            index += 1
            if not use_lin:
                bounds = bayesian_interval_bound_pass(inp, risk_params, safe_params)
                SAT, threat_class = bound_checker(bounds[list(bounds)[-1]], true_class)
            else:
                if layers == 2:
                    low, high = propogate_lines(inp, risk_params, safe_params, 0)
                    bounds = [np.array([low, high]).T]
                else:
                    low, high = propogate_lines(inp, risk_params, safe_params, 0)
                    bounds = [np.array([low, high]).T]
                assert np.min(bounds[0][:, 0] <= bounds[0][:, 1]), "Issue with LBP"
                SAT, threat_class = bound_checker(bounds[-1], true_class)
            if not SAT and index == 1:
                return
            if not SAT or index == sample_num:
                safe_params = store_safe
                risk_params = store_risk
                return [risk_params, safe_params, iter_counter]  # , index]

    '''Define main algorithm for wicker method'''
    def main_verifier_wick(args):
        samples, std_devs = args

        risk_halt = False
        safe_halt = False

        risk_params = copy.deepcopy(samples)
        safe_params = copy.deepcopy(samples)

        index = 0
        while True:

            if not risk_halt:
                for i, param in enumerate(risk_params):
                    risk_params[i] = [param[0] - (std_devs[i][0] * grad_stepsize),
                                      param[1] - (std_devs[i][1] * grad_stepsize)]

            if not safe_halt:
                for i, param in enumerate(safe_params):
                    safe_params[i] = [param[0] + (std_devs[i][0] * grad_stepsize),
                                      param[1] + (std_devs[i][1] * grad_stepsize)]
            index += 1
            if not use_lin:
                bounds = bayesian_interval_bound_pass(inp, risk_params, safe_params)
                SAT, threat_class = bound_checker(bounds[list(bounds)[-1]], true_class)
            else:
                if layers == 2:
                    low, high = propogate_lines(inp, risk_params, safe_params, 0)
                    bounds = [np.array([low, high]).T]
                else:
                    low, high = propogate_lines(inp, risk_params, safe_params, 0)
                    bounds = [np.array([low, high]).T]
                assert np.min(bounds[0][:, 0] <= bounds[0][:, 1]), "Issue with LBP"
                SAT, threat_class = bound_checker(bounds[-1], true_class)

            if SAT:
                return [risk_params, safe_params]
            else:
                return

    '''Compute p_safe from list of valid orthotopes'''
    def probability_compute(valid_intervals, params, comp_cap):
        valid_intervals = list(filter(None, valid_intervals))
        if len(valid_intervals) == 0:
            return -2
        if comp_cap:
            mean_int = 0
            for interval in valid_intervals:
                mean_int += interval[-1]
            mean_int /= len(valid_intervals)
            valid_intervals = valid_intervals[:int(np.floor(sample_num / mean_int))]

        index_combinations = list(itertools.combinations(range(len(valid_intervals)), 2))

        for indexes in index_combinations:
            non_overlap = False
            h_rect_1 = valid_intervals[indexes[0]]
            h_rect_2 = valid_intervals[indexes[1]]
            for layer in range(len(h_rect_1[0])):
                for i in range(h_rect_1[0][layer][0].shape[0]):
                    for j in range(h_rect_1[0][layer][0].shape[1]):
                        interval_1 = [h_rect_1[0][layer][0][i, j], h_rect_1[1][layer][0][i, j]]
                        interval_2 = [h_rect_2[0][layer][0][i, j], h_rect_2[1][layer][0][i, j]]
                        overlap = max(0, min(interval_1[1], interval_2[1]) - max(interval_1[0], interval_2[0]))
                        if overlap == 0:
                            non_overlap = True
                        elif layer == 0 and i == 0 and j == 0:
                            min_gap = [overlap, layer, i, j]
                        elif overlap < min_gap[0]:
                            min_gap = [overlap, layer, i, j]

                        if non_overlap:
                            break
                    if non_overlap:
                        break
                if non_overlap:
                    break
            for layer in range(len(h_rect_1[0])):
                for i in range(h_rect_1[0][layer][1].shape[0]):
                    interval_1 = [h_rect_1[0][layer][1][i], h_rect_1[1][layer][1][i]]
                    interval_2 = [h_rect_2[0][layer][1][i], h_rect_2[1][layer][1][i]]
                    overlap = max(0, min(interval_1[1], interval_2[1]) - max(interval_1[0], interval_2[0]))
                    if non_overlap:
                        break
                    elif overlap == 0:
                        non_overlap = True
                    elif overlap < min_gap[0]:
                        min_gap = [overlap, layer, i, None]
                    # if non_overlap:
                    #     break
                if non_overlap:
                    break
            if not non_overlap:
                if indexes[0] == 0:
                    '''interval 1 is the sample around mean - represents highest probability density'''
                    if min_gap[-1] != None:
                        interval_1 = [h_rect_1[0][min_gap[1]][0][min_gap[2], min_gap[3]],
                                      h_rect_1[1][min_gap[1]][0][min_gap[2], min_gap[3]]]
                        interval_2 = [h_rect_2[0][min_gap[1]][0][min_gap[2], min_gap[3]],
                                      h_rect_2[1][min_gap[1]][0][min_gap[2], min_gap[3]]]
                        if interval_2[0] < interval_1[0]:
                            interval_2[1] = interval_1[0]
                        elif interval_2[1] > interval_1[1]:
                            interval_2[0] = interval_1[1]
                        else:
                            interval_2[0] = interval_1[1];
                            interval_2[1] = interval_1[1]
                        valid_intervals[indexes[0]][0][min_gap[1]][0][min_gap[2], min_gap[3]] = \
                            interval_1[0]
                        valid_intervals[indexes[0]][1][min_gap[1]][0][min_gap[2], min_gap[3]] = \
                            interval_1[1]
                        valid_intervals[indexes[1]][0][min_gap[1]][0][min_gap[2], min_gap[3]] = \
                            interval_2[0]
                        valid_intervals[indexes[1]][1][min_gap[1]][0][min_gap[2], min_gap[3]] = \
                            interval_2[1]
                    else:
                        '''interval 1 is the sample around mean - represents highest probability density'''
                        interval_1 = [h_rect_1[0][min_gap[1]][1][min_gap[2]],
                                      h_rect_1[1][min_gap[1]][1][min_gap[2]]]
                        interval_2 = [h_rect_2[0][min_gap[1]][1][min_gap[2]],
                                      h_rect_2[1][min_gap[1]][1][min_gap[2]]]
                        if interval_2[0] < interval_1[0]:
                            interval_2[1] = interval_1[0]
                        elif interval_2[1] > interval_1[1]:
                            interval_2[0] = interval_1[1]
                        else:
                            interval_2[0] = interval_1[1]
                            interval_2[1] = interval_1[1]
                        valid_intervals[indexes[0]][0][min_gap[1]][1][min_gap[2]] = \
                            interval_1[0]
                        valid_intervals[indexes[0]][1][min_gap[1]][1][min_gap[2]] = \
                            interval_1[1]
                        valid_intervals[indexes[1]][0][min_gap[1]][1][min_gap[2]] = \
                            interval_2[0]
                        valid_intervals[indexes[1]][1][min_gap[1]][1][min_gap[2]] = \
                            interval_2[1]
                else:
                    if min_gap[-1] != None:
                        '''Min is in weights'''
                        interval_1 = [h_rect_1[0][min_gap[1]][0][min_gap[2], min_gap[3]],
                                      h_rect_1[1][min_gap[1]][0][min_gap[2], min_gap[3]]]
                        interval_2 = [h_rect_2[0][min_gap[1]][0][min_gap[2], min_gap[3]],
                                      h_rect_2[1][min_gap[1]][0][min_gap[2], min_gap[3]]]
                        sorted_list, interval_index = (list(t) for t in zip(*sorted(
                            zip([interval_1, interval_2], [1, 2]))))
                        sorted_list = merger(sorted_list)
                        valid_intervals[indexes[0]][0][min_gap[1]][0][min_gap[2], min_gap[3]] = \
                        sorted_list[np.argmin(interval_index)][0]
                        valid_intervals[indexes[0]][1][min_gap[1]][0][min_gap[2], min_gap[3]] = \
                        sorted_list[np.argmin(interval_index)][1]
                        valid_intervals[indexes[1]][0][min_gap[1]][0][min_gap[2], min_gap[3]] = \
                        sorted_list[np.argmax(interval_index)][0]
                        valid_intervals[indexes[1]][1][min_gap[1]][0][min_gap[2], min_gap[3]] = \
                        sorted_list[np.argmax(interval_index)][1]
                    else:
                        '''Min is in bias'''
                        interval_1 = [h_rect_1[0][min_gap[1]][1][min_gap[2]],
                                      h_rect_1[1][min_gap[1]][1][min_gap[2]]]
                        interval_2 = [h_rect_2[0][min_gap[1]][1][min_gap[2]],
                                      h_rect_2[1][min_gap[1]][1][min_gap[2]]]
                        sorted_list, interval_index = (list(t) for t in zip(*sorted(
                            zip([interval_1, interval_2], [1, 2]))))
                        sorted_list = merger(sorted_list)
                        valid_intervals[indexes[0]][0][min_gap[1]][1][min_gap[2]] = \
                            sorted_list[np.argmin(interval_index)][0]
                        valid_intervals[indexes[0]][1][min_gap[1]][1][min_gap[2]] = \
                            sorted_list[np.argmin(interval_index)][1]
                        valid_intervals[indexes[1]][0][min_gap[1]][1][min_gap[2]] = \
                            sorted_list[np.argmax(interval_index)][0]
                        valid_intervals[indexes[1]][1][min_gap[1]][1][min_gap[2]] = \
                            sorted_list[np.argmax(interval_index)][1]

        p_sum = 0.0
        for interval in range(len(valid_intervals)):
            p_interval = 0.0
            zero_log = False
            for lay_num in range(len(valid_intervals[0][0])):
                for i in range(valid_intervals[0][0][lay_num][0].shape[0]):
                    for j in range(valid_intervals[0][0][lay_num][0].shape[1]):
                        tmp_interval = [valid_intervals[interval][0][lay_num][0][i][j],
                                        valid_intervals[interval][1][lay_num][0][i][j]]
                        tmp_prob = single_erf(tmp_interval, params[lay_num][0][i, j], std_devs[lay_num][0][i, j])
                        if tmp_prob != 0.0:
                            p_interval += math.log(tmp_prob)
                        else:
                            zero_log = True

                for k in range(valid_intervals[0][0][lay_num][1].shape[0]):
                    tmp_interval = [valid_intervals[interval][0][lay_num][1][k],
                                    valid_intervals[interval][1][lay_num][1][k]]
                    tmp_prob = single_erf(tmp_interval, params[lay_num][1][k], std_devs[lay_num][1][k])
                    if tmp_prob != 0.0:
                        p_interval += math.log(tmp_prob)
                    else:
                        zero_log = True
            if not zero_log:
                p_sum += math.exp(p_interval)

        return p_sum


    samps = []
    sample_list = sample_net(sample_num, net_params)
    '''Main loop for wicker method'''
    if implementation == 'wicker':
        dict_names = sample_list.keys()
        for b in range(sample_num):
            if b == 0:
                single_sample = [[means[i][0], means[i][1]] for i in range(len(means))]
            else:
                single_sample = [[sample_list['sW_{}'.format(i)][b], sample_list['sb_{}'.format(i)][b]] for i in
                                 range(int(len(sample_list) / 2))]
            samples = {}
            for name in dict_names:
                samples[name] = sample_list[name][b:b + 1]
            if not mod_type == 'cnn':
                y = propagate_MLP(x, samples)
                mid_x = x
            else:
                mid_x = propagate_conv(x, net_params['kernels'], strides).flatten()
                y = propagate_MLP(mid_x, samples)
            if not np.argmax(y) == true_class:
                continue
            y[true_class] = np.min(y)
            target_class = np.argmax(y)

            samps.append([single_sample, std_devs])
            # valid_intervals.append(main_verifier_wick(samps[b]))

        pool = ProcessPool(nodes=multiprocessing.cpu_count() - 1)
        valid_intervals = pool.map(main_verifier_wick, samps)
        p = probability_compute(valid_intervals, means, False)

    '''Main loop for our methods'''
    if implementation == 'ours':
        dict_names = sample_list.keys()
        for b in range(sample_num):
            if b == 0:
                single_sample = [[means[i][0], means[i][1]] for i in range(len(means))]
            else:
                single_sample = [[sample_list['sW_{}'.format(i)][b], sample_list['sb_{}'.format(i)][b]] for i in
                                 range(int(len(sample_list) / 2))]
            samples = {}
            for name in dict_names:
                samples[name] = sample_list[name][b:b + 1]
            if not mod_type == 'cnn':
                y = propagate_MLP(x, samples)
                mid_x = x
            else:
                mid_x = propagate_conv(x, net_params['kernels'], strides).flatten()
                y = propagate_MLP(mid_x, samples)
            if not np.argmax(y) == true_class:
                continue
            y[true_class] = np.min(y)
            target_class = np.argmax(y)
            if mod_type == 'cnn':
                temp_size = 512
            else:
                temp_size = 784
            safe_param_gradients = compute_gradient(np.reshape(mid_x, (-1, temp_size)), single_sample, true_class,
                                                    target_class, mod_type)

            samps.append([b, copy.deepcopy(safe_param_gradients), safe_param_gradients, single_sample, std_devs]),
            # args = [b, risk_param_gradients, safe_param_gradients, single_sample, std_devs]
            # valid_intervals.append(main_verifier_ours(args))

        pool = ProcessPool(nodes=multiprocessing.cpu_count() - 1)
        valid_intervals = []
        breaker = False
        for g in range(10):
            mid_valid_intervals = pool.map(main_verifier_ours, samps[int(np.ceil(g * (sample_num / 10))):int(
                np.ceil(g * (sample_num / 10) + sample_num / 10))])
            valid_intervals += mid_valid_intervals
            sum = 0
            for interval in valid_intervals:
                if interval != None:
                    sum += interval[-1]
            if sum > sample_num:
                break

        del samps

        p = probability_compute(valid_intervals, means, True)

    return p


import time

if __name__ == '__main__':
    probs = []
    for i in range(im_number):
        print("\nbeginning image {}".format(i))
        start = time.time()
        ans = main(i)
        res = {
            'image': i,
            'p_safe_lb': ans,
            'time_taken': time.time() - start,
            'sample_num': sample_num,
            'grad_stepsize': grad_stepsize
        }
        probs.append(res)

    with open("results_net_{0}_dset_{1}.h5".format(net_name, dset), 'wb') as f:
        pickle.dump(probs, f)
