
import numpy as np
import copy
from scipy.signal import convolve2d
from scipy.special import erf
from scipy.stats import norm
import math

def relu(x):
    return x * (x > 0)


def overlap_interval(interval_1, interval_2):
    intervals = sorted([interval_1, interval_2])
    if intervals[1][1] < intervals[0][1]:
        '''Higher interval totally within lower interval'''
        return intervals[1]
    elif intervals[1][0] < intervals[0][1]:
        '''Lower bound of upper interval overlaps with upper bound of of lower interval'''
        return [intervals[1][0], intervals[0][1]]
    else:
        return None


def interval_matrix_product(weights, input):
    assert weights.shape[0] == input.shape[0], "Matrix/Vector shape mismatch."
    output = np.zeros((weights.shape[1], 2))
    for i in range(weights.shape[1]):
        min_sum = 0
        max_sum = 0
        for j in range(weights.shape[0]):
            tmp = (weights[j, i] * input[j, 0], weights[j, i] * input[j, 1])
            min_sum += np.min(tmp)
            max_sum += np.max(tmp)
        output[i, :] = (min_sum, max_sum)
    return output


def bayes_interval_matrix_product(input, weights1, weights2):
    assert weights1.shape[0] == input.shape[0], "Matrix/Vector shape mismatch."
    assert weights1.shape == weights2.shape, "Weights 1/2 mismatch."
    output = np.zeros((weights1.shape[1], 2))
    for i in range(weights1.shape[1]):
        min_sum = 0
        max_sum = 0
        for j in range(weights1.shape[0]):
            weight1 = weights1[j, i]
            weight2 = weights2[j, i]
            tmp = (weight1 * input[j, 0], weight2 * input[j, 0],
                   weight1 * input[j, 1], weight2 * input[j, 1])
            min_sum += np.min(tmp)
            max_sum += np.max(tmp)
        output[i, :] = (min_sum, max_sum)
    return output


def bound_checker(out_bounds, true_class):
    output_bounds = copy.deepcopy(out_bounds)
    threat_class = true_class
    output_bounds[true_class, 1] = np.min(output_bounds[:, 1])
    SAT = (output_bounds[true_class, 0] > np.max(output_bounds[:, 1]))
    if not SAT:
        threat_class = np.argmax(output_bounds[:, 1])
    return SAT, threat_class



def single_erf(interval, mean, stddev):
    val1 = (1 + erf((interval[1] - mean) / (math.sqrt(2) * stddev))) / 2
    val2 = (1 + erf((interval[0] - mean) / (math.sqrt(2) * stddev))) / 2
    if val1 < val2:
        print(interval, mean, stddev)
    assert val1 + 0.00000001 >= val2, "Issue with ERF probability"
    assert val1 - val2 >= 0., "Issue with ERF"
    return val1 - val2

def merger(sorted_list):
    for el in range(1, len(sorted_list)):
        if sorted_list[el][0] < sorted_list[el-1][1]:
            sorted_list[el][0] = sorted_list[el-1][1]
        if sorted_list[el][1] < sorted_list[el][0]:
            sorted_list[el][1] = sorted_list[el][0]
    return sorted_list


def list_sort(intervals):  # intervals: list of lists of safe weight intervals
    sorted_intervals = []
    intervals = sorted(intervals)
    sorted_intervals.append(intervals[0])
    for interval in intervals[1:]:
        if interval[0] > sorted_intervals[-1][1]:
            sorted_intervals.append(interval)
        elif interval[1] < sorted_intervals[-1][1]:
            continue
        else:
            sorted_intervals[-1][1] = interval[1]
    return sorted_intervals


'''
Performs convolution for each filter within the kernel.
'''
def single_conv(filter, input, out_dim, stride):
    output = np.zeros((out_dim[0], out_dim[1]))
    filt_dims = (filter.shape[0], filter.shape[1])
    assert len(input.shape) == len(filter.shape), "Input and filter shape do not match"
    for l in range(out_dim[1]):
        for i in range(out_dim[0]):
            start_ind = (i * stride, l * stride)
            output[i, l] = np.sum(np.multiply(
                input[start_ind[0]:start_ind[0] + filt_dims[0], start_ind[1]:start_ind[1] + filt_dims[1], :],
                filter))
    return output


'''
Performs convolution for an entire layer.
input: the input to the layer
filter: the layer's entire filter kernel
'''
def single_conv_layer(args):
    input, filter, bias, stride, padding = args
    if padding == 'SAME':
        out_dim = (int(input.shape[0] / stride), int(input.shape[1] / stride))
        output = np.zeros((int(input.shape[0] / stride),
                               int(input.shape[1] / stride),
                               filter.shape[-1]))
        new_dims = [input.shape[0] + filter.shape[0] - stride, input.shape[1] + filter.shape[1] - stride]
        added_rows = new_dims[0] - input.shape[0]
        new_input = np.zeros((new_dims[0], new_dims[1], input.shape[2]))
        if added_rows % 2 == 0:
            new_input[int(added_rows / 2): int(added_rows / 2) + input.shape[0],
            int(added_rows / 2): int(added_rows / 2) + input.shape[1], :] = input
        else:
            added_rows -= 1
            new_input[int(added_rows / 2): int(added_rows / 2) + input.shape[0],
            int(added_rows / 2): int(added_rows / 2) + input.shape[1], :] = input

        for i in range(filter.shape[-1]):
            output[:, :, i] = single_conv(filter[:, :, :, i], new_input, out_dim, stride)
        output += bias
        return relu(output)
    else:
        out_dim = (int(np.floor(input.shape[0] - filter.shape[0]) / stride) + 1,
                   int(np.floor(input.shape[1] - filter.shape[1]) / stride) + 1)
        output = np.zeros((out_dim[0], out_dim[1], filter.shape[-1]))
        for i in range(filter.shape[-1]):
            output[:, :, i] = single_conv(filter[:, :, :, i], input, out_dim, stride)
        output += bias
        return relu(output)


'''This function takes the index of the top-left position of the filter and returns all input coordinates associated
with the convolution in that position.
'''
def get_conv_coords(i, j, kernel_dims, input_dims):
    input_coords = []
    assert i + kernel_dims[0] <= input_dims[0] and j + kernel_dims[1] <= input_dims[1], "Kernel goes outside input"
    for i_ in range(i, i + kernel_dims[0]):
        for j_ in range(j, j + kernel_dims[1]):
            for k_ in range(kernel_dims[2]):
                input_coords.append((input_dims[2]*((i_*input_dims[0])+j_)) + k_)
    return input_coords


def get_conv_coord_from_index(index, input_dims):
    k = (index % input_dims[2])
    i = (np.floor(index/(input_dims[1] * input_dims[2])))
    j = (np.floor((index - (i* (input_dims[1] * input_dims[2]))) / input_dims[2]))
    assert int(k) == k and int(j) == j and int(i) == i, "Something has gone wrong in the coordinate conversion."
    return [int(i), int(j), int(k)]


'''
Performs interval bound propagation through conv layers.
This function takes only a single kernel filter at a time (for Cifar 32x32x3).
OUTPUT OF THIS FUNCTION IS PRE-RELU!
'''
def interval_conv_prop(input_lb, input_ub, bias, filters, stride):
    # input_lb, input_ub, filters, bias, stride = args
    # output_min = np.zeros((int(np.floor(input_lb.shape[0] - filters.shape[0]) / stride) + 1, int(np.floor(input_lb.shape[1] - filters.shape[1]) / stride)+1,
    #                    filters.shape[-1]))
    output_min = np.zeros((int(input_lb.shape[0] / stride),
                           int(input_lb.shape[1] / stride),
                           filters.shape[-1]))
    output_max = np.zeros(output_min.shape)
    new_dims = [input_lb.shape[0] + filters.shape[0] - stride, input_lb.shape[1] + filters.shape[1] - stride]
    added_rows = new_dims[0] - input_lb.shape[0]
    new_input_lb = np.zeros((new_dims[0], new_dims[1], input_lb.shape[2]))
    new_input_ub = np.zeros((new_dims[0], new_dims[1], input_lb.shape[2]))
    if added_rows%2 == 0:
        new_input_lb[int(added_rows/2): int(added_rows/2)+ input_lb.shape[0],
        int(added_rows/2): int(added_rows/2)+ input_lb.shape[1], :] = input_lb

        new_input_ub[int(added_rows/2): int(added_rows/2)+ input_lb.shape[0],
        int(added_rows/2): int(added_rows/2)+ input_lb.shape[1], :] = input_ub
    else:
        added_rows -= 1
        new_input_lb[int(added_rows/2): int(added_rows/2)+ input_lb.shape[0],
        int(added_rows/2): int(added_rows/2)+ input_lb.shape[1], :] = input_lb

        new_input_ub[int(added_rows/2): int(added_rows/2)+ input_lb.shape[0],
        int(added_rows/2): int(added_rows/2)+ input_lb.shape[1], :] = input_ub


    vec_input_lb = new_input_lb.flatten()
    vec_input_ub = new_input_ub.flatten()
    for filter in range(filters.shape[-1]):
        for i in range(output_min.shape[0]):
            for j in range(output_min.shape[1]):
                min_output = 0
                max_output = 0
                input_indices = get_conv_coords(i*stride, j*stride, filters[:, :, :, filter].shape, new_input_lb.shape)
                upper_vector = [vec_input_ub[i] for i in input_indices]
                lower_vector = [vec_input_lb[i] for i in input_indices]
                min_output += np.sum([min(f*low, f*up) for f, up, low in zip(list(filters[:, :, :, filter].flatten()),
                                                                                upper_vector, lower_vector)])
                max_output += np.sum([max(f * low, f * up) for f, up, low in zip(filters[:, :, :, filter].flatten(),
                                                                                    upper_vector, lower_vector)])
                output_min[i, j, filter] = min_output
                output_max[i, j, filter] = max_output
        output_min[:, :, filter] += bias[filter]
        output_max[:, :, filter] += bias[filter]
    return output_min, output_max
