def fed_avg(zero_model, w_local_list, total_data_size):
    keys = w_local_list[0][0].keys()

    for k in keys:
        for w in w_local_list:
            beta = float(w[1]) / float(total_data_size)
            if 'num_batches_tracked' in k:
                zero_model[k] = w[0][k]
            else:
                zero_model[k] += (w[0][k] * beta)
    return zero_model
