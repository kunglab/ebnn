import numpy as np

storage_ts = ['row_major', 'col_major']


def binarize(W):
    return np.where(W >= 0, 1, -1).astype(np.float32, copy=False)


def binarize_real(W):
    return np.where(W >= 0, 1, 0).astype(int, copy=False)


def np_to_floatC(xs, name, storage_t):
    '''
    Converts a numpy array into a float C array
    '''
    if storage_t not in storage_ts:
        print('storage_t: {} invalid. Options are {}.'.format(
            storage_t, storage_ts))
        return

    if storage_t == 'col_major':
        xs = xs.T

    xs = xs.flatten()
    float_buf = []
    float_buf = list(map(repr, xs))

    c_str = 'float {}[{}] = {{{}}};'.format(
        name, len(float_buf), ','.join(list(map(str, float_buf))))

    return c_str


def np_to_uint8C(xs, name, storage_t, pad='0'):
    '''
    Converts a numpy array into a binary C array stored in uint8s
    '''
    if storage_t not in storage_ts:
        print('storage_t: {} invalid. Options are {}.'.format(
            storage_t, storage_ts))

    bit_range = 8
    int_buf = []

    if storage_t == 'col_major':
        xs = xs.T

    for x in xs:
        for i in range(0, len(x), bit_range):
            xi = x[i:i + bit_range]
            xi = ''.join(map(str, xi))
            xi = xi.ljust(bit_range, pad)

            xi = int(xi, 2)
            int_buf.append(xi)

    c_str = 'uint8_t {}[{}] = {{{}}};'.format(
        name, len(int_buf), ','.join(map(str, int_buf)))

    return c_str


def np_to_packed_uint8C(xs, name, storage_t, pad='0'):
    '''
    Converts a numpy array into a binary C array stored in uint8s
    '''
    if storage_t not in storage_ts:
        print('storage_t: {} invalid. Options are {}.'.format(
            storage_t, storage_ts))

    bit_range = 8
    int_buf = []

    if storage_t == 'col_major':
        xs = xs.T

    xs = xs.flatten()
    for i in range(0, len(xs), bit_range):
        xi = xs[i:i + bit_range]
        xi = ''.join(map(str, xi))
        xi = xi.ljust(bit_range, pad)
        xi = int(xi, 2)
        int_buf.append(xi)

    c_str = 'uint8_t {}[{}] = {{{}}};'.format(
        name, len(int_buf), ','.join(map(str, int_buf)))

    return c_str
