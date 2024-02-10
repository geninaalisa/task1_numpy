import numpy as np

def prod_non_zero_diag_v(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Vectorized implementation.
    """
    return np.prod(np.diag(x)[np.diag(x)!=0])

def are_multisets_equal_v(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Vectorized implementation.
    """
    if(np.shape(np.unique(x, return_counts=True)[0]) != np.shape(np.unique(y, return_counts=True)[0])):
        return False
    if(np.any(np.unique(x, return_counts=True)[0] != np.unique(y, return_counts=True)[0]) or np.any(np.unique(x, return_counts=True)[1] != np.unique(y, return_counts=True)[1])):
        return False
    return True

def max_after_zero_v(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Vectorized implementation.
    """
    ind = np.hstack((np.ones((1)), x)) == 0
    return np.max(x[ind[:-1]])

def convert_image_v(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Vectorized implementation.
    """
    return np.sum(img * coefs, axis=-1)

def run_length_encoding_v(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Vectorized implementation.
    """
    y = np.hstack((np.ones(1), x[:- 1]))
    first_pos = x != y
    first_pos[0] = True
    ind1 = np.arange(np.size(x))[first_pos]
    ind2 = np.hstack((ind1[1:], np.array([np.size(x)])))
    return x[first_pos], ind2 - ind1

def pairwise_distance_v(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Vctorized implementation.
    """
    return np.sqrt(np.sum((x[:, np.newaxis] - y) ** 2, axis=-1))
