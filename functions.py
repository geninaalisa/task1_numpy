def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Not vectorized implementation.
    """
    p = 1
    for i in range(len(x)):
        if i < len(x[i]):
            if x[i][i] != 0:
                p *= x[i][i]
    return p

def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Not vectorized implementation.
    """
    x.sort()
    y.sort()
    if np.all(x == y):
        return True
    return False

def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Not vectorized implementation.
    """
    m = min(x)
    for i in range(len(x) - 1):
        if x[i] == 0 and x[i + 1] != 0 and x[i + 1] > m:
            m = x[i + 1]
    return m

def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Not vectorized implementation.
    """
    h = len(img)
    w = len(img[0])
    res = []
    for i in range(h):
        str = []
        for j in range(w):
            sum = 0
            for k in range(len(coefs)):
                sum += img[i][j][k] * coefs[k]
            str.append(sum)
        res.append(str)
    return res

def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Not vectorized implementation.
    """
    dig = []
    count = []
    for i in range(len(x)):
        if not(x[i] in dig):
            dig.append(x[i])
            count.append(1)
        else:
            for j in range(len(dig)):
                if x[i] == dig[j]:
                    count[j] += 1
    return (dig, count)

def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Not vectorized implementation.
    """
    res = []
    for i in range(len(x)):
        curr_lst = []
        for j in range(len(y)):
            dst = 0
            for k in range(len(x[0])):
                dst += (x[i][k] - y[j][k]) ** 2
            curr_lst.append((dst) ** 0.5)
        res.append(curr_lst)
    return res
