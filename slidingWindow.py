# Got this sources from the http://www.johnvinyard.com/blog/?p=268

import numpy as np
from numpy.lib.stride_tricks import as_strided as ast


def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(image, windowSize, shiftSize=None, flatten=True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        image  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is shiftSize:
        # ss was not provided. the windows will not overlap in any direction.
        shiftSize = windowSize
    windowSize = norm_shape(windowSize)
    shiftSize = norm_shape(shiftSize)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    windowSize = np.array(windowSize)
    shiftSize = np.array(shiftSize)
    shape = np.array(image.shape)


    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(windowSize),len(shiftSize)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(windowSize > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(image.shape),str(windowSize)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - windowSize) // shiftSize) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(windowSize)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(image.strides) * shiftSize) + image.strides
    strided = ast(image,shape = newshape,strides = newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(windowSize) if windowSize.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if windowSize.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)