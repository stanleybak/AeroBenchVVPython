'''
Python utilities for F-16 GCAS
'''

from math import floor, ceil
import numpy as np

class Freezable(object):
    'a class where you can freeze the fields (prevent new fields from being created)'

    _frozen = False

    def freeze_attrs(self):
        'prevents any new attributes from being created in the object'
        self._frozen = True

    def __setattr__(self, key, value):
        if self._frozen and not hasattr(self, key):
            raise TypeError("{} does not contain attribute '{}' (object was frozen)".format(self, key))

        object.__setattr__(self, key, value)

def printmat(mat, main_label, row_label_str, col_label_str):
    'print a matrix'

    if isinstance(row_label_str, list) and len(row_label_str) == 0:
        row_label_str = None

    assert isinstance(main_label, str)
    assert row_label_str is None or isinstance(row_label_str, str)
    assert isinstance(col_label_str, str)

    mat = np.array(mat)
    if len(mat.shape) == 1:
        mat.shape = (1, mat.shape[0]) # one-row matrix

    print "{} =".format(main_label)

    row_labels = None if row_label_str is None else row_label_str.split(' ')
    col_labels = col_label_str.split(' ')

    width = 7

    width = max(width, max([len(l) for l in col_labels]))

    if row_labels is not None:
        width = max(width, max([len(l) for l in row_labels]))

    width += 1

    # add blank space for row labels
    if row_labels is not None:
        print "{: <{}}".format('', width),

    # print col lables
    for col_label in col_labels:
        if len(col_label) > width:
            col_label = col_label[:width]

        print "{: >{}}".format(col_label, width),

    print ""

    if row_labels is not None:
        assert len(row_labels) == mat.shape[0], \
            "row labels (len={}) expected one element for each row of the matrix ({})".format( \
            len(row_labels), mat.shape[0])

    for r in xrange(mat.shape[0]):
        row = mat[r]

        if row_labels is not None:
            label = row_labels[r]

            if len(label) > width:
                label = label[:width]

            print "{:<{}}".format(label, width),

        for num in row:
            #print "{:#<{}}".format(num, width),
            print "{:{}.{}g}".format(num, width, width-3),

        print ""


def fix(ele):
    'round towards zero'

    assert isinstance(ele, float)

    if ele > 0:
        rv = int(floor(ele))
    else:
        rv = int(ceil(ele))

    return rv

def sign(ele):
    'sign of a number'

    if ele < 0:
        rv = -1
    elif ele == 0:
        rv = 0
    else:
        rv = 1

    return rv

