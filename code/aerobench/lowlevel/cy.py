'''
Stanley Bak
Python F-16 GCAS
'''

def cy(beta, ail, rdr):
    'cy function'

    return -.02 * beta + .021 * (ail / 20) + .086 * (rdr / 30)
