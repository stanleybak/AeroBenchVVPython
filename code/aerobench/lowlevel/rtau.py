'''
Stanley Bak
Python F-16

Rtau function
'''

def rtau(dp):
    'rtau function'

    if dp <= 25:
        rt = 1.0
    elif dp >= 50:
        rt = .1
    else:
        rt = 1.9 - .036 * dp

    return rt
