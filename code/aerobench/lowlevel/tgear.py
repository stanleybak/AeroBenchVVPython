'''
Stanley Bak
Python F-16 GCAS
'''

def tgear(thtl):
    'tgear function'

    if thtl <= .77:
        tg = 64.94 * thtl
    else:
        tg = 217.38 * thtl - 117.38

    return tg
