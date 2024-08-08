'''
Stanley Bak
Python F-16
power derivative (pdot)
'''


def pdot(p3, p1, model):
    'pdot function'
    if model == 'stevens_f16' or model == 'morelli_f16':
        from aircraft_FDM.f16.engine.rtau import rtau
        if p1 >= 50:
            if p3 >= 50:
                t = 5
                p2 = p1
            else:
                p2 = 60
                t = rtau(p2 - p3)
        else:
            if p3 >= 50:
                t = 5
                p2 = 40
            else:
                p2 = p1
                t = rtau(p2 - p3)

        pd = t * (p2 - p3)

    return pd
