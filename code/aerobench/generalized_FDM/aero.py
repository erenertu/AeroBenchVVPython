

def aero(model, x, u, geom):
    if model == 'stevens_f16' or model == 'morelli_f16':
        from aircraft_FDM.f16.aero.aero import aero
        cxt, cyt, czt, clt, cmt, cnt, d = aero(model, x, u, geom)
        
    return cxt, cyt, czt, clt, cmt, cnt, d