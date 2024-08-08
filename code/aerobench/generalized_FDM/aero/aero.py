from math import sin, cos, pi
def aero(model, x, u, geom):
    rtod = 57.29578  # rad to degree
    if model == 'stevens_f16' or model == 'morelli_f16':
        from aircraft_FDM.f16.aero.cx import cx
        from aircraft_FDM.f16.aero.cy import cy
        from aircraft_FDM.f16.aero.cz import cz
        from aircraft_FDM.f16.aero.cl import cl
        from aircraft_FDM.f16.aero.dlda import dlda
        from aircraft_FDM.f16.aero.dldr import dldr
        from aircraft_FDM.f16.aero.cm import cm
        from aircraft_FDM.f16.aero.cn import cn
        from aircraft_FDM.f16.aero.dnda import dnda
        from aircraft_FDM.f16.aero.dndr import dndr
        from aircraft_FDM.f16.aero.dampp import dampp
        from aircraft_FDM.f16.aero.morellif16 import Morellif16
        dail = u[2]/20
        drdr = u[3]/30
        if model == 'stevens_f16':
            # stevens & lewis (look up table version)
            cxt = cx(x[1]*rtod, u[1])
            cyt = cy(x[2]*rtod, u[2], u[3])
            czt = cz(x[1]*rtod, x[2]*rtod, u[1])

            clt = cl(x[1]*rtod, x[2]*rtod) + dlda(x[1]*rtod, x[2]*rtod) * dail + dldr(x[1]*rtod, x[2]*rtod) * drdr
            cmt = cm(x[1]*rtod, u[1])
            cnt = cn(x[1]*rtod, x[2]*rtod) + dnda(x[1]*rtod, x[2]*rtod) * dail + dndr(x[1]*rtod, x[2]*rtod) * drdr
        else:
        # morelli model (polynomial version)
            cxt, cyt, czt, clt, cmt, cnt = Morellif16(x[1], x[2], u[1]*pi/180, u[2]*pi/180, u[3]*pi/180, \
                                                    x[6], x[7], x[8], geom['cbar'], geom['b'], x[0], geom['xcg'], geom['xcgr'])
        d = dampp(x[1]*rtod)
    return cxt, cyt, czt, clt, cmt, cnt, d