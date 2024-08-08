'''
Stanley Bak
Python F-16 subf16
outputs aircraft state vector deriative
'''

#         x[0] = air speed, VT    (ft/sec)
#         x[1] = angle of attack, alpha  (rad)
#         x[2] = angle of sideslip, beta (rad)
#         x[3] = roll angle, phi  (rad)
#         x[4] = pitch angle, theta  (rad)
#         x[5] = yaw angle, psi  (rad)
#         x[6] = roll rate, P  (rad/sec)
#         x[7] = pitch rate, Q  (rad/sec)
#         x[8] = yaw rate, R  (rad/sec)
#         x[9] = northward horizontal displacement, pn  (feet)
#         x[10] = eastward horizontal displacement, pe  (feet)
#         x[11] = altitude, h  (feet)
#         x[12] = engine thrust dynamics lag state, pow
#
#         u[0] = throttle command  0.0 < u(1) < 1.0
#         u[1] = elevator command in degrees
#         u[2] = aileron command in degrees
#         u[3] = rudder command in degrees
#

from math import sin, cos, pi

from atmosphere.adc import adc
from mass_inertia.mass_inertia import mass_inertia
from engine.thrust import thrust
from engine.tgear import tgear
from engine.pdot import pdot
from aero.aero import aero

def sub_models(x, u, model, adjust_cy=True):
    '''output aircraft state vector derivative for a given input

    The reference for the model is Appendix A of Stevens & Lewis
    '''

    assert model in ['stevens_f16', 'morelli_f16']
    assert len(x) == 13
    assert len(u) == 4

    # Getting input control values
    thtlc, el, ail, rdr = u

    # Specific to the aircraft
    geom = mass_inertia(model)
    xcg = geom['xcg']  # CG position in the X direction, current
    s = geom['s']     # Total wing area
    b = geom['b']      # Wing span
    cbar = geom['cbar']  # Wing chord
    rm = geom['rm']  # 1/mass
    xcgr = geom['xcgr'] # CG position in the X direction, reference
    he = geom['he']  # could represent a moment-related constant, such as an aerodynamic hinge moment
    # These are related to the moment of inertia values
    c1 = geom['c1']
    c2 = geom['c2']
    c3 = geom['c3']
    c4 = geom['c4']
    c5 = geom['c5']
    c6 =  geom['c6']
    c7 =  geom['c7']
    c8 =  geom['c8']
    c9 =  geom['c9']

    # Constants
    rtod = 57.29578  # rad to degree
    g = 32.17    # gravitational constant (ft/s²)

    xd = x.copy()
    vt = x[0]
    alpha = x[1]*rtod
    beta = x[2]*rtod
    phi = x[3]
    theta = x[4]
    psi = x[5]
    p = x[6]
    q = x[7]
    r = x[8]
    alt = x[11]
    power = x[12]

    # ----- Air data computer ----- #
    amach, qbar = adc(vt, alt)

    # ----- Engine model ----- #
    cpow = tgear(thtlc, model)
    xd[12] = pdot(power, cpow, model)
    t = thrust(power, alt, amach, model)

    # ----- Aero model and damping matrix ----- #
    cxt, cyt, czt, clt, cmt, cnt, d = aero(model, x, u, geom)
    
    # add damping derivatives
    tvt = .5 / vt
    b2v = b * tvt
    cq = cbar * q * tvt
    
    # ----- Flight Dynamics Equations ----- #
    # get ready for state equations
    cxt = cxt + cq * d[0]
    cyt = cyt + b2v * (d[1] * r + d[2] * p)
    czt = czt + cq * d[3]
    clt = clt + b2v * (d[4] * r + d[5] * p)
    cmt = cmt + cq * d[6] + czt * (xcgr-xcg)
    cnt = cnt + b2v * (d[7] * r + d[8] * p)-cyt * (xcgr-xcg) * cbar/b
    cbta = cos(x[2])
    u = vt * cos(x[1]) * cbta
    v = vt * sin(x[2])
    w = vt * sin(x[1]) * cbta
    sth = sin(theta)
    cth = cos(theta)
    sph = sin(phi)
    cph = cos(phi)
    spsi = sin(psi)
    cpsi = cos(psi)
    qs = qbar * s
    qsb = qs * b
    rmqs = rm * qs
    gcth = g * cth
    qsph = q * sph
    ay = rmqs * cyt
    az = rmqs * czt

    # force equations
    udot = r * v-q * w-g * sth + rm * (qs * cxt + t)
    vdot = p*w - r*u + gcth*sph + ay
    wdot = q * u-p * v + gcth * cph + az
    dum = (u * u + w * w)

    xd[0] = (u * udot + v * vdot + w * wdot)/vt
    xd[1] = (u * wdot-w * udot)/dum
    xd[2] = (vt * vdot-v * xd[0]) * cbta/dum

    # kinematics
    xd[3] = p + (sth/cth) * (qsph + r * cph)
    xd[4] = q * cph-r * sph
    xd[5] = (qsph + r * cph)/cth

    # moments
    xd[6] = (c2 * p + c1 * r + c4 * he) * q + qsb * (c3 * clt + c4 * cnt)

    xd[7] = (c5 * p-c7 * he) * r + c6 * (r*r - p*p) + qs * cbar * c7 * cmt
    xd[8] = (c8 * p-c2 * r + c9 * he) * q + qsb * (c4 * clt + c9 * cnt)

    # navigation
    t1 = sph * cpsi
    t2 = cph * sth
    t3 = sph * spsi
    s1 = cth * cpsi
    s2 = cth * spsi
    s3 = t1 * sth-cph * spsi
    s4 = t3 * sth + cph * cpsi
    s5 = sph * cth
    s6 = t2 * cpsi + t3
    s7 = t2 * spsi-t1
    s8 = cph * cth
    xd[9] = u * s1 + v * s3 + w * s6 # north speed
    xd[10] = u * s2 + v * s4 + w * s7 # east speed
    xd[11] = u * sth-v * s5-w * s8 # vertical speed

    return xd
