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

from adc import adc
from mass_inertia import mass_inertia
from engine import engine
from aero import aero

# Constants
rtod = 57.29578  # rad to degree
g = 32.17    # gravitational constant (ft/s²)

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
    xcg = geom['xcg']  # CG position in the X direction, current # TODO: It can be modelled dynamically to calculate CG after mass drops.
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
    amach, qbar = adc(vt, alt)  # Mach number and dynamic pressure

    # ----- Engine model ----- #
    t, xd[12] = engine(power, alt, amach, model, thtlc)

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
    cos_beta = cos(x[2])
    u = vt * cos(x[1]) * cos_beta
    v = vt * sin(x[2])
    w = vt * sin(x[1]) * cos_beta

    sin_theta = sin(theta)
    cos_theta = cos(theta)
    sin_phi = sin(phi)
    cos_phi = cos(phi)
    sin_psi = sin(psi)
    cos_psi = cos(psi)

    qs = qbar * s
    qsb = qs * b
    rmqs = rm * qs

    gcos_theta = g * cos_theta
    qsin_phi = q * sin_phi

    ay = rmqs * cyt
    az = rmqs * czt
    ax = rm * (qs * cxt + t)

    # force equations
    udot = r * v-q * w-g * sin_theta + ax
    vdot = p*w - r*u + gcos_theta*sin_phi + ay
    wdot = q * u-p * v + gcos_theta * cos_phi + az
    dum = (u * u + w * w)

    xd[0] = (u * udot + v * vdot + w * wdot)/vt
    xd[1] = (u * wdot-w * udot)/dum
    xd[2] = (vt * vdot-v * xd[0]) * cos_beta/dum

    # kinematics
    xd[3] = p + (sin_theta/cos_theta) * (qsin_phi + r * cos_phi)
    xd[4] = q * cos_phi-r * sin_phi
    xd[5] = (qsin_phi + r * cos_phi)/cos_theta

    # moments
    xd[6] = (c2 * p + c1 * r + c4 * he) * q + qsb * (c3 * clt + c4 * cnt)

    xd[7] = (c5 * p-c7 * he) * r + c6 * (r*r - p*p) + qs * cbar * c7 * cmt
    xd[8] = (c8 * p-c2 * r + c9 * he) * q + qsb * (c4 * clt + c9 * cnt)

    # navigation
    t1 = sin_phi * cos_psi
    t2 = cos_phi * sin_theta
    t3 = sin_phi * sin_psi
    s1 = cos_theta * cos_psi
    s2 = cos_theta * sin_psi
    s3 = t1 * sin_theta - cos_phi * sin_psi
    s4 = t3 * sin_theta + cos_phi * cos_psi
    s5 = sin_phi * cos_theta
    s6 = t2 * cos_psi + t3
    s7 = t2 * sin_psi-t1
    s8 = cos_phi * cos_theta
    xd[9] = u * s1 + v * s3 + w * s6 # north speed
    xd[10] = u * s2 + v * s4 + w * s7 # east speed
    xd[11] = u * sin_theta-v * s5-w * s8 # vertical speed

    return xd
