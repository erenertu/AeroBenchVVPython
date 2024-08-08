

def mass_inertia(model):
    geom = {}
    if model == 'stevens_f16' or model == 'morelli_f16':
        geom['xcg'] = 0.35  # CG position in the X direction, current
        geom['s'] = 300     # Total wing area
        geom['b'] = 30      # Wing span
        geom['cbar'] = 11.32  # Wing chord
        geom['rm'] = 1.57e-3  # 1/mass
        geom['xcgr'] = .35  # CG position in the X direction, reference
        geom['he'] = 160.0  # could represent a moment-related constant, such as an aerodynamic hinge moment
        # These are related to the moment of inertia values
        geom['c1'] = -.770
        geom['c2'] = .02755
        geom['c3'] = 1.055e-4
        geom['c4'] = 1.642e-6
        geom['c5'] = .9604
        geom['c6'] = 1.759e-2
        geom['c7'] = 1.792e-5
        geom['c8'] = -.7336
        geom['c9'] = 1.587e-5

    return geom