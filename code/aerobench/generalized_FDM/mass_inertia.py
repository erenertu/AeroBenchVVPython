

def mass_inertia(model):
    
    if model == 'stevens_f16' or model == 'morelli_f16':
        from aircraft_FDM.f16.mass_inertia.mass_inertia import mass_inertia
        geom = mass_inertia()

    return geom