

def thrust(power, alt, amach, model, thtlc):
    if model == 'stevens_f16' or model == 'morelli_f16':
        from aircraft_FDM.f16.engine.thrust import thrust
        thrust_val, power_dot = thrust(power, alt, amach, thtlc)
        
    return thrust_val, power_dot