def calculate_cfl(DX, DT, heat_cond_coeff):
    return DT * heat_cond_coeff / (DX**2)