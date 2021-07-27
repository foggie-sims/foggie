def flatten_thelphil(bins, Lhst, Mhst):
    t = np.mean((bins[-2][:-1], bins[-2][1:]), axis = 0)
    p = np.mean((bins[-1][:-1], bins[-1][1:]), axis = 0)
    TT, PP = np.meshgrid(t, p[::-1])
    TTrad = TT*pi/180
    PPrad = PP*pi/180
    x = Lhst * sin(PPrad) * cos(TTrad)
    y = Lhst * sin(PPrad) * sin(TTrad)
    z = Lhst * cos(PPrad)

    x_sum = np.nansum(x)*u.g*u.cm**2./u.s
    y_sum = np.nansum(y)*u.g*u.cm**2./u.s
    z_sum = np.nansum(z)*u.g*u.cm**2./u.s

    L_sum = np.sqrt(x_sum**2 + y_sum**2. + z_sum**2)
    thel_mean = np.arctan2(y_sum,x_sum)*180./pi
    phil_mean = np.arccos(z_sum/L_sum)*180./pi
    L_sum = L_sum
    M_sum = np.sum(Mhst)*u.Msun
    j_sum = L_sum/M_sum
    return L_sum, j_sum.to('cm**2/s'), thel_mean, phil_mean, x_sum/M_sum, y_sum/M_sum, z_sum/M_sum, M_sum