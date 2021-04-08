##!/usr/bin/env python3

"""

    Title :      make_mappings_grid.py
    Notes :      To compute 4D MAPPINGS grid of age-nII(density)-<U> (volume avergaed ionisation parameter)-metallicity, in parallel using multiple cores
    Output :     One master txt file with the predicted emission line fluxes for all model grid point
    Author :     Ayan Acharyya
    Started :    January 2021
    Example :    run make_mappings_grid.py

"""
from header import *
from util import *
start_time = time.time()

# ----------------------------------------------------------------------------------------
def collate_grid(lines_to_pick, outtag):
    '''
    Function to collate the recently produced model grid and write as one txt file
    '''

    gridfilename = mappings_lab_dir + 'totalspec' + outtag + '.txt'
    fout = open(gridfilename, 'w')

    head = 'GN Z age nII <U> logQ0   Om  lpok    Rs(pc)  r_i(pc) r_m(pc) lqin'
    for n in lines_to_pick['label']: head += '  ' + n

    i = 0
    ind, ag, nII, U, lQ0, Om, lpok, R_s, r_i, r_m, lqin, em, ZZ = [], [], [], [], [], [], [], [], [], [], [], [], []

    for Z in Z_arr:
        for age in age_arr:
            for lognII in lognII_arr:
                for logU in logU_arr:
                    i += 1
                    Z1, ag1, nII1, U1, lQ01, Om1, lpok1, R_s1, r_i1, r_m1, lqin1 = calcprint(i, Z, age, lognII, logU, table=True)
                    lpok.append(lpok1), R_s.append(R_s1), r_i.append(r_i1), r_m.append(r_m1), lqin.append(lqin1), \
                    ind.append(i), ag.append(ag1), nII.append(nII1), U.append(U1), lQ0.append(lQ01), Om.append(Om1), ZZ.append(Z1)
                    fname = mappings_lab_dir + 'results' + outtag + '/spec' + str(i) + '.csv'
                    fin = open(fname, 'r')
                    lines = fin.readlines()
                    em1 = []
                    hb_line_ind = int(subprocess.check_output(['grep -nF  "H-beta" ' + fname], shell=True).split()[0][:-1]) - 1
                    hb = float(num(lines[hb_line_ind].split()[4]))
                    for j in lines_to_pick['wave_air'].round(3).astype(str): # since MAPPINGS models output in air wavelengths
                        flag = 0
                        for line in lines:
                            if j in line:
                                em1.append(float(num(line.split()[2])) * hb)
                                flag = 1
                                break
                        if flag == 0:
                            em1.append(0.)
                    if i == 1:
                        em = em1
                    else:
                        em = np.vstack((em, em1))
                    fin.close()
    em = np.transpose(np.array(em))
    outar = np.row_stack((ind, ZZ, ag, nII, U, lQ0, Om, lpok, R_s, r_i, r_m, lqin, em))
    np.savetxt(fout, np.transpose(outar), "%d  %.2F  %.0E  %.0E  %.0E  %.3F  %.1E  %.3F  %.1E  %.1E  %.1E  %.3F" + " %.2E" * len(lines_to_pick), header=head, comments='')
    fout.close()
    print ('Saved model grid as', gridfilename)

    photgrid = pd.read_table(gridfilename, comment='#', delim_whitespace=True)
    return photgrid

# --------------------------------------------------------
def getset(i):
    '''
    Function to setup the working directory for running each MAPPINGS model
    '''

    os.chdir(mappings_lab_dir + '')
    subprocess.call(['mkdir -p child' + str(i)], shell=True)
    os.chdir('child' + str(i))
    subprocess.call(['ln -sf ../data data'], shell=True)
    subprocess.call(['ln -sf ../abund abund'], shell=True)
    subprocess.call(['ln -sf ../atmos atmos'], shell=True)
    subprocess.call(['ln -sf ../scripts scripts'], shell=True)
    subprocess.call(['ln -sf ../map.prefs map.prefs'], shell=True)
    subprocess.call(['ln -sf ../mapStd.prefs mapStd.prefs'], shell=True)

# --------------------------------------------------------
def func(nII, lQ0):
    '''
    Function used to solve for Omega
    '''

    global alpha_B, ltemp
    return ((81. * nII * (10 ** lQ0) * alpha_B ** 2 / (256. * np.pi * (c*1e3) ** 3)) ** (1 / 3.)) # Acharyya+2019a

# --------------------------------------------------------
def solve_for_Om(Om, nII, U, lQ0):
    '''
    Function to solve for Omega
    '''

    global alpha_B, ltemp
    return (np.abs(func(nII, lQ0) * ((1 + Om) ** (4 / 3.) - (4. / 3. + Om) * Om ** (1 / 3.)) - U)) # Acharyya+2019a

# --------------------------------------------------------
def calcprint(i, Z, age, lognII, logU, table=False):
    '''
    Function to calculate quantities to print on screen, or save in table
    '''

    global alpha_B, ltemp, lQ0
    nII = 10 ** lognII
    U = 10 ** logU
    lQ0 = SB99_logQ[np.where(SB99_age == age * 10 ** 6)[0][0]] + np.log10(mappings_starparticle_mass) - np.log10(sb99_mass)  # taking logQ of each gridpoint from the
    # corresponding age  in the SB99 quanta file and
    # then scaling sb99_mass (1e6 Msun) to mappings_starparticle_mass (1e3 Msun)
    Om = op.fminbound(solve_for_Om, 0., 1e5, args=(nII, U, lQ0)) # Omega of HII region, determines volume ratio of wind cavity to ionised shell (see Acharyya+2019b)
    lpok = lognII + ltemp - 6.  # MAPPINGS needs p/k in cgs units, hence -6
    Rs = (3 * (10 ** lQ0) / (4 * np.pi * alpha_B * nII ** 2)) ** (1 / 3.) # Stromgen radius
    r_i = Rs * (Om ** (1 / 3.)) # inner radius of ionised shell
    r_m = Rs * ((1 + Om) ** (1 / 3.)) # outer radius of ionises shell
    lUin = np.log10((10 ** lQ0) / (4 * np.pi * r_i ** 2 * nII * (c*1e3))) # (log) ionisation parameter at innermost edge of shell, this is required for input to MAPPINGS
    if not table:
        return lpok, lUin, lQ0, (str(i) + '\t' + str('%.2F' % Z) + '\t' + str(format(age * 10 ** 6, '0.0e')) + '\t' + str(
            format(nII, '0.0e')) + '\t' + str(format(U, '0.0e')) + \
                            '\t' + str(format(lQ0, '0.3f')) + '\t' + str(format(Om, '0.1e')) + '\t' + str(
                    format(lpok, '0.3f')) + '\t' + str(format(Rs / 3.086e16, '0.1e')) + '\t' + str(
                    format(r_i / 3.086e16, \
                           '0.1e')) + '\t' + str(format(r_m / 3.086e16, '0.1e')) + '\t' + str(
                    format(lUin, '0.3f')) + '\t')
    else:
        return Z, age * 10 ** 6, nII, U, lQ0, Om, lpok, Rs / 3.086e16, r_i / 3.086e16, r_m / 3.086e16, lUin

# -------------------------------------------------------------------------------
def rungridpoint(i, Z, age, lognII, logU, parallel, clobber=False):
    '''
    Function to compute MAPPINGS model for a single grid point
    '''

    start_time2 = time.time()
    global alpha_B, ltemp, outtag
    if not clobber and os.path.exists(mappings_lab_dir + 'results' + outtag + '/spec' + str(i) + '.csv'):
        print('lines_to_pick file already exists. Use clobber=True to overwrite.\n')
        return
    lpok, lUin, loglum, s = calcprint(i, Z, age, lognII, logU)
    getset(i)
    if not parallel:
        sys.stdout.write(s)
        sys.stdout.flush()
    input_phot = mappings_input_dir + 'Int_' + str(Z) + 'sol.sou'
    abun_filename = mappings_input_dir + 'lgc' + '%03d' % int(100 * Z) + '.abn'  # to run with lgc.abn files

    # -------------------Calling MAPPINGS-------------------------------------
    replacements = {'abun_file': abun_filename, 'dust_file': dust_filename, 'phot_file': input_phot,
                    'loglum': str(loglum), 'lpok': str(lpok), 'ltemp': str(ltemp), \
                    'luin': str(lUin), 'name': 'gridpoint' + str(i) + '_age' + str(age) + '_lnII' + str(lognII) + '_lU' + str(logU)}  # to run with lgc.abn files
    with open(fin) as infile, open(fout, 'w') as outfile:
        for line in infile:
            for src, lines_to_pick in replacements.items():
                line = line.replace(src, lines_to_pick)
            outfile.write(line)
    subprocess.call(['cp ' + fout + ' ' + mappings_lab_dir + 'results' + outtag + '/USED_inp.txt'], shell=True)
    subprocess.call(['../map51'], stdin=open(fout), stdout=open('junk.txt', 'w'))
    subprocess.call(['rm -f *otn* *lss* *apn* *sem*'], shell=True)
    subprocess.call(['mv spec* ' + mappings_lab_dir + 'results' + outtag + '/spec' + str(i) + '.csv'], shell=True)

    os.chdir(mappings_lab_dir + '')
    subprocess.call(['rm -r child' + str(i)], shell=True)
    if not parallel:
        print(str(format((time.time() - start_time2) / 60, '0.2f')))
    else:
        print(s + str(format((time.time() - start_time2) / 60, '0.2f')))


# ----------- declaring constants ---------------------------------------------
global i, alpha_B, ltemp, outtag
ltemp = 4.  # assumed 1e4 K temp; for what? for the initial guess??
mappings_starparticle_mass = 1000. # Msun, mass to scale the output SB99 luminosity to; 1000 Msun because median mass for FOGGIE star particles ~ 1000 Msun
# speed of light c is imported from header.py, in units of km/s

i_start = 1849 # set to non-zero in case some of the model grid is already computed
i_end = 1e10 # set to some crazy high value to let the code run through till the end of the grid
clobber = False

# ------------declaring paths--------------------------------------------
fin = mappings_lab_dir + 'inp_template_lgcabun_ion_lum.txt'  # to run with lgc abn files # input of total ionising luminosity NOT bolometric luminosity
fout = 'inp.txt'
input_quanta = sb99_dir + sb99_model + '/' + sb99_model + '.quanta'
dust_filename = mappings_input_dir + '/Depln_Fe_1.50.txt'

# ----------------reading in Starburst99 output-----------------------------
SB99_age = np.array([float(x.split()[0]) for x in open(input_quanta).readlines()[6:]])
SB99_logQ = np.array([float(x.split()[1]) for x in open(input_quanta).readlines()[6:]])

# --------declaring four 1D arrays over which the model grid would be computed------------------------------------------------
Z_arr = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0])  # in Zsun units; metallicity array of MAPPINGS grid #for outtag _MADtemp
age_arr = np.linspace(0., 10., 11)  # in Myr
lognII_arr = np.linspace(6., 11., 6)  # nII in particles/m^3 # so that log(P/k) is from 4 to 9 (assuming T=1e4 K)
logU_arr = np.linspace(-4., -1., 4)  # dimensionless

# ------------declaring filenames--------------------------------------------
outtag = '_sph_logT' + str(ltemp) + '_mass' + str(mappings_starparticle_mass) + '_MADtemp_ion_lum_from_age' + '_Z' + str(np.min(Z_arr)) + ',' + str(np.max(Z_arr)) + '_age' + str(
    np.min(age_arr)) + ',' + str(np.max(age_arr)) + '_lnII' + str(np.min(lognII_arr)) + ',' + str(
    np.max(lognII_arr)) + '_lU' + str(np.min(logU_arr)) + ',' + str(logU_arr[-1]) + '_4D' # string to be used in file/folder names

# ------------main function--------------------------------------------
if __name__ == '__main__':
    cpu = int(sys.argv[1]) if len(sys.argv) > 1 else int(mproc.cpu_count() / 2)

    # --------------------------------------------------------
    subprocess.call(['mkdir -p ' + mappings_lab_dir + 'results' + outtag], shell=True)
    subprocess.call(['cp make_mappings_grid.py ' + mappings_lab_dir + 'results' + outtag + '/USED_make_mappings_grid.py'], shell=True)

    # ---------------------Parallelised loop--------------------------------------------
    parallel = True
    print('Running in parallel using', cpu, 'cores.')
    print('Grid\tZ\tage\tnII\t<U>\tlogQ0\tOm\tlpok\tRs(pc)\tr_i(pc)\tr_m(pc)\tlUin\truntime(min)')
    m = mproc.Manager()
    q = m.Queue()
    p = mproc.Pool(cpu)
    i = 0
    for Z in Z_arr:
        for age in age_arr:
            for lognII in lognII_arr:
                for logU in logU_arr:
                    i += 1
                    if i < i_start:
                        print('Skipping', i, Z, age, lognII, logU, '...')
                        continue
                    if i > i_end or i > len(Z_arr) * len(age_arr) * len(logU_arr) * len(lognII_arr): break
                    p.apply_async(rungridpoint, (i, Z, age, lognII, logU, parallel, clobber))
    p.close()
    p.join()

    # ------------collating the output grid into one txt file-----------------------------------------
    lines_to_pick = read_linelist(mappings_lab_dir + 'targetlines.txt')
    photgrid = collate_grid(lines_to_pick, outtag)
    # -----------------------------------------------------
    print('Done in %s minutes' % ((time.time() - start_time)/60))
