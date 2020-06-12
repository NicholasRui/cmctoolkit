import numpy as np
import pandas as pd
import scipy.optimize
import scipy.interpolate
import gzip
import time
import os

################################################################################
# DEFINE CGS CONSTANTS
################################################################################

#Universal constants
h, c, k, hbar = 6.6260755e-27, 2.99792458e10, 1.380658e-16, 1.05457266e-27
G, sigma = 6.67259e-8, 5.67051e-5

# Subsets
startype_all = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
startype_star = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
startype_ms = np.array([0, 1])
startype_giant = np.array([2, 3, 4, 5, 6, 7, 8, 9])
startype_wd = np.array([10, 11, 12])
startype_other = np.array([7])
startype_remnant = np.array([10, 11, 12, 13, 14])
startype_bh = np.array([14])

def make_unitdict(convfile):
    """
    Helper function which converts a conversion ratio file into a unit dictionary.
    An example of a conversion file is 'initial.conv.sh'. Function is called in
    Snapshot class.

    Parameters
    ----------
    convfile: list
        specially formatted convfile list wherein each line is an element
    """
    base_dict = {'code': 1, # code units

                 # Fundamental
                    'g': float(convfile[5][12:]), # grams, massunitcgs
                 'msun': float(convfile[7][13:]), # msun, massunitmsun
                   'cm': float(convfile[13][14:]), # cm, lengthunitcgs
                   'pc': float(convfile[15][17:]), # pc, lengthunitparsec
                    's': float(convfile[17][12:]), # s, timeunitcgs
                  'myr': float(convfile[19][13:]), # myr, timeunitsmyr

                 # Stellar
                  's_g': float(convfile[9][13:]), # g stellar, mstarunitcgs
               's_msun': float(convfile[11][14:]), # msun stellar, mstarunitmsun

                 # N-body
                 'nb_s': float(convfile[21][14:]), # s N-body s, nbtimeunitcgs
               'nb_myr': float(convfile[23][15:]), # myr N-body Myr, nbtimeunitsmyr
                }

    custom_dict = {
                 # Custom
                 'nb_km/s': 1e-5 * base_dict['cm'] / base_dict['nb_s'], # km/s
                     'erg': base_dict['g'] * base_dict['cm'] ** 2 / base_dict['s'] ** 2, # erg
                   'erg.s': base_dict['g'] * base_dict['cm'] ** 2 / base_dict['s'], # erg*s (angular momentum)
                   'erg/s': base_dict['g'] * base_dict['cm'] ** 2 / base_dict['s'] ** 3, # erg/s
                    'lsun': 2.599e-34 * base_dict['g'] * base_dict['cm'] ** 2 / base_dict['s'] ** 3, # lsun
                    'rsun': (1 / 6.96e10) * base_dict['cm'], # rsun
                'angstrom': 1e8 * base_dict['cm'], # angstrom
                     'kpc': 1e-3 * base_dict['pc'], # kpc
                    'g/s2': base_dict['g'] / base_dict['s'] ** 2, # g/s^2 (spectral flux unit)
                      'jy': 1e23 * base_dict['g'] / base_dict['s'] ** 2, # jansky
                     'gyr': 1e-3 * base_dict['myr'], # gyr
                      'kg': 1e-3 * base_dict['g'], #kg
                      'Hz': base_dict['s'] ** -1, # Hz (frequency)
                    '1/yr': 31556926.0153 * base_dict['s'] ** -1, # yr^-1 (frequency)
                   'nb_Hz': base_dict['nb_s'] ** -1, # n-body Hz (frequency)

                 # Angular
                     'rad': 1, # radians
                     'deg': 180 / np.pi, # degrees
                  'arcmin': 60 * 180 / np.pi, # arcmin
                  'arcsec': 3600 * 180 / np.pi, # arcsec
                     'mas': 1e3 * 3600 * 180 / np.pi, # milliarcsecond
                  }

    unitdict = {**base_dict, **custom_dict}

    return unitdict

def load_filter(fname):
    """
    Filter function which loads ascii filter functions with two columns:
       1. wavelength[ANGSTROM]
       2. transmission
    Function assumes no header. Also assumes that the last line is empty.

    Parameters
    ----------
    fname: str
        path of filter function

    Returns
    -------
    filt: pd.DataFrame
        filter function table
    """
    # Read filter function
    f = open(fname, 'r')
    text = f.read().split('\n')[:-1] # Assumes last line is empty
    f.close()

    # Convert to a pandas table
    wavelength = np.array([text[ii].split()[0] for ii in range(len(text))])
    transmission = np.array([text[ii].split()[1] for ii in range(len(text))])

    filt = {'wavelength[ANGSTROM]': wavelength.astype(float),
            'transmission': transmission.astype(float)}
    filt = pd.DataFrame(filt)

    return filt

def load_filtertable(fname):
    """
    Filter function which loads ascii filter functions with five columns:
       1. filtname
       2. path
       3. wavelength_mean[ANGSTROM]
       4. wavelength_min[ANGSTROM]
       5. wavelength_max[ANGSTROM]
       6. zp_spectralflux[JY]
    Function assumes no header. Also assumes that the last line is empty.

    Parameters
    ----------
    fname: str
        path of filter table file

    Returns
    -------
    filttable: pd.DataFrame
        table containing information about filter functions
    """
    # Read filter table
    f = open(fname, 'r')
    text = f.read().split('\n')
    f.close()

    # Convert to a pandas table
    filtname = np.array([text[ii].split()[0] for ii in range(len(text))])
    path = np.array([text[ii].split()[1] for ii in range(len(text))])
    wavelength_mean = np.array([text[ii].split()[2] for ii in range(len(text))])
    wavelength_min = np.array([text[ii].split()[3] for ii in range(len(text))])
    wavelength_max = np.array([text[ii].split()[4] for ii in range(len(text))])
    zp_spectralflux = np.array([text[ii].split()[5] for ii in range(len(text))])

    filttable = {'filtname': filtname,
            'path': path,
            'wavelength_mean[ANGSTROM]': wavelength_mean.astype(float),
            'wavelength_min[ANGSTROM]': wavelength_min.astype(float),
            'wavelength_max[ANGSTROM]': wavelength_max.astype(float),
            'zp_spectralflux[JY]': zp_spectralflux.astype(float)}
    filttable = pd.DataFrame(filttable)

    return filttable

def add_mags(mag1, mag2):
    """
    Helper function which adds two magnitudes together along luminosity lines.

    Parameters
    ----------
    mag1: float
        first magnitude being added

    mag2: float
        second magnitude being added

    Returns
    -------
    tot_mag: float
        sum of the magnitudes
    """
    tot_mag = -2.5 * np.log10( 10 ** (-mag1 / 2.5) + 10 ** (-mag2 / 2.5) )
    return tot_mag

def find_t_ms(z, m):
    """
    Helper function for find_MS_TO()
    """
    eta = np.log10(z/0.02)
    a1 = 1.593890e3+2.053038e3*eta+1.231226e3*eta**2.+2.327785e2*eta**3.
    a2 = 2.706708e3+ 1.483131e3*eta+ 5.772723e2*eta**2.+ 7.411230e1*eta**3.
    a3 = 1.466143e2 - 1.048442e2*eta - 6.795374e1*eta**2. - 1.391127e1*eta**3.
    a4 = 4.141960e-2 + 4.564888e-2*eta + 2.958542e-2*eta**2 + 5.571483e-3*eta**3.
    a5 = 3.426349e-1
    a6 = 1.949814e1 + 1.758178*eta - 6.008212*eta**2. - 4.470533*eta**3.
    a7 = 4.903830
    a8 = 5.212154e-2 + 3.166411e-2*eta - 2.750074e-3*eta**2. - 2.271549e-3*eta**3.
    a9 = 1.312179 - 3.294936e-1*eta + 9.231860e-2*eta**2. + 2.610989e-2*eta**3.
    a10 = 8.073972e-1

    m_hook = 1.0185 + 0.16015*eta + 0.0892*eta**2.
    m_HeF = 1.995 + 0.25*eta + 0.087*eta**2.
    m_FGB = 13.048*(z/0.02)**0.06/(1+0.0012*(0.02/z)**1.27)

    t_BGB = (a1+a2*m**4.+a3*m**5.5+m**7.)/(a4*m**2.+a5*m**7.)
    x = np.max([0.95,np.min([0.95-0.03*(eta+0.30103)]),0.99])
    mu = np.max([0.5, 1.0-0.01*np.max([a6/(m**a7), a8+a9/m**a10])])
    t_hook = mu*t_BGB

    t_MS = np.max([t_hook, x*t_BGB])

    return (t_MS)

def find_MS_TO(t, z):
    """
    Iteratively calculate main sequence mass turnoff as a function of cluster age
    and metallicity.
    
    Parameters
    ----------
    t: float
        age of cluster (in Gyr)
    
    z: float
        cluster metallicity
        
    Returns
    -------
    mto: float
        turn-off mass
    """
    t *= 1000
    
    # Make a grid of masses
    m_arr = np.logspace(-2, 3, 1000)
    t_arr = np.array([find_t_ms(z, m) for m in m_arr])
    
    # Interpolate t as a function of m to find turn-off mass
    interp = scipy.interpolate.interp1d(t_arr, m_arr)
    mto = float(interp(t))

    return mto

class Snapshot:
    """
    Snapshot class for snapshot file, usually something like 'initial.snap0137.dat.gz',
    paired alongside conversion file and, preferably, distance and metallicity.

    Parameters
    ----------
    fname: str
        filename of snapshot

    conv: str, dict, or pd.DataFrame
        if str, filename of unitfile (e.g., initial.conv.sh)
        if dict, dictionary of unit conversion factors
        if pd.DataFrame, table corresponding to initial.conv.sh

    age: float
        age of the cluster at the time of the snapshot in Myr

    dist: float (default: None)
        distance to cluster in kpc
        
    z: float (default: None)
        cluster metallicity

    Attributes
    ----------
    data: pd.DataFrame
        snapshot table

    unitdict: dict
        Dictionary containing unit conversion information

    dist: float (None)
        distance to cluster in kpc

    filtertable: pd.DataFrame
        table containing information about all filter for which photometry exists
    """
    def __init__(self, fname, conv, dist=None, z=None):
        self.dist = dist
        self.z = z

        # Read in snapshot as long list where each line is a str
        f = gzip.open(fname,'rb')
        text = str(f.read()).split('\\n')
        f.close()

        # Extract column names
        colrow = text[1].replace('#',' ').replace('  ',' ').split()
        colnames = [ colrow[ii][len(str(ii+1))+1:].replace(':','') for ii in range(len(colrow)) ]

        # Extract snapshot time
        t_snapshot = float(text[0].split('#')[1].split('=')[1].split()[0])

        # Make a list of lists each of which contains the contents of each object
        text = text[2:-1]
        rows = np.array([ np.array(row.split())[:len(colnames)] for row in text ])

        rows[np.where(rows == 'na')] = 'nan'
        rows = rows.astype(float)

        # Build a dictionary and cast to pandas DataFrame object
        table = {}
        for ii in range(len(colnames)):
            table[colnames[ii]] = rows[:, ii]

        self.data = pd.DataFrame(table)

        # Now, build conversion dictionary
        if (type(conv) == str) or (type(conv) == pd.DataFrame):
            if type(conv) == str:
                f = open(conv, 'r')
                convfile = f.read().split('\n')
                f.close()

            # Produce unit dictionary
            self.unitdict = make_unitdict(convfile)

        elif type(conv) == dict:
            self.unitdict = conv

        else:
            raise ValueError('convfile must be either str or pd.DataFrame')

        # Also read in the time of the snapshot (code units) and convert to gyr
        self.age = self.convert_units(t_snapshot, 'code', 'gyr')

        self.filtertable = pd.DataFrame({'filtname': [],
                                             'path': [],
                        'wavelength_mean[ANGSTROM]': [],
                         'wavelength_min[ANGSTROM]': [],
                         'wavelength_max[ANGSTROM]': [],
                              'zp_spectralflux[JY]': []})

    def convert_units(self, arr, in_unit='code', out_unit='code'):
        """
        Converts an array from CODE units to 'unit' using conversion factors specified
        in unitfile.

        Note: 's_' preceding an out_unit refers to 'stellar' quantities. 'nb_' refers
        to n-body units. Without these tags, it is presumed otherwise.

        Parameters
        ----------
        arr: array-like
            array to be converted

        in_unit: str (default: 'code')
            unit from which arr is to be converted

        out_unit: str (default: 'code')
            unit to which arr is to be converted

        Returns
        -------
        converted: array-like
            converted array
        """
        # Make sure both specified units are good
        if in_unit in self.unitdict.keys():
            ValueError('{} is not a recognized unit.'.format(in_unit))
        elif out_unit in self.unitdict.keys():
            ValueError('{} is not a recognized unit.'.format(out_unit))

        # Converted array
        converted = self.unitdict[out_unit] * arr / self.unitdict[in_unit]

        return converted

    def make_cuts(self, min_mass=None, max_mass=None, dmin=None, dmax=None, max_lum=None, fluxdict=None):
        """
        Helper method to return a boolean array where a given set of cuts are
        satisfied.

        Parameters
        ----------
        min_mass: float (default: None)
            If specified, only include stars above this mass

        min_mass: float (default: None)
            If specified, only include stars below this mass

        dmin: float (default: None)
            If specified, only include stars outside this projected radius

        dmax: float (default: None)
            If specified, only include stars inside this projected radius

        max_lum: float (default: None)
            IF specified, only include stars below this luminosity [LSUN]

        fluxdict: dict (default: None)
            If specified, makes upper and lower (observed) magnitude cuts in certain filters
            Should be in the following format:
            {'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

            If you don't have a cut, put None

        Returns
        -------
        good: array-like of bool
            boolean array specifying where cuts are satisfied
        """
        # At the beginning, nothing is cut
        good = np.ones(len(self.data)).astype(bool)

        single = (self.data['binflag'] != 1)
        binary = (self.data['binflag'] == 1)

        # Mass cuts
        if min_mass is not None: # Pretend binaries are a single star with double mass
            good = good & ( ( (self.data['m[MSUN]'] > min_mass)                          & single ) |
                            ( (self.data['m0[MSUN]'] + self.data['m1[MSUN]'] > min_mass) & binary ) )
        if max_mass is not None:
            good = good & ( ( (self.data['m[MSUN]'] < max_mass)                          & single ) |
                            ( (self.data['m0[MSUN]'] + self.data['m1[MSUN]'] < max_mass) & binary ) )

        # Cuts on projected radius (in d)
        if (dmin is not None) | (dmax is not None):
            if 'd[PC]' not in self.data.keys():
                self.make_2d_projection()

            d_pc_arr = np.array(self.data['d[PC]'])

            if dmin is not None:
                good = good & ( d_pc_arr > dmin )
            if dmax is not None:
                good = good & ( d_pc_arr < dmax )
                
        # Cut on luminosity
        if max_lum is not None:
            good = good & ( ( single & (self.data['luminosity[LSUN]'] < max_lum) ) |
                            ( binary & (self.data['bin_star_lum0[LSUN]'] + self.data['bin_star_lum1[LSUN]'] < max_lum) ) )

        # Make sure all of the filters are actually there
        if fluxdict is not None:
            if not np.in1d(np.array(fluxdict.keys()), self.filtertable['filtname']).all():
                raise ValueError('One or more filters specified do not have photometry in this table.')

            # Cut on (observed) magnitudes
            for filtname in fluxdict.keys():
                faint_cut = fluxdict[filtname][0]
                bright_cut = fluxdict[filtname][1]

                colname = 'obsMag_{}'.format(filtname)
                bincolname0 = 'bin_obsMag0_{}'.format(filtname)
                bincolname1 = 'bin_obsMag1_{}'.format(filtname)

                if faint_cut is not None:
                    good = good & ( ( (self.data[colname] < faint_cut)                                       & single ) |
                                    ( (add_mags(self.data[bincolname0], self.data[bincolname1]) < faint_cut) & binary ) )

                if bright_cut is not None:
                    good = good & ( ( (self.data[colname] > bright_cut)                                       & single ) |
                                    ( (add_mags(self.data[bincolname0], self.data[bincolname1]) > bright_cut) & binary ) )

        return good

    def add_photometry(self, filttable):
        """
        Function which, assuming black-body behavior, assigns observed magnitudes
        to stars in desired filters

        For each filter, adds the following columns (# = filter name):
        absMag_#: absolute magnitude in filter # for single (np.nan for binary or black hole)
        bin_absMag0_#: absolute magnitude in filter # for first star in binary (np.nan for single or black hole)
        bin_absMag1_#: absolute magnitude in filter # for second star in binary (np.nan for single or black hole)
        tot_absMag_#: total magnitude in filter #, same as absMag_# for singles and is the magnitude sum of a binary pair if binary

        If distance is given, also add:
        obsMag_#: observed magnitude in filter # for single (np.nan for binary or black hole)
        bin_obsMag0_#: observed magnitude in filter # for first star in binary (np.nan for single or black hole)
        bin_obsMag1_#: observed magnitude in filter # for second star in binary (np.nan for single or black hole)
        tot_obsMag_#: total observed magnitude in filter #, same as absMag_# for singles and is the magnitude sum of a binary pair if binary

        Parameters
        ----------
        filttable: str or pd.DataFrame
            if str, path to filter table
            if pd.DataFrame, table containing information about filters (see function: load_filtertable)

        Returns
        -------
        none
        """
        # If Teff has not been calculated, calculate it
        if 'Teff[K]' not in self.data.keys():
            self.calc_Teff()

        if type(filttable) == str:
            filttable = load_filtertable(filttable)
        elif type(filttable) != pd.DataFrame:
            raise ValueError('filttable must be either str or pd.DataFrame')

        # Read filter files
        filtnames = np.array(filttable['filtname'])
        filtfuncs = [load_filter(filttable.loc[ii,'path']) for ii in range(len(filttable))]

        wavelength_mean_angstrom = np.array(filttable['wavelength_mean[ANGSTROM]'])
        wavelength_min_angstrom = np.array(filttable['wavelength_min[ANGSTROM]'])
        wavelength_max_angstrom = np.array(filttable['wavelength_max[ANGSTROM]'])
        wavelength_min = self.convert_units(wavelength_min_angstrom, 'angstrom', 'cm')
        wavelength_max = self.convert_units(wavelength_max_angstrom, 'angstrom', 'cm')
        passband_hz = c / wavelength_min - c / wavelength_max

        zp_spectralflux_jy = np.array(filttable['zp_spectralflux[JY]'])
        zp_spectralflux = self.convert_units(zp_spectralflux_jy, 'jy', 'g/s2')

        if self.dist is not None:
            distance_modulus = 5 * np.log10(self.dist / 0.01)

        # Calculate magnitudes
        for ii in range(len(filtnames)):
            self.data['absMag_' + filtnames[ii]] = np.nan * np.ones(len(self.data))
            self.data['bin_absMag0_' + filtnames[ii]] = np.nan * np.ones(len(self.data))
            self.data['bin_absMag1_' + filtnames[ii]] = np.nan * np.ones(len(self.data))

            # Get filter function information
            wavelength_cm = np.array(self.convert_units(filtfuncs[ii]['wavelength[ANGSTROM]'], 'angstrom', 'cm'))
            transmission = np.array(filtfuncs[ii]['transmission'])

            # Use trapezoid rule to evaluate integral of filtfunc * Planck distribution
            Teff_K = self.data.loc[(self.data['binflag'] != 1) & (self.data['startype'] != 14), 'Teff[K]']
            planck = 2 * h * c ** 2 / (wavelength_cm.reshape((1, wavelength_cm.size)) ** 5 * (np.exp(h * c / (k * np.outer(Teff_K, wavelength_cm))) - 1))

            planck_weighted = planck * transmission.reshape((1, transmission.size))
            integrated_planck_weighted = np.sum(0.5 * (planck_weighted[:,1:] + planck_weighted[:,:-1]) * (wavelength_cm[1:] - wavelength_cm[:-1]), axis=1)

            rad_rsun = self.data.loc[(self.data['binflag'] != 1) & (self.data['startype'] != 14), 'radius[RSUN]']
            rad_cm = self.convert_units(rad_rsun, 'rsun', 'cm')
            luminosity_cgs = 4 * np.pi ** 2 * rad_cm ** 2 * integrated_planck_weighted

            spectral_lum = luminosity_cgs / (4 * np.pi * self.convert_units(10, 'pc', 'cm') ** 2 * passband_hz[ii])

            # Calculate magnitudes (exclude black holes)
            self.data.loc[(self.data['binflag'] != 1) & (self.data['startype'] != 14), 'absMag_' + filtnames[ii]] = -2.5 * np.log10(spectral_lum / zp_spectralflux[ii])

            if self.dist is not None:
                self.data['obsMag_' + filtnames[ii]] = self.data['absMag_' + filtnames[ii]]
                self.data['obsMag_' + filtnames[ii]] += distance_modulus

            # Repeat this process for the first star in each binary
            Teff0_K = self.data.loc[(self.data['binflag'] == 1) & (self.data['bin_startype0'] != 14), 'bin_Teff0[K]']
            planck0 = 2 * h * c ** 2 / (wavelength_cm.reshape((1, wavelength_cm.size)) ** 5 * (np.exp(h * c / (k * np.outer(Teff0_K, wavelength_cm))) - 1))

            planck_weighted0 = planck0 * transmission.reshape((1, transmission.size))
            integrated_planck_weighted0 = np.sum(0.5 * (planck_weighted0[:,1:] + planck_weighted0[:,:-1]) * (wavelength_cm[1:] - wavelength_cm[:-1]), axis=1)

            rad0_rsun = self.data.loc[(self.data['binflag'] == 1) & (self.data['bin_startype0'] != 14), 'bin_star_radius0[RSUN]']
            rad0_cm = self.convert_units(rad0_rsun, 'rsun', 'cm')
            luminosity0_cgs = 4 * np.pi ** 2 * rad0_cm ** 2 * integrated_planck_weighted0

            spectral_lum0 = luminosity0_cgs / (4 * np.pi * self.convert_units(10, 'pc', 'cm') ** 2 * passband_hz[ii])

            # Calculate magnitudes
            self.data.loc[(self.data['binflag'] == 1) & (self.data['bin_startype0'] != 14), 'bin_absMag0_' + filtnames[ii]] = -2.5 * np.log10(spectral_lum0 / zp_spectralflux[ii])

            if self.dist is not None:
                self.data['bin_obsMag0_' + filtnames[ii]] = self.data['bin_absMag0_' + filtnames[ii]]
                self.data['bin_obsMag0_' + filtnames[ii]] += distance_modulus

            # Repeat this process for the second star in each binary
            Teff1_K = self.data.loc[(self.data['binflag'] == 1) & (self.data['bin_startype1'] != 14), 'bin_Teff1[K]']
            planck1 = 2 * h * c ** 2 / (wavelength_cm.reshape((1, wavelength_cm.size)) ** 5 * (np.exp(h * c / (k * np.outer(Teff1_K, wavelength_cm))) - 1))

            planck_weighted1 = planck1 * transmission.reshape((1, transmission.size))
            integrated_planck_weighted1 = np.sum(0.5 * (planck_weighted1[:,1:] + planck_weighted1[:,:-1]) * (wavelength_cm[1:] - wavelength_cm[:-1]), axis=1)

            rad1_rsun = self.data.loc[(self.data['binflag'] == 1) & (self.data['bin_startype1'] != 14), 'bin_star_radius1[RSUN]']
            rad1_cm = self.convert_units(rad1_rsun, 'rsun', 'cm')
            luminosity1_cgs = 4 * np.pi ** 2 * rad1_cm ** 2 * integrated_planck_weighted1

            spectral_lum1 = luminosity1_cgs / (4 * np.pi * self.convert_units(10, 'pc', 'cm') ** 2 * passband_hz[ii])

            # Calculate magnitudes
            self.data.loc[(self.data['binflag'] == 1) & (self.data['bin_startype1'] != 14), 'bin_absMag1_' + filtnames[ii]] = -2.5 * np.log10(spectral_lum1 / zp_spectralflux[ii])

            if self.dist is not None:
                self.data['bin_obsMag1_' + filtnames[ii]] = self.data['bin_absMag1_' + filtnames[ii]]
                self.data['bin_obsMag1_' + filtnames[ii]] += distance_modulus
                
            # Add total magnitude columns together
            self.data['tot_absMag_' + filtnames[ii]] = np.nan * np.ones(len(self.data))
            
            good_single = (self.data['binflag'] != 1) & (self.data['startype'] != 14)
            self.data.loc[good_single, 'tot_absMag_' + filtnames[ii]] = self.data.loc[good_single, 'absMag_' + filtnames[ii]]
            
            good_binary = (self.data['binflag'] == 1) & (self.data['bin_startype0'] != 14) & (self.data['bin_startype1'] != 14)
            self.data.loc[good_binary, 'tot_absMag_' + filtnames[ii]] = add_mags(self.data.loc[good_binary, 'bin_absMag0_' + filtnames[ii]],
                                                                                 self.data.loc[good_binary, 'bin_absMag1_' + filtnames[ii]])
                                                                                 
            good0_bad1 = (self.data['binflag'] == 1) & (self.data['bin_startype0'] != 14) & (self.data['bin_startype1'] == 14)
            self.data.loc[good0_bad1, 'tot_absMag_' + filtnames[ii]] = self.data.loc[good0_bad1, 'bin_absMag0_' + filtnames[ii]]
            
            good1_bad0 = (self.data['binflag'] == 1) & (self.data['bin_startype0'] == 14) & (self.data['bin_startype1'] != 14)
            self.data.loc[good1_bad0, 'tot_absMag_' + filtnames[ii]] = self.data.loc[good1_bad0, 'bin_absMag1_' + filtnames[ii]]
            
            if self.dist is not None:
                self.data['tot_obsMag_' + filtnames[ii]] = self.data['tot_absMag_' + filtnames[ii]]
                self.data['tot_obsMag_' + filtnames[ii]] += distance_modulus

            # Add filter to filtertable
            filterrow = pd.DataFrame({'filtname': [filtnames[ii]],
                                          'path': [filttable.loc[ii,'path']],
                     'wavelength_mean[ANGSTROM]': [wavelength_mean_angstrom[ii]],
                      'wavelength_min[ANGSTROM]': [wavelength_min_angstrom[ii]],
                      'wavelength_max[ANGSTROM]': [wavelength_max_angstrom[ii]],
                           'zp_spectralflux[JY]': [zp_spectralflux_jy[ii]],
                                     })

            self.filtertable = self.filtertable.append(filterrow, ignore_index=True)

    def make_2d_projection(self, seed=0):
        """
        Appends to a snapshot table a column projecting stellar positions onto the
        2-dimensional plane of the sky (with randomly generated angles). This adds
        a column 'd' which is the projected radius.

        Adds the following columns:
        d : Projected radial distance (code units)
        x, y, z: Projected x, y, z coordinates (code units)
        vx, vy, vz: Projected x, y, z components of velocity (code units)
        vd: Projected radial component of velocity (code units)
        va: Projected azimuthal component of velocity (code units)
        r[PC] : Radial distance (pc)
        d[PC] : Projected radial distance (pc)
        x[PC], y[PC], z[PC]: Projected x, y, z coordinates (pc)
        vx[KM/S], vy[KM/S], vz[KM/S]: Projected x, y, z components of velocity (km/s)
        vd[KM/S]: Projected radial component of velocity (km/s)
        va[KM/S]: Projected azimuthal component of velocity (km/s)

        Parameters
        ----------
        seed: int (default: 0)
            random seed
        """
        r_arr = self.data['r']
        vr_arr = self.data['vr']
        vt_arr = self.data['vt']

        # Generate needed random numbers
        np.random.seed(seed)
        cosTheta = np.random.uniform(-1, 1, len(r_arr)) # Cosine of polar angle
        sinTheta = np.sqrt(1 - cosTheta ** 2)
        Phi = np.random.uniform(0, 2*np.pi, len(r_arr)) # Azimuthal angle
        vAngle = np.random.uniform(0, 2*np.pi, len(r_arr)) # Additional angle for tangential velocities

        # Project positions and convert to pc
        d_arr = r_arr * sinTheta
        x_arr = d_arr * np.cos(Phi)
        y_arr = d_arr * np.sin(Phi)
        z_arr = r_arr * cosTheta

        r_pc_arr = self.convert_units(r_arr, 'code', 'pc')
        d_pc_arr = self.convert_units(d_arr, 'code', 'pc')
        x_pc_arr = self.convert_units(x_arr, 'code', 'pc')
        y_pc_arr = self.convert_units(y_arr, 'code', 'pc')
        z_pc_arr = self.convert_units(z_arr, 'code', 'pc')

        # Project velocities and convert to km/s
        abs_v = np.hypot(vr_arr, vt_arr)
        dotTheta = np.cos(vAngle) * vt_arr / r_arr
        dotPhi = np.sin(vAngle) * vt_arr / (r_arr * sinTheta)

        vx_arr = vr_arr * sinTheta * np.cos(Phi) - r_arr * dotPhi * sinTheta * np.sin(Phi) - r_arr * dotTheta * cosTheta * np.cos(Phi)
        vy_arr = vr_arr * sinTheta * np.sin(Phi) + r_arr * dotPhi * sinTheta * np.cos(Phi) - r_arr * dotTheta * cosTheta * np.sin(Phi)
        vz_arr = vr_arr * cosTheta + r_arr * dotTheta * sinTheta
        vd_arr = vx_arr * np.cos(Phi) + vy_arr * np.sin(Phi) # projected radial component of velocity
        va_arr = -vx_arr * np.sin(Phi) + vy_arr * np.cos(Phi) # azimuthal component of velocity (line-of-sight tangential)

        vr_kms_arr = self.convert_units(vr_arr, 'code', 'nb_km/s')
        vx_kms_arr = self.convert_units(vx_arr, 'code', 'nb_km/s')
        vy_kms_arr = self.convert_units(vy_arr, 'code', 'nb_km/s')
        vz_kms_arr = self.convert_units(vz_arr, 'code', 'nb_km/s')
        vd_kms_arr = self.convert_units(vd_arr, 'code', 'nb_km/s')
        va_kms_arr = self.convert_units(va_arr, 'code', 'nb_km/s')

        # Add new columns to table
        self.data['d'] = d_arr
        self.data['x'], self.data['y'], self.data['z'] = x_arr, y_arr, z_arr
        self.data['vx'], self.data['vy'], self.data['vz'] = vx_arr, vy_arr, vz_arr
        self.data['vd'], self.data['va'] = vd_arr, va_arr

        self.data['r[PC]'] = r_pc_arr
        self.data['d[PC]'] = d_pc_arr
        self.data['x[PC]'], self.data['y[PC]'], self.data['z[PC]'] = x_pc_arr, y_pc_arr, z_pc_arr
        self.data['vx[KM/S]'], self.data['vy[KM/S]'], self.data['vz[KM/S]'] = vx_kms_arr, vy_kms_arr, vz_kms_arr
        self.data['vd[KM/S]'], self.data['va[KM/S]'] = vd_kms_arr, va_kms_arr

    def calc_Teff(self):
        """
        Calculate the effective temperature Teff for every (non-BH) star in the catalog.

        Adds the following columns:
        Teff[K] : Effective temperature of a single star, np.nan for a binary or black hole
        bin_Teff0[K] : Effective temperature for first star in a binary, np.nan for a single or black hole
        bin_Teff1[K] : Effective temperature for second star in a binary, np.nan for a single or black hole

        Parameters
        ----------
        none
        """
        # Add columns for Teff
        self.data['Teff[K]'] = np.nan * np.ones(len(self.data))
        self.data['bin_Teff0[K]'] = np.nan * np.ones(len(self.data))
        self.data['bin_Teff1[K]'] = np.nan * np.ones(len(self.data))

        # First, calculate Teff for non-BH singles
        single = (self.data['binflag'] != 1) & (self.data['startype'] != 14)
        lum_lsun = self.data.loc[single, 'luminosity[LSUN]']
        rad_rsun = self.data.loc[single, 'radius[RSUN]']

        lum = self.convert_units(lum_lsun, 'lsun', 'erg/s')
        rad = self.convert_units(rad_rsun, 'rsun', 'cm')

        self.data.loc[single, 'Teff[K]'] = (lum / (4 * np.pi * rad ** 2 * sigma)) ** 0.25

        # Next, calculate Teff for non-BH doubles... start with the first of the pair
        binary0 = (self.data['binflag'] == 1) & (self.data['bin_startype0'] != 14)
        lum0_lsun = self.data.loc[binary0, 'bin_star_lum0[LSUN]']
        rad0_rsun = self.data.loc[binary0, 'bin_star_radius0[RSUN]']

        lum0 = self.convert_units(lum0_lsun, 'lsun', 'erg/s')
        rad0 = self.convert_units(rad0_rsun, 'rsun', 'cm')

        self.data.loc[(self.data['binflag'] == 1) & (self.data['bin_startype0'] != 14), 'bin_Teff0[K]'] = (lum0 / (4 * np.pi * rad0 ** 2 * sigma)) ** 0.25

        # Same as above but for the second star in each binary pair
        binary1 = (self.data['binflag'] == 1) & (self.data['bin_startype1'] != 14)
        lum1_lsun = self.data.loc[binary1, 'bin_star_lum1[LSUN]']
        rad1_rsun = self.data.loc[binary1, 'bin_star_radius1[RSUN]']

        lum1 = self.convert_units(lum1_lsun, 'lsun', 'erg/s')
        rad1 = self.convert_units(rad1_rsun, 'rsun', 'cm')

        self.data.loc[binary1, 'bin_Teff1[K]'] = (lum1 / (4 * np.pi * rad1 ** 2 * sigma)) ** 0.25

    def calc_surface_gravity(self):
        """
        Adds surface gravity columns to snapshot table.

        Adds the following columns:
        g[CM/S] : Surface gravity of a single star, np.nan for a binary
        bin_g0[CM/S] : Surface gravity for first star in a binary, np.nan for a single
        bin_g1[CM/S] : Surface gravity for second star in a binary, np.nan for a single

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        # Add columns for g
        self.data['g[CM/S2]'] = np.nan * np.ones(len(self.data))
        self.data['bin_g0[CM/S2]'] = np.nan * np.ones(len(self.data))
        self.data['bin_g1[CM/S2]'] = np.nan * np.ones(len(self.data))

        # Add surface gravity for single
        mass_msun = self.data.loc[self.data['binflag'] != 1, 'm[MSUN]']
        mass_g = self.convert_units(mass_msun, 'msun', 'g')
        rad_rsun = self.data.loc[self.data['binflag'] != 1, 'radius[RSUN]']
        rad_cm = self.convert_units(rad_rsun, 'rsun', 'cm')

        self.data.loc[self.data['binflag'] != 1, 'g[CM/S2]'] = G * mass_g / rad_cm ** 2

        # Add surface gravity for first binary
        mass0_msun = self.data.loc[self.data['binflag'] != 1, 'm0[MSUN]']
        mass0_g = self.convert_units(mass0_msun, 'msun', 'g')
        rad0_rsun = self.data.loc[self.data['binflag'] != 1, 'bin_star_radius0[RSUN]']
        rad0_cm = self.convert_units(rad0_rsun, 'rsun', 'cm')

        self.data.loc[self.data['binflag'] == 1, 'g0[CM/S2]'] = G * mass0_g / rad0_cm ** 2

        # Add surface gravity for second binary
        mass1_msun = self.data.loc[self.data['binflag'] != 1, 'm1[MSUN]']
        mass1_g = self.convert_units(mass1_msun, 'msun', 'g')
        rad1_rsun = self.data.loc[self.data['binflag'] != 1, 'bin_star_radius1[RSUN]']
        rad1_cm = self.convert_units(rad1_rsun, 'rsun', 'cm')

        self.data.loc[self.data['binflag'] == 1, 'g1[CM/S2]'] = G * mass1_g / rad1_cm ** 2

    def make_spatial_density_profile(self, bin_edges=None, min_mass=None, max_mass=None, fluxdict=None, startypes=startype_star):
        """
        Creates a spatial density profile

        Parameters
        ----------
        bin_edges: array-like
            Bin edges of radial density profile (if None, make 100 bins from min to max, logarithmic spacing)

        min_mass: float (default: None)
            If specified, only include stars above this mass

        min_mass: float (default: None)
            If specified, only include stars below this mass

        fluxdict: dict (default: None)
            If specified, makes upper and lower (observed) magnitude cuts in certain filters
            Should be in the following format:
            {'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

            If you don't have a cut, put None

        startypes: array-like (default: startype_star)
            If specified, only include startypes in this list

        Returns
        -------
        bin_edges: array-like
            Bin edges

        profile: array-like
            Number density in each annulus

        e_profile: array-like
            Poisson error in number density in each annulus
        """
        if 'd[PC]' not in self.data.keys():
            self.make_2d_projection()

        # If bin_edges is not specified, use default
        if bin_edges is None:
            bin_edges = np.logspace( np.log10(np.min(self.data['d[PC]'])), np.log10(np.max(self.data['d[PC]'])), 100 )

        # Read columns and make cuts
        startype_arr = self.data['startype']
        bin_startype0_arr = self.data['bin_startype0']
        bin_startype1_arr = self.data['bin_startype1']

        good = self.make_cuts(min_mass=min_mass, max_mass=max_mass, fluxdict=fluxdict) # make cuts
        good = good & ( ((self.data['binflag'] != 1) & np.in1d(startype_arr, startypes))
                      | ((self.data['binflag'] == 1) & (np.in1d(bin_startype0_arr, startypes) | np.in1d(bin_startype1_arr, startypes)) ) )
        mass_arr = np.array(self.data.loc[good, 'm[MSUN]'])
        d_pc_arr = np.array(self.data.loc[good, 'd[PC]'])

        count, _ = np.histogram(d_pc_arr, bin_edges)

        # Create spatial density profile by binning and dividing by annulus area
        area = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)
        profile = count / area
        e_profile = np.sqrt(count) / area

        return bin_edges, profile, e_profile

    def velocity_dispersion(self, min_mass=None, max_mass=None, dmin=None, dmax=None, startypes=startype_star):
        """
        Calculate velocity dispersion in km/s.

        Parameters
        ----------
        min_mass: float (default: None)
            If specified, only include stars above this mass

        min_mass: float (default: None)
            If specified, only include stars below this mass

        dmin: float (default: None)
            If specified, only include stars outside this projected radius

        dmax: float (default: None)
            If specified, only include stars inside this projected radius

        startypes: array-like (default: startype_star)
            If specified, only include startypes in this list

        Returns
        -------
        veldisp: float
            Velocity dispersion (km/s)

        e_veldisp: float
            Uncertainty in velocity dispersion (km/s)
        """
        if dmin == None:
            dmin = 0
        if dmax == None:
            dmax = np.inf

        _, veldisp, e_veldisp = self.make_velocity_dispersion_profile(bin_edges=[dmin, dmax], min_mass=min_mass, max_mass=max_mass, startypes=startypes)

        veldisp = veldisp[0]
        e_veldisp = e_veldisp[0]

        return veldisp, e_veldisp

    def make_velocity_dispersion_profile(self, bin_edges=None, min_mass=None, max_mass=None, fluxdict=None, startypes=startype_star):
        """
        Calculate velocity dispersion in km/s.

        Parameters
        ----------
        bin_edges: array-like (default: None)
            Bin edges of mass function (if None, make 100 bins from min to max, logarithmic spacing)

        min_mass: float (default: None)
            If specified, only include stars above this mass

        min_mass: float (default: None)
            If specified, only include stars below this mass

        fluxdict: dict (default: None)
            If specified, makes upper and lower (observed) magnitude cuts in certain filters
            Should be in the following format:
            {'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

            If you don't have a cut, put None

        startypes: array-like (default: startype_star)
            If specified, only include startypes in this list

        Returns
        -------
        bin_edges: array-like
            Bin edges of mass function (pc)

        veldisp_profile: array-like
            Velocity dispersion profile (km/s)

        e_veldisp_profile: array-like
            Uncertainty in velocity dispersion profile (km/s)
        """
        if 'd[PC]' not in self.data.keys():
            self.make_2d_projection()

        # Define cut and find relevant arrays, only using relevant startypes
        # As long as one of a binary pair is a good startype, include it
        startype_arr = self.data['startype']
        bin_startype0_arr = self.data['bin_startype0']
        bin_startype1_arr = self.data['bin_startype1']

        good = self.make_cuts(min_mass=min_mass, max_mass=max_mass, fluxdict=fluxdict)
        good = good & ( ((self.data['binflag'] != 1) & np.in1d(startype_arr, startypes))
                      | ((self.data['binflag'] == 1) & (np.in1d(bin_startype0_arr, startypes) | np.in1d(bin_startype1_arr, startypes)) ) )
        mass_arr = np.array(self.data.loc[good, 'm[MSUN]'])
        d_pc_arr = np.array(self.data.loc[good, 'd[PC]'])
        v_kms_arr = np.array(np.hypot(self.data.loc[good, 'vx[KM/S]'], np.hypot(self.data.loc[good, 'vy[KM/S]'], self.data.loc[good, 'vz[KM/S]'])))

        # If bin_edges is not specified, use default
        if bin_edges is None:
            bin_edges = np.logspace( np.log10(np.min(self.data['d[PC]'])), np.log10(np.max(self.data['d[PC]'])), 100 )

        # Calculate velocity dispersion profile
        hist, _ = np.histogram(d_pc_arr, bin_edges)
        veldisp_profile = np.array([ np.sqrt(np.average(v_kms_arr[np.where( (d_pc_arr >= bin_edges[ii]) & (d_pc_arr < bin_edges[ii+1]) )] ** 2))
                                  for ii in range(len(bin_edges) - 1) ]) / np.sqrt(3)

        e_veldisp_profile = veldisp_profile / np.sqrt(2 * hist)

        return bin_edges, veldisp_profile, e_veldisp_profile

    def make_mass_function(self, bin_edges=None, dmin=None, dmax=None, fluxdict=None, startypes=startype_star):
        """
        Creates a mass function

        Parameters
        ----------
        bin_edges: array-like
            Bin edges of mass function (if None, make 100 bins from min to max, logarithmic spacing)

        dmin: float (default: None)
            If specified, only include stars outside this projected radius

        dmax: float (default: None)
            If specified, only include stars inside this projected radius

        fluxdict: dict (default: None)
            If specified, makes upper and lower (observed) magnitude cuts in certain filters
            Should be in the following format:
            {'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

            If you don't have a cut, put None

        startypes: array-like (default: startype_star)
            If specified, only include startypes in this list

        Returns
        -------
        bin_edges: array-like
            Bin edges

        mf: array-like
            Mass function

        e_mf: array-like
            Mass function uncertainty
        """
        # If bin_edges is not specified, use default
        if bin_edges is None:
            bin_edges = np.logspace( np.log10(np.min(self.data['m[MSUN]'])), np.log10(np.max(self.data['m[MSUN]'])), 100 )

        # Make relevant cuts
        startype_arr = self.data['startype']
        bin_startype0_arr = self.data['bin_startype0']
        bin_startype1_arr = self.data['bin_startype1']

        good = self.make_cuts(dmin=dmin, dmax=dmax, fluxdict=fluxdict)
        
        good = good & ( ((self.data['binflag'] != 1) & np.in1d(startype_arr, startypes))
                      | ((self.data['binflag'] == 1) & (np.in1d(bin_startype0_arr, startypes) | np.in1d(bin_startype1_arr, startypes)) ) )

        mass_arr = np.array(self.data['m[MSUN]'])

        binary = np.array(self.data['binflag'] == 1)
        good_both_bin = np.array(np.in1d(bin_startype0_arr, startypes) & np.in1d(bin_startype1_arr, startypes))
        good_bin0 = np.array(np.in1d(bin_startype0_arr, startypes) & ~np.in1d(bin_startype1_arr, startypes))
        good_bin1 = np.array(~np.in1d(bin_startype0_arr, startypes) & np.in1d(bin_startype1_arr, startypes))
        
        mass_arr[good & binary & good_both_bin] = np.array(self.data.loc[good & binary & good_both_bin, 'm0[MSUN]']) + np.array(self.data.loc[good & binary & good_both_bin, 'm1[MSUN]'])
        mass_arr[good & binary & good_bin0] = np.array(self.data.loc[good & binary & good_bin0, 'm0[MSUN]'])
        mass_arr[good & binary & good_bin1] = np.array(self.data.loc[good & binary & good_bin1, 'm1[MSUN]'])
        
        mass_arr = mass_arr[good]

        # Obtain mass function
        count, _ = np.histogram(mass_arr, bin_edges)
        dm = bin_edges[1:] - bin_edges[:-1]

        mf = count / dm
        e_mf = np.sqrt(count) / dm

        return mf, e_mf

    def fit_mass_function_slope(self, init_guess=2.35, min_mass=None, max_mass=None, dmin=None, dmax=None, fluxdict=None, startypes=startype_star): # SCIPY DEPENDENCY
        """
        Fits the mass function slope in a bin-independent way

        Parameters
        ----------
        init_guess: float (default: 2.35)
            initial guess for the power law slope (if unspecified, guess Salpeter)

        min_mass: float (default: None)
            If specified, only include stars above this mass

        max_mass: float (default: None)
            If specified, only include stars below this mass

        dmin: float (default: None)
            If specified, only include stars outside this projected radius

        dmax: float (default: None)
            If specified, only include stars inside this projected radius

        fluxdict: dict (default: None)
            If specified, makes upper and lower (observed) magnitude cuts in certain filters
            Should be in the following format:
            {'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

            If you don't have a cut, put None

        startypes: array-like (default: startype_star)
            If specified, only include startypes in this list

        Returns
        -------
        alpha: float
            Converged value for mass function slope (if failed fit, returns np.nan)
        """
        # Make relevant cuts
        startype_arr = self.data['startype']
        bin_startype0_arr = self.data['bin_startype0']
        bin_startype1_arr = self.data['bin_startype1']

        good = self.make_cuts(min_mass=min_mass, max_mass=max_mass, dmin=dmin, dmax=dmax, fluxdict=fluxdict)
        good = good & ( ((self.data['binflag'] != 1) & np.in1d(startype_arr, startypes))
                      | ((self.data['binflag'] == 1) & (np.in1d(bin_startype0_arr, startypes) | np.in1d(bin_startype1_arr, startypes)) ) )
        mass_arr = np.array(self.data.loc[good, 'm[MSUN]'])

        # Define -1 * loglikelihood function, we wish to minimize this likelihood function
        # (in other words, minimize -1 * loglikelihood)
        if min_mass == None:
            min_mass = np.min(mass_arr)
        if max_mass == None:
            max_mass = np.inf

        N = len(mass_arr)
        minusloglike = lambda alpha: -1 * ( N * np.log((1 - alpha) / (max_mass ** (1-alpha) - min_mass ** (1-alpha))) - alpha * np.sum(np.log(mass_arr)) )

        # Perform fit
        res = scipy.optimize.minimize(minusloglike, init_guess)

        if res.success:
            alpha = res.x[0]
        else:
            alpha = np.nan

        return alpha

    def make_smoothed_number_profile(self, bins=80, min_mass=None, max_mass=None, fluxdict=None, startypes=startype_star, min_logr=-1.5):
        """
        Creates smoothed number density profile by smearing out stars probabilistically.

        Parameters
        ----------
        bins: int (default: 80)
            number of bins used for number density profile

        min_mass: float (default: None)
            If specified, only include stars above this mass

        max_mass: float (default: None)
            If specified, only include stars below this mass

        fluxdict: dict (default: None)
            If specified, makes upper and lower (observed) magnitude cuts in certain filters
            Should be in the following format:
            {'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

            If you don't have a cut, put None

        startypes: array-like (default: startype_star)
            If specified, only include startypes in this list

        min_logr: float (default: -1.5)
            Minimum logarithmic radius in parsec

        Returns
        -------
        bin_center: array-like
            radial points at which profile is evaluated (in pc)

        profile: array-like
            array of number densities

        e_profile: array-like
            uncertainty in number density
        """
        # Define cut and find relevant arrays, only using relevant startypes
        # As long as one of a binary pair is a good startype, include it
        startype_arr = self.data['startype']
        bin_startype0_arr = self.data['bin_startype0']
        bin_startype1_arr = self.data['bin_startype1']

        good = self.make_cuts(min_mass=min_mass, max_mass=max_mass, fluxdict=fluxdict)
        good = good & ( ((self.data['binflag'] != 1) & np.in1d(startype_arr, startypes))
                      | ((self.data['binflag'] == 1) & (np.in1d(bin_startype0_arr, startypes) | np.in1d(bin_startype1_arr, startypes)) ) )

        r_pc_arr = self.convert_units(self.data.loc[good, 'r'], 'code', 'pc')
        r_pc_arr = r_pc_arr.values.reshape(len(r_pc_arr), 1)

        # Probabilistically count stars in each bin
        bin_edges = np.logspace( min_logr, np.log10(np.max(r_pc_arr)), bins+1 )
        bin_edges = bin_edges.reshape(1, len(bin_edges))

        inner = r_pc_arr ** 2 - bin_edges[:,:-1] ** 2
        outer = r_pc_arr ** 2 - bin_edges[:,1:] ** 2
        inner[inner < 0] = 0
        outer[outer < 0] = 0

        weight = r_pc_arr ** -1 * ( np.sqrt(inner) - np.sqrt(outer) )
        weight = np.sum(weight, axis=0)

        # Make profile
        bin_center = (bin_edges[0,:-1] + bin_edges[0,1:]) / 2

        area = np.pi * (bin_edges[0,1:] ** 2 - bin_edges[0,:-1] ** 2)
        profile = weight / area
        e_profile = np.sqrt(weight) / area

        return bin_center, profile, e_profile

    def make_smoothed_brightness_profile(self, filtname, bins=80, min_mass=None, max_mass=None, max_lum=None, fluxdict=None, startypes=startype_star, min_logr=-1.5):
        """
        Creates smoothed surface brightness profile by smearing out stars probabilistically.

        Parameters
        ----------
        filtname: str
            filter name

        bins: int (default: 80)
            number of bins used for number density profile

        min_mass: float (default: None)
            If specified, only include stars above this mass

        max_mass: float (default: None)
            If specified, only include stars below this mass
            
        max_lum: float (default: None)
            IF specified, only include stars below this luminosity [LSUN]

        fluxdict: dict (default: None)
            If specified, makes upper and lower (observed) magnitude cuts in certain filters
            Should be in the following format:
            {'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

            If you don't have a cut, put None

        startypes: array-like (default: startype_star)
            If specified, only include startypes in this list

        min_logr: float (default: -1.5)
            Minimum logarithmic radius in parsec

        Returns
        -------
        bin_center: array-like
            radial points at which profile is evaluated (in arcsec)

        profile: array-like
            array of surface brightness values (mag arcsec^-2)
        """
        assert self.dist is not None
        
        # Make relevant cuts
        startype_arr = self.data['startype']
        bin_startype0_arr = self.data['bin_startype0']
        bin_startype1_arr = self.data['bin_startype1']

        good = self.make_cuts(min_mass=min_mass, max_mass=max_mass, fluxdict=fluxdict, max_lum=max_lum)
        good = good & ( ((self.data['binflag'] != 1) & np.in1d(startype_arr, startypes))
                      | ((self.data['binflag'] == 1) & (np.in1d(bin_startype0_arr, startypes) | np.in1d(bin_startype1_arr, startypes)) ) )
        r_pc_arr = self.convert_units(self.data.loc[good, 'r'], 'code', 'pc')
        r_pc_arr = r_pc_arr.values.reshape(len(r_pc_arr), 1)

        # Probabilistically count stars in each bin
        bin_edges = np.logspace( min_logr, np.log10(np.max(r_pc_arr)), bins+1 )
        bin_edges = bin_edges.reshape(1, len(bin_edges))

        inner = r_pc_arr ** 2 - bin_edges[:,:-1] ** 2
        outer = r_pc_arr ** 2 - bin_edges[:,1:] ** 2
        inner[inner < 0] = 0
        outer[outer < 0] = 0

        weight = r_pc_arr ** -1 * ( np.sqrt(inner) - np.sqrt(outer) )

        # Pull magnitudes from specified filter (making sure binaries
        # are handled correctly)
        if filtname not in np.array(self.filtertable['filtname']): # make sure the filter actually has photometry
            raise ValueError('Given filter does not have photometry')

        mag = self.data.loc[good, 'obsMag_{}'.format(filtname)]
        binmag0 = self.data.loc[good, 'bin_obsMag0_{}'.format(filtname)]
        binmag1 = self.data.loc[good, 'bin_obsMag1_{}'.format(filtname)]
        startype0 = self.data.loc[good, 'bin_startype0']
        startype1 = self.data.loc[good, 'bin_startype1']
        binflag = self.data.loc[good, 'binflag']

        good_bin01 = (binflag == 1) & np.in1d(startype0, startypes) & np.in1d(startype1, startypes)
        good_bin0 = (binflag == 1) & np.in1d(startype0, startypes) & ~np.in1d(startype1, startypes)
        good_bin1 = (binflag == 1) & ~np.in1d(startype0, startypes) & np.in1d(startype1, startypes)
        mag[good_bin01] = add_mags(binmag0[good_bin01], binmag1[good_bin01])
        mag[good_bin0] = binmag0[good_bin0]
        mag[good_bin1] = binmag1[good_bin1]
        mag = mag.values.reshape(len(mag), 1)

        mag_weight = np.sum(10 ** (-mag / 2.5) * weight, axis=0)

        # Make profile (mag / arcsec^2)
        area = np.pi * (bin_edges[0,1:] ** 2 - bin_edges[0,:-1] ** 2)
        arcsec2 = self.convert_units(1, 'arcsec', 'rad') ** 2
        dist_pc = self.convert_units(self.dist, 'kpc', 'pc')
        angular_area = area / dist_pc ** 2

        bin_center = 206265 * ( (bin_edges[0,:-1] + bin_edges[0,1:]) / 2 ) / dist_pc
        profile = -2.5 * np.log10( (arcsec2 / angular_area) * mag_weight )

        return bin_center, profile

    def make_smoothed_veldisp_profile(self, bins=80, min_mass=None, max_mass=None, dmax=None, fluxdict=None, startypes=startype_star, min_logr=-1.5):
        """
        Creates smoothed velocity dispersion profile by smearing out stars probabilistically.

        Parameters
        ----------
        bins: int (default: 80)
            number of bins used for number density profile

        min_mass: float (default: None)
            If specified, only include stars above this mass

        max_mass: float (default: None)
            If specified, only include stars below this mass
            
        dmax: float (default: None)
            If specified, the outermost bin boundary is this value (otherwise, it's the max radial value)

        fluxdict: dict (default: None)
            If specified, makes upper and lower (observed) magnitude cuts in certain filters
            Should be in the following format:
            {'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

            If you don't have a cut, put None

        startypes: array-like (default: startype_star)
            If specified, only include startypes in this list
        
        min_logr: float (default: -1.5)
            Minimum logarithmic radius in parsec

        Returns
        -------
        bin_center: array-like
            radial points at which profile is evaluated (in pc)

        veldisp_profile: array-like
            array of velocity dispersions (in km/s)

        e_veldisp_profile: array-like
            uncertainty in velocity dispersions (in km/s)
        """
        # Make relevant cuts
        startype_arr = self.data['startype']
        bin_startype0_arr = self.data['bin_startype0']
        bin_startype1_arr = self.data['bin_startype1']

        good = self.make_cuts(min_mass=min_mass, max_mass=max_mass, fluxdict=fluxdict)
        good = good & ( ((self.data['binflag'] != 1) & np.in1d(startype_arr, startypes))
                      | ((self.data['binflag'] == 1) & (np.in1d(bin_startype0_arr, startypes) | np.in1d(bin_startype1_arr, startypes)) ) )
        r_pc_arr = self.convert_units(self.data.loc[good, 'r'], 'code', 'pc')
        r_pc_arr = r_pc_arr.values.reshape(len(r_pc_arr), 1)

        # Probabilistically count stars in each bin
        if dmax is not None:
            bin_edges = np.logspace( min_logr, np.log10(dmax), bins+1 )
        else:
            bin_edges = np.logspace( min_logr, np.log10(np.max(r_pc_arr)), bins+1 )
        bin_edges = bin_edges.reshape(1, len(bin_edges))

        inner = r_pc_arr ** 2 - bin_edges[:,:-1] ** 2
        outer = r_pc_arr ** 2 - bin_edges[:,1:] ** 2
        inner[inner < 0] = 0
        outer[outer < 0] = 0

        weight = r_pc_arr ** -1 * ( np.sqrt(inner) - np.sqrt(outer) )

        # Read in velocities
        bin_center = (bin_edges[0,:-1] + bin_edges[0,1:]) / 2

        v_arr = np.array(np.hypot(self.data.loc[good, 'vt'], self.data.loc[good, 'vr']))
        v_kms_arr = self.convert_units(v_arr, 'code', 'nb_km/s')
        v_kms_arr = v_kms_arr.reshape(len(v_kms_arr), 1)

        # Calculate velocity dispersion & uncertainty
        veldisp_profile = np.sqrt(np.sum(weight * v_kms_arr ** 2, axis=0) / np.sum(3 * weight, axis=0))
        weight = np.sum(weight, axis=0)
        e_veldisp_profile = veldisp_profile / np.sqrt(2 * weight)

        return bin_center, veldisp_profile, e_veldisp_profile

    def calculate_renclosed(self, enclosed_frac=0.5, qty='mass', bins=200, min_mass=None, max_mass=None, fluxdict=None, startypes=startype_star):
        """
        Calculate radius enclosing some percentage of mass/light by probabilistically binning stars and interpolating the CDF

        Parameters
        ----------
        enclosed_frac: float between 0 and 1, exclusive (default: 0.5)
            fraction of enclosed mass at which radius is desired
            
        qty: str, either 'mass' or 'light' (default: mass)
            depending on option, either calculates half-mass or half-light radius
        
        bins: int (default: 80)
            number of bins used for number density profile

        min_mass: float (default: None)
            If specified, only include stars above this mass

        max_mass: float (default: None)
            If specified, only include stars below this mass

        fluxdict: dict (default: None)
            If specified, makes upper and lower (observed) magnitude cuts in certain filters
            Should be in the following format:
            {'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

            If you don't have a cut, put None

        startypes: array-like (default: startype_star)
            If specified, only include startypes in this list

        Returns
        -------
        rhm: float
            half-mass radius
        """
        if (enclosed_frac >= 1) or (enclosed_frac <= 0):
            raise ValueError('enclosed_frac must be between 0 and 1, exclusive')
        
        # Find startypes and make cuts
        startype_arr = self.data['startype']
        bin_startype0_arr = self.data['bin_startype0']
        bin_startype1_arr = self.data['bin_startype1']
        
        good = self.make_cuts(min_mass=min_mass, max_mass=max_mass, fluxdict=fluxdict)
        good = good & ( ((self.data['binflag'] != 1) & np.in1d(startype_arr, startypes))
                      | ((self.data['binflag'] == 1) & (np.in1d(bin_startype0_arr, startypes) | np.in1d(bin_startype1_arr, startypes)) ) )
        
        # Calculate weighted mass distribution
        r_pc_arr = self.convert_units(self.data.loc[good, 'r'], 'code', 'pc')
        r_pc_arr = r_pc_arr.values.reshape(len(r_pc_arr), 1)

        binflag = np.array(self.data.loc[good, 'binflag'])
        startype0 = np.array(self.data.loc[good, 'bin_startype0'])
        startype1 = np.array(self.data.loc[good, 'bin_startype1'])
        
        binary = (binflag == 1)
        startype0_ok = (np.in1d(startype0, startypes))
        startype1_ok = (np.in1d(startype1, startypes))
        
        if qty == 'mass':
            mass_arr = np.array(self.data.loc[good, 'm[MSUN]'])
            mass0_arr = np.array(self.data.loc[good, 'm0[MSUN]'])
            mass1_arr = np.array(self.data.loc[good, 'm1[MSUN]'])
            
            mass_arr[binary & startype0_ok & startype1_ok] = mass0_arr[binary & startype0_ok & startype1_ok] + mass1_arr[binary & startype0_ok & startype1_ok]
            mass_arr[binary & startype0_ok & ~startype1_ok] = mass0_arr[binary & startype0_ok & ~startype1_ok]
            mass_arr[binary & ~startype0_ok & startype1_ok] = mass1_arr[binary & ~startype0_ok & startype1_ok]
        elif qty == 'light':
            lum_arr = np.array(self.data.loc[good,'luminosity[LSUN]'])
            lum0_arr = np.array(self.data.loc[good,'bin_star_lum0[LSUN]'])
            lum1_arr = np.array(self.data.loc[good,'bin_star_lum1[LSUN]'])
            
            lum_arr[binary & startype0_ok & startype1_ok] = lum0_arr[binary & startype0_ok & startype1_ok] + lum1_arr[binary & startype0_ok & startype1_ok]
            lum_arr[binary & startype0_ok & ~startype1_ok] = lum0_arr[binary & startype0_ok & ~startype1_ok]
            lum_arr[binary & ~startype0_ok & startype1_ok] = lum1_arr[binary & ~startype0_ok & startype1_ok]
        else:
            raise ValueError("qty must be either 'mass' or 'light'")
            
        # Probabilistically count stars in each bin and calculate CDF
        bin_edges = np.logspace( -5, np.log10(np.max(r_pc_arr)), bins+1 )
        bin_edges = bin_edges.reshape(1, len(bin_edges))
        bin_center = (bin_edges[0,:-1] + bin_edges[0,1:]) / 2
    
        inner = r_pc_arr ** 2 - bin_edges[:,:-1] ** 2
        outer = r_pc_arr ** 2 - bin_edges[:,1:] ** 2
        inner[inner < 0] = 0
        outer[outer < 0] = 0

        weight = r_pc_arr ** -1 * ( np.sqrt(inner) - np.sqrt(outer) )
        
        if qty == 'mass':
            bin_mass = np.sum(mass_arr.values.reshape(len(mass_arr), 1) * weight, axis=0)
            cumdist = np.cumsum(bin_mass) / np.sum(bin_mass)
        elif qty == 'light':
            bin_light = np.sum(lum_arr.values.reshape(len(lum_arr), 1) * weight, axis=0)
            cumdist = np.cumsum(bin_light) / np.sum(bin_light)
        
        # Interpolate bin_center AS A FUNCTION OF cumdist and calculate rhm
        interp = scipy.interpolate.interp1d(cumdist, bin_center)
        renclosed = float(interp(enclosed_frac))
        
        return renclosed

    def make_paramdict(self):
        """
        Create a dictionary with some useful quantities about the cluster.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        paramdict: dict
            parameter dictionary
        """
        # Initialize parameter dictionary
        paramdict = {}

        # Get relevant columns
        binary = (self.data['binflag'] == 1)
        startype = np.array(self.data['startype'])
        bin_startype0 = np.array(self.data['bin_startype0'])
        bin_startype1 = np.array(self.data['bin_startype1'])
        ospin = np.array(self.data['ospin'])
        ospin0 = np.array(self.data['ospin0'])
        ospin1 = np.array(self.data['ospin1'])

        mass = np.array(self.data['m[MSUN]'])
        mass0 = np.array(self.data['m0[MSUN]'])
        mass1 = np.array(self.data['m1[MSUN]'])

        lum = np.array(self.data['luminosity[LSUN]'])
        lum0 = np.array(self.data['bin_star_lum0[LSUN]'])
        lum1 = np.array(self.data['bin_star_lum1[LSUN]'])

        dmdt0 = np.array(self.data['dmdt0'])
        dmdt1 = np.array(self.data['dmdt1'])

        # Get count values
        paramdict['Nsys'] = len(startype)
        paramdict['Nobj'] = np.sum(~binary) + 2 * np.sum(binary)
        paramdict['Nlum'] = np.sum(~binary & np.in1d(startype, startype_star)) + np.sum(binary & (np.in1d(bin_startype0, startype_star) | np.in1d(bin_startype1, startype_star)))
        paramdict['Nstar'] = np.sum(np.in1d(startype, startype_star)) + np.sum(np.in1d(bin_startype0, startype_star)) + np.sum(np.in1d(bin_startype1, startype_star))

        for ii in range(16):
            paramdict['N_{}'.format(ii)] = np.sum(np.in1d(startype, ii)) + np.sum(np.in1d(bin_startype0, ii)) + np.sum(np.in1d(bin_startype1, ii))

        # Counting binaries
        paramdict['Nbin'] = np.sum(binary)
        paramdict['Nbinl'] = np.sum(np.in1d(bin_startype0, startype_star) | np.in1d(bin_startype1, startype_star))
        paramdict['Nbins'] = np.sum(np.in1d(bin_startype0, startype_star) & np.in1d(bin_startype1, startype_star))
        paramdict['Nbinr'] = np.sum(np.in1d(bin_startype0, startype_remnant) | np.in1d(bin_startype1, startype_remnant))
        paramdict['Nbint'] = np.sum( (dmdt0 != 0) & ~np.isnan(dmdt0) )

        for ii in range(16):
            for jj in range(ii+1):
                if ii == jj:
                    paramdict['Nbin_{}_{}'.format(ii, jj)] = np.sum(np.in1d(bin_startype0, ii) & np.in1d(bin_startype1, ii))
                else:
                    paramdict['Nbin_{}_{}'.format(ii, jj)] = np.sum(np.in1d(bin_startype0, ii) & np.in1d(bin_startype1, jj)) + np.sum(np.in1d(bin_startype0, jj) & np.in1d(bin_startype1, ii))

        # Mass transferring binaries
        for ii in range(16):
            for jj in range(16):
                if ii == jj:
                    paramdict['Nbint_{}_on_{}'.format(ii, jj)] = np.sum(np.in1d(bin_startype0, ii) & np.in1d(bin_startype1, ii) & (dmdt0 != 0) & ~np.isnan(dmdt0))
                else:
                    paramdict['Nbint_{}_on_{}'.format(ii, jj)] = np.sum(np.in1d(bin_startype0, ii) & np.in1d(bin_startype1, jj) & (dmdt1 > 0) & ~np.isnan(dmdt0)) + np.sum(np.in1d(bin_startype0, jj) & np.in1d(bin_startype1, ii) & (dmdt0 > 0) & ~np.isnan(dmdt0))

        # Get mass quantities
        paramdict['Mtot'] = np.sum(mass * ~binary) + np.sum(mass0 * binary) + np.sum(mass1 * binary)
        paramdict['Mstar'] = np.sum(mass * np.in1d(startype, startype_star)) + np.sum(mass0 * np.in1d(bin_startype0, startype_star)) + np.sum(mass1 * np.in1d(bin_startype1, startype_star))
        paramdict['Mbh'] = np.sum(mass * np.in1d(startype, 14)) + np.sum(mass0 * np.in1d(bin_startype0, 14)) + np.sum(mass1 * np.in1d(bin_startype1, 14))

        paramdict['Ltot'] = np.sum(lum * np.in1d(startype, startype_star)) + np.sum(lum0 * np.in1d(bin_startype0, startype_star)) + np.sum(lum1 * np.in1d(bin_startype1, startype_star))

        # Count other exotica

        # Millisecond pulsars
        per = 2 * np.pi / self.convert_units(ospin[np.in1d(startype, 13)], '1/yr', 'Hz')
        per0 = 2 * np.pi / self.convert_units(ospin0[np.in1d(bin_startype0, 13)], '1/yr', 'Hz')
        per1 = 2 * np.pi / self.convert_units(ospin1[np.in1d(bin_startype1, 13)], '1/yr', 'Hz')
        per_list = list(per) + list(per0) + list(per1)

        if len(per_list) > 0:
            paramdict['min_ns_per'] = np.min(per_list)
            paramdict['Nmsp'] = np.sum(np.array(per_list) <= 0.01)
        else:
            paramdict['min_ns_per'] = np.nan
            paramdict['Nmsp'] = 0

        # Turn-off mass, but only calculate this for cluster ages > 10 Myr
        if (self.z is not None) and (self.age > 0.01):
            mto = find_MS_TO(self.age, self.z)
            paramdict['mto'] = mto
            paramdict['Nbs'] = np.sum((mass > mto) & np.in1d(startype, startype_ms)) + np.sum((mass0 > mto) & np.in1d(bin_startype0, startype_ms)) + np.sum((mass1 > mto) & np.in1d(bin_startype1, startype_ms))
        else:
            paramdict['mto'] = np.nan
            paramdict['Nbs'] = np.nan
            
        # Most massive object of each stellar type
        for ii in range(16):
            if paramdict['N_{}'.format(ii)] != 0:
                paramdict['mmax_{}'.format(ii)] = np.max(np.append(mass, np.append(mass0, mass1))[np.append(startype, np.append(bin_startype0, bin_startype1)) == ii])
                paramdict['mmin_{}'.format(ii)] = np.min(np.append(mass, np.append(mass0, mass1))[np.append(startype, np.append(bin_startype0, bin_startype1)) == ii])
            else:
                paramdict['mmax_{}'.format(ii)] = np.nan
                paramdict['mmin_{}'.format(ii)] = np.nan
        
        # Half-mass & half-light radii (calculated using only stars)
        paramdict['rhm'] = self.calculate_renclosed(enclosed_frac=0.5, qty='mass')
        paramdict['rhl'] = self.calculate_renclosed(enclosed_frac=0.5, qty='light')
        
        # Velocity dispersion for all startypes (velocity dispersion of any combination of startypes can be deduced)
        # & central velocity dispersion
        paramdict['sig'] = self.make_smoothed_veldisp_profile(bins=1)[1][0]
        paramdict['sighm'] = self.make_smoothed_veldisp_profile(bins=1, dmax=paramdict['rhm'])[1][0]
        paramdict['sighl'] = self.make_smoothed_veldisp_profile(bins=1, dmax=paramdict['rhl'])[1][0]
        
        paramdict['sig_star'] = self.make_smoothed_veldisp_profile(bins=1, startypes=startype_star)[1][0]
        paramdict['sighm_star'] = self.make_smoothed_veldisp_profile(bins=1, dmax=paramdict['rhm'], startypes=startype_star)[1][0]
        paramdict['sighl_star'] = self.make_smoothed_veldisp_profile(bins=1, dmax=paramdict['rhl'], startypes=startype_star)[1][0]
        
        paramdict['sig_ms'] = self.make_smoothed_veldisp_profile(bins=1, startypes=startype_ms)[1][0]
        paramdict['sighm_ms'] = self.make_smoothed_veldisp_profile(bins=1, dmax=paramdict['rhm'], startypes=startype_ms)[1][0]
        paramdict['sighl_ms'] = self.make_smoothed_veldisp_profile(bins=1, dmax=paramdict['rhl'], startypes=startype_ms)[1][0]
        
        if np.sum(np.in1d(startype, startype_giant)) + np.sum(np.in1d(bin_startype0, startype_giant)) + np.sum(np.in1d(bin_startype1, startype_giant)) != 0:
            paramdict['sig_ms'] = self.make_smoothed_veldisp_profile(bins=1, startypes=startype_giant)[1][0]
            paramdict['sighm_ms'] = self.make_smoothed_veldisp_profile(bins=1, dmax=paramdict['rhm'], startypes=startype_giant)[1][0]
            paramdict['sighl_ms'] = self.make_smoothed_veldisp_profile(bins=1, dmax=paramdict['rhl'], startypes=startype_giant)[1][0]
        else:
            paramdict['sig_ms'] = np.nan
            paramdict['sighm_ms'] = np.nan
            paramdict['sighl_ms'] = np.nan
        
        for ii in range(16):
            if paramdict['N_{}'.format(ii)] != 0:
                paramdict['sig_{}'.format(ii)] = self.make_smoothed_veldisp_profile(bins=1, startypes=ii)[1][0]
                paramdict['sighm_{}'.format(ii)] = self.make_smoothed_veldisp_profile(bins=1, dmax=paramdict['rhm'], startypes=ii)[1][0]
                paramdict['sighl_{}'.format(ii)] = self.make_smoothed_veldisp_profile(bins=1, dmax=paramdict['rhl'], startypes=ii)[1][0]
            else:
                paramdict['sig_{}'.format(ii)] = np.nan
                paramdict['sighm_{}'.format(ii)] = np.nan
                paramdict['sighl_{}'.format(ii)] = np.nan
        
        paramdict['age'] = self.age
        
        return paramdict









