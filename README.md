# Summary

*cmctoolkit* is a set of *python* methods for analyzing the output of *Cluster Monte Carlo* (*CMC*), a star cluster simulation code ([Pattabiraman et al. 2013](https://iopscience.iop.org/article/10.1088/0067-0049/204/2/15/meta)). The methods are structured around a *Snapshot* class. These functions are provided to supplement the upcoming release of *CMC*, as well as the recent release of the *CMC Cluster Catalog*, a grid of globular cluster models using *CMC* ([Kremer et al. 2020](https://iopscience.iop.org/article/10.3847/1538-4365/ab7919/meta?casa_token=w1TxNMpr1hUAAAAA:IODrgj9xFbvD1a8AjHjbVoieXoGPYH9BcEbSLKAdUOttsypdcKPArieSkzIKpRA5OoYwhLuVIg)). We include in this directory also a Jupyter notebook (*examples.ipynb*) to demonstrate its use, as well as files containing the simulated V-band surface brightness profile (*SBP####.txt*), 1D velocity dispersion profile (*VDP####.txt*), and miscellaneous other parameters described below (*PARAM####.txt*).

This set of functions accompanies Rui et al. in prep. on matching observed globular clusters to models in the *CMC Cluster Catalog*.

# Contents of *PARAM####.txt*

We have included in this repository a *PARAM####.txt* corresponding to every snapshot in all 148 models in the *CMC Cluster Catalog* for which the snapshot time is >10 Gyr. This file gives for each model a list of interesting parameters/properties, described below. Some parameters are given as nan when a meaningful value cannot be assigned (e.g., the velocity dispersion of black holes in a cluster with no black holes).

## SSE startypes

The keywords in *PARAM####.txt* sometimes make reference to startypes from the single-star evolution (SSE) prescription from [Hurley et al. 2000](https://academic.oup.com/mnras/article/315/3/543/972062). We reproduce here the meaning of all startype numbers.

0 • MS star M ≲ 0.7 deeply or fully convective

1 • MS star M ≳ 0.7

2 • Hertzsprung Gap (HG)

3 • First Giant Branch (GB)

4 • Core Helium Burning (CHeB)

5 • Early Asymptotic Giant Branch (EAGB)

6 • Thermally Pulsing Asymptotic Giant Branch (TPAGB)

7 • Naked Helium Star MS (HeMS)

8 • Naked Helium Star Hertzsprung Gap (HeHG)

9 • Naked Helium Star Giant Branch (HeGB)

10 • Helium White Dwarf (HeWD)

11 • Carbon/Oxygen White Dwarf (COWD)

12 • Oxygen/Neon White Dwarf (ONeWD)

13 • Neutron Star (NS)

14 • Black Hole (BH)

15 • massless remnant

## Bulk cluster properties

**Ltot**: Total cluster luminosity (Lsun)

**Mbh**: Total mass in black holes (Msun)

**Mstar**: Total mass in stars (SSE startypes 0-9)

**Mtot**: Total cluster mass (Msun)

**age**: Age of the cluster (Gyr)

## Stellar populations

**N\_()** describes a family of keywords for the number of a given SSE startype inside our simulation. For example, the keyword **N_13** refers to the total number of all neutron stars in the cluster.

**mmax\_()** describes a family of keywords for the maximum mass of a given SSE startype in the cluster. For example, the keyword **mmax_14** refers to the mass of the most massive black hole in the cluster.

**mmin\_()** describes a family of keywords for the minimum mass of a given SSE startype in the cluster.

**Nobj**: Number of objects (sum of total number of single stars and binary stars, with a binary system corresponding to two objects)

**Nstar**: Number of stars (SSE startypes 0-9)

**Nsys**: Number of systems (sum of total number of single stars and binary systems, with a binary system corresponding to one system)

**Nlum**: Number of luminous systems, i.e., the sum of all single stars and all binaries with at least one star in them

**Nmsp**: Number of millisecond pulsars (neutron stars with periods <10 ms)

**min\_ns\_per**: Minimum neutron star period (s)

## Binaries

**Nbin\_()\_()** describes a family of keywords for the number of binaries consisting of members from two different SSE startypes (with the higher number listed first). For example, the keyword **Nbin_10_0** corresponds to all low-mass main-sequence stars in a binary with a He WD, and **Nbin_14_14** corresponds to all BH-BH binaries.

**Nbint_()\_on\_()** describes a family of keywords for mass-transferring binaries from some specified SSE startypes, with the first listed startype accreting onto the second. For example, the keyword **Nbint_1_on_13** corresponds to all high-mass main-sequence stars accreting onto neutron stars.

**Nbin**: Number of binary systems

**Nbinl**: Number of binary systems where at least one of the objects is a star (SSE startypes 0-9)

**Nbins**: Number of binary systems where both objects are stars (SSE startypes 0-9)

**Nbinr**: Number of binary systems where at least one of the objects is a remnant (WD, NS, or BH; SSE startypes 10-14)

**Nbint**: Number of mass-transferring binaries

## Velocity dispersion

**sig\_()** describes a family of keywords for the velocity dispersion of a given SSE startype. For example, the keyword **sig_3** describes the velocity dispersion of red giants on the first red giant branch.

**sig**: Cluster velocity dispersion (km/s)

**sig_ms**: Velocity dispersion of main sequence stars (km/s; SSE startypes 0-1)

**sig_star**: Velocity dispersion of stars (km/s; SSE startypes 0-9)

# Description of *Snapshot* Class

Snapshot class for snapshot file, usually something like 'initial.snap0137.dat.gz',
paired alongside conversion file and, preferably, distance and metallicity.

### Parameters
**fname**: str
filename of snapshot

**conv**: str, dict, or pd.DataFrame
if str, filename of unitfile (e.g., initial.conv.sh)
if dict, dictionary of unit conversion factors
if pd.DataFrame, table corresponding to initial.conv.sh

**age**: float
age of the cluster at the time of the snapshot in Myr

**dist**: float (default: None)
distance to cluster in kpc

**z**: float (default: None)
cluster metallicity

### Attributes

**data**: pd.DataFrame
snapshot table

**unitdict**: dict
Dictionary containing unit conversion information

**dist**: float (None)
distance to cluster in kpc

**filtertable**: pd.DataFrame
table containing information about all filter for which photometry exists

# Methods

Described below are the methods of the *Snapshot* object.

## *convert_units*

Converts an array from CODE units to 'unit' using conversion factors specified
in unitfile.

Note: 's_' preceding an out_unit refers to 'stellar' quantities. 'nb_' refers
to n-body units. Without these tags, it is presumed otherwise.

### Parameters
**arr**: array-like
array to be converted

**in_unit**: str (default: 'code')
unit from which arr is to be converted

**out_unit**: str (default: 'code')
unit to which arr is to be converted

### Returns
**converted**: array-like
converted array

## *make_cuts*
Helper method to return a boolean array where a given set of cuts are
satisfied.

### Parameters
**min_mass**: float (default: None)
If specified, only include stars above this mass

**min_mass**: float (default: None)
If specified, only include stars below this mass

**dmin**: float (default: None)
If specified, only include stars outside this projected radius

**dmax**: float (default: None)
If specified, only include stars inside this projected radius

**max_lum**: float (default: None)
IF specified, only include stars below this luminosity [LSUN]

**fluxdict**: dict (default: None)
If specified, makes upper and lower (observed) magnitude cuts in certain filters
Should be in the following format:
{'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

If you don't have a cut, put None

### Returns
**good**: array-like of bool
boolean array specifying where cuts are satisfied

## *add_photometry*
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

### Parameters
**filttable**: str or pd.DataFrame
if str, path to filter table
if pd.DataFrame, table containing information about filters (see function: load_filtertable)

### Returns
none

## *make_2d_projection*
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

### Parameters
**seed**: int (default: None)
random seed, if None then don't set a seed

### Returns
none

## *calc_Teff*
Calculate the effective temperature Teff for every (non-BH) star in the catalog.

Adds the following columns:
Teff[K] : Effective temperature of a single star, np.nan for a binary or black hole
bin_Teff0[K] : Effective temperature for first star in a binary, np.nan for a single or black hole
bin_Teff1[K] : Effective temperature for second star in a binary, np.nan for a single or black hole

### Parameters
none

### Returns
none

## *calc_surface_gravity*
Adds surface gravity columns to snapshot table.

Adds the following columns:
g[CM/S] : Surface gravity of a single star, np.nan for a binary
bin_g0[CM/S] : Surface gravity for first star in a binary, np.nan for a single
bin_g1[CM/S] : Surface gravity for second star in a binary, np.nan for a single

### Parameters
none

### Returns
none

## *make_spatial_density_profile*

Creates a spatial density profile.

### Parameters
**bin_edges**: array-like
Bin edges of radial density profile (if None, make 100 bins from min to max, logarithmic spacing)

**min_mass**: float (default: None)
If specified, only include stars above this mass

**min_mass**: float (default: None)
If specified, only include stars below this mass

**fluxdict**: dict (default: None)
If specified, makes upper and lower (observed) magnitude cuts in certain filters
Should be in the following format:
{'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

If you don't have a cut, put None

**startypes**: array-like (default: startype_star)
If specified, only include startypes in this list

### Returns
**bin_edges**: array-like
Bin edges

**profile**: array-like
Number density in each annulus

**e_profile**: array-like
Poisson error in number density in each annulus

## *velocity_dispersion*
Calculate velocity dispersion in km/s.

### Parameters
**min_mass**: float (default: None)
If specified, only include stars above this mass

**min_mass**: float (default: None)
If specified, only include stars below this mass

**dmin**: float (default: None)
If specified, only include stars outside this projected radius

**dmax**: float (default: None)
If specified, only include stars inside this projected radius

**startypes**: array-like (default: startype_star)
If specified, only include startypes in this list

### Returns
**veldisp**: float
Velocity dispersion (km/s)

**e_veldisp**: float
Uncertainty in velocity dispersion (km/s)

## *make_velocity_dispersion_profile*

Calculate velocity dispersion in km/s.

### Parameters
**bin_edges**: array-like (default: None)
Bin edges of mass function (if None, make 100 bins from min to max, logarithmic spacing)

**min_mass**: float (default: None)
If specified, only include stars above this mass

**min_mass**: float (default: None)
If specified, only include stars below this mass

**fluxdict**: dict (default: None)
If specified, makes upper and lower (observed) magnitude cuts in certain filters
Should be in the following format:
{'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

If you don't have a cut, put None

**startypes**: array-like (default: startype_star)
If specified, only include startypes in this list

### Returns
**bin_edges**: array-like
Bin edges of mass function (pc)

**veldisp_profile**: array-like
Velocity dispersion profile (km/s)

**e_veldisp_profile**: array-like
Uncertainty in velocity dispersion profile (km/s)

## *make_mass_function*

Creates a mass function.

### Parameters
**bin_edges**: array-like
Bin edges of mass function (if None, make 100 bins from min to max, logarithmic spacing)

**dmin**: float (default: None)
If specified, only include stars outside this projected radius

**dmax**: float (default: None)
If specified, only include stars inside this projected radius

**fluxdict**: dict (default: None)
If specified, makes upper and lower (observed) magnitude cuts in certain filters
Should be in the following format:
{'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

If you don't have a cut, put None

**startypes**: array-like (default: startype_star)
If specified, only include startypes in this list

### Returns
**bin_edges**: array-like
Bin edges

**mf**: array-like
Mass function

**e_mf**: array-like
Mass function uncertainty

## *fit_mass_function_slope*

Fits the mass function slope in a bin-independent way.

### Parameters
**init_guess**: float (default: 2.35)
initial guess for the power law slope (if unspecified, guess Salpeter)

**min_mass**: float (default: None)
If specified, only include stars above this mass

**max_mass**: float (default: None)
If specified, only include stars below this mass

**dmin**: float (default: None)
If specified, only include stars outside this projected radius

**dmax**: float (default: None)
If specified, only include stars inside this projected radius

**fluxdict**: dict (default: None)
If specified, makes upper and lower (observed) magnitude cuts in certain filters
Should be in the following format:
{'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

If you don't have a cut, put None

**startypes**: array-like (default: startype_star)
If specified, only include startypes in this list

### Returns
**alpha**: float
Converged value for mass function slope (if failed fit, returns np.nan)

## *make_smoothed_number_profile*

Creates smoothed number density profile by smearing out stars probabilistically.

### Parameters
**bins**: int (default: 80)
number of bins used for number density profile

**min_mass**: float (default: None)
If specified, only include stars above this mass

**max_mass**: float (default: None)
If specified, only include stars below this mass

**fluxdict**: dict (default: None)
If specified, makes upper and lower (observed) magnitude cuts in certain filters
Should be in the following format:
{'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

If you don't have a cut, put None

**startypes**: array-like (default: startype_star)
If specified, only include startypes in this list

**min_logr**: float (default: -1.5)
Minimum logarithmic radius in parsec

### Returns
**bin_center**: array-like
radial points at which profile is evaluated (in pc)

**profile**: array-like
array of number densities

**e_profile**: array-like
uncertainty in number density

## *make_smoothed_brightness_profile*
Creates smoothed surface brightness profile by smearing out stars probabilistically.

### Parameters
**filtname**: str
filter name

**bins**: int (default: 80)
number of bins used for number density profile

**min_mass**: float (default: None)
If specified, only include stars above this mass

**max_mass**: float (default: None)
If specified, only include stars below this mass

**max_lum**: float (default: None)
If specified, only include stars below this luminosity [LSUN]

**fluxdict**: dict (default: None)
If specified, makes upper and lower (observed) magnitude cuts in certain filters
Should be in the following format:
{'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

If you don't have a cut, put None

**startypes**: array-like (default: startype_star)
If specified, only include startypes in this list

**min_logr**: float (default: -1.5)
Minimum logarithmic radius in parsec

### Returns
**bin_center**: array-like
radial points at which profile is evaluated (in arcsec)

**profile**: array-like
array of surface brightness values (mag arcsec^-2)

## *make_smoothed_veldisp_profile*

Creates smoothed velocity dispersion profile by smearing out stars probabilistically.

### Parameters
**bins**: int (default: 80)
number of bins used for number density profile

**min_mass**: float (default: None)
If specified, only include stars above this mass

**max_mass**: float (default: None)
If specified, only include stars below this mass

**dmax**: float (default: None)
If specified, the outermost bin boundary is this value (otherwise, it's the max radial value)

**fluxdict**: dict (default: None)
If specified, makes upper and lower (observed) magnitude cuts in certain filters
Should be in the following format:
{'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

If you don't have a cut, put None

**startypes**: array-like (default: startype_star)
If specified, only include startypes in this list

**min_logr**: float (default: -1.5)
Minimum logarithmic radius in parsec

### Returns
**bin_center**: array-like
radial points at which profile is evaluated (in pc)

**veldisp_profile**: array-like
array of velocity dispersions (in km/s)

**e_veldisp_profile**: array-like
uncertainty in velocity dispersions (in km/s)

### *calculate_renclosed*

Calculate radius enclosing some percentage of mass/light by probabilistically binning stars and interpolating the CDF

### Parameters
**enclosed_frac**: float between 0 and 1, exclusive (default: 0.5)
fraction of enclosed mass at which radius is desired

**qty**: str, either 'mass' or 'light' (default: mass)
depending on option, either calculates half-mass or half-light radius

**bins**: int (default: 80)
number of bins used for number density profile

**min_mass**: float (default: None)
If specified, only include stars above this mass

**max_mass**: float (default: None)
If specified, only include stars below this mass

**fluxdict**: dict (default: None)
If specified, makes upper and lower (observed) magnitude cuts in certain filters
Should be in the following format:
{'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

If you don't have a cut, put None

**startypes**: array-like (default: startype_star)
If specified, only include startypes in this list

### Returns
**rhm**: float
half-mass radius

## *binary_fraction*
Calculate the binary fraction subject to some cuts.

### Parameters
**min_q**: float (default: None)
If specified, only report binary fraction for binaries with q > qmin.

**max_q**: float (default: None)
If specified, only report binary fraction for binaries with q < qmax.

**min_mass**: float (default: None)
If specified, only include stars above this mass

**max_mass**: float (default: None)
If specified, only include stars below this mass

**dmin**: float (default: None)
If specified, only include stars outside this projected radius. This
is not done by random projection.

**dmax**: float (default: None)
If specified, only include stars inside this projected radius. This
is not done by random projection.

**fluxdict**: dict (default: None)
If specified, makes upper and lower (observed) magnitude cuts in certain filters
Should be in the following format:
{'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

If you don't have a cut, put None

**startypes**: array-like (default: startype_star)
If specified, only include startypes in this list for consideration

**bin_startypes**: array-like (default: startype_star)
If specified, only include startypes in this list as binaries

### Returns
**binfrac**: float
Binary fraction, defined as the fraction of detectable point sources which are
actually binary systems (!= fraction of stars in binaries).

**e_binfrac**: float
Error in the binary fraction, assuming Poissoninan counting error.

## *binary_fraction_photometric*

Calculate the binary fraction subject to some cuts. Use cuts on the CMD.

### Parameters
**filttable**: pd.DataFrame
table containing information about filter functions

**mag_filter**: str (default: None)
Filter name for the "mag filter" (y-axis of CMD)

**blue_filter**: str (default: None)
Filter name for the "blue filter"

**red_filter**: str (default: None)
Filter name for the "red filter"

**min_q**: float (default: None)
If specified, only report binary fraction for binaries with q > qmin.

**max_q**: float (default: None)
If specified, only report binary fraction for binaries with q < qmax.

**min_mass**: float (default: None)
If specified, only include stars above this mass

**max_mass**: float (default: None)
If specified, only include stars below this mass

**dmin**: float (default: None)
If specified, only include stars outside this projected radius. This
is not done by random projection.

**dmax**: float (default: None)
If specified, only include stars inside this projected radius. This
is not done by random projection.

**fluxdict**: dict (default: None)
If specified, makes upper and lower (observed) magnitude cuts in certain filters
Should be in the following format:
{'filtname1': [faint1, brite1], 'filtname2': [faint1, brite1], ...}

If you don't have a cut, put None

**primary_flux_faint**: float (default: None)
If specified, makes faint magnitude cut in the magnitude filter only to the primary
star, not the whole system. This is used to more closely match
the cuts applied to the ACS Globular Cluster Survey in Milone et al. 2012.

Give in absolute magnitude.

**primary_flux_brite**: float (default: None)
If specified, makes brite magnitude cut in the magnitude filter only to the primary
star, not the whole system. This is used to more closely match
the cuts applied to the ACS Globular Cluster Survey in Milone et al. 2012.

Cuts on mag filter, interpolates wrt. blue_filter-red_filter color, cut between faint_mag
and brite_mag. Note: You're probably not intending to use both this argument and fluxdict.

Give in absolute magnitude.

**color_pad**: float (default: 0.1)
To be considered as a single star or binary of interest, enforce that a source
is at most this many magnitudes redder than the turnoff. To remove, set to None.

### Returns
**binfrac**: float
Binary fraction, defined as the fraction of detectable point sources which are
actually binary systems (!= fraction of stars in binaries).

**e_binfrac**: float
Error in the binary fraction, assuming Poissoninan counting error.

## *get_blue_stragglers*

A function to identify observationally realistic blue stragglers.
Procedure is to locate all stars brighter than the turn-off magnitude
and bluer than the turnoff color optionally padded by some amount.
Additionally, blue stragglers are restricted to singles which are
main sequence or binaries containing at least one main sequence star
and no giant (where such binaries are counted as a single star).

### Parameters
**mag_filter**: str (default: None)
Filter name for the "mag filter" (y-axis of CMD)

**blue_filter**: str (default: None)
Filter name for the "blue filter"

**red_filter**: str (default: None)
Filter name for the "red filter"

**color_pad**: float (default: 0.1)
To be counted as an "observational" blue straggler, a system must
be bluer than the turn-off color by at least color_pad.

### Returns
**bs_cat**: pd.DataFrame
Blue straggler catalog
