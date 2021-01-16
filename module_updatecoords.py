from astropy.time import Time
from astropy.coordinates import SkyCoord, get_sun, earth_orientation
import astropy.units as u
import numpy as np
from read_config import read_config
###########################################################################
# Extract parameter values from catalog.
'''
Inputs:
cfg_catalog = Configuration script containing astrometric parameters of desired source
CFG_DIR = Path to cfg_catalog
output_coord_frame =  Output coordinate frame for Astropy SkyCoord object ('fk5', 'icrs', or other frames)
'''
def extract_catparams(cfg_catalog,CFG_DIR,output_coord_frame='fk5'):
	print('Reading configuration script: %s'% (CFG_DIR+cfg_catalog))
	catalog = read_config(CFG_DIR+cfg_catalog)
	frame = catalog['frame'] # Coordinate frame used in catalog
	source = catalog['source_name']

	print('Loading astrometric paramters of %s'% (source))
	ref_epoch = Time(catalog['ref_epoch'],format='mjd')
	position = SkyCoord(ra=catalog['ra']*u.deg,dec=catalog['dec']*u.deg,frame=frame).transform_to(output_coord_frame)
	ra = position.ra # Degrees
	dec = position.dec # Degrees
	pmra_cosdec = catalog['pmra_cosdec']*u.mas/u.yr # mas/yr
	pmdec = catalog['pmdec']*u.mas/u.yr # mas/yr
	parallax = catalog['parallax']*u.mas # mas

	# Errors on astrometric parameters
	err_RAcosdec = catalog['err_ra_cosdec']*u.mas # mas
	err_RA = err_RAcosdec/np.cos(dec.rad) # mas
	err_DEC = catalog['err_dec']*u.mas # mas
	err_pmra_cosdec = np.nan_to_num(catalog['err_pmra_cosdec'])*u.mas/u.yr # mas/yr
	err_pmdec =  np.nan_to_num(catalog['err_pmdec'])*u.mas/u.yr # mas/yr
	err_parallax = catalog['err_parallax']*u.mas # mas

	# Correlation coefficients
	ra_dec_corr = catalog['ra_dec_corr']
	ra_pmracosdec_corr = catalog['ra_pmracosdec_corr']
	ra_pmdec_corr = catalog['ra_pmdec_corr']
	dec_pmracosdec_corr = catalog['dec_pmracosdec_corr']
	dec_pmdec_corr = catalog['dec_pmdec_corr']
	pmracosdec_pmdec_corr = catalog['pmracosdec_pmdec_corr']

	return position, ref_epoch, ra, dec, pmra_cosdec, pmdec, parallax, err_RAcosdec, err_RA, err_DEC, err_pmra_cosdec, err_pmdec, err_parallax, ra_dec_corr, ra_pmracosdec_corr, ra_pmdec_corr, dec_pmracosdec_corr, dec_pmdec_corr, pmracosdec_pmdec_corr
###########################################################################
# Convert equitorial coordinates (RA, Dec) and their errors to geocentric ecliptic coordinates (lambda, beta).
'''
Inputs (with astropy units):
equitorial_skycoord = SkyCoord object containing right ascension and declination information
err_ra = Uncertainty on right ascension (mas)
err_dec = Uncertainty on declination (mas)
epoch = astropy Time object representing equinox at which Earth's obliquity must be determined
'''
def equitorial2ecliptic(equitorial_skycoord, err_ra, err_dec, epoch):
	eps = earth_orientation.obliquity(epoch.jd, algorithm=2006)*u.deg # Use IAU 2006 model.
	ecliptic_skycoord = equitorial_skycoord.transform_to('geocentrictrueecliptic')
	ra = equitorial_skycoord.ra.rad
	dec = equitorial_skycoord.dec.rad
	ecl_lambda = ecliptic_skycoord.lon.rad
	beta = ecliptic_skycoord.lat.rad
	# Error on beta
	term1 = ((np.cos(dec)*np.cos(eps.to('radian')) + np.sin(dec)*np.sin(eps.to('radian'))*np.sin(ra))* err_dec)**2
	term2 = (np.cos(dec)*np.sin(eps.to('radian'))*np.cos(ra)*err_ra)**2
	err_beta = np.sqrt(term1+term2)/np.abs(np.cos(beta))
	# Error on lambda
	term1 = (np.sin(ra)*np.cos(dec)*err_ra/np.cos(beta))**2
	term2 = (np.cos(ra)*np.sin(dec)*err_dec/np.cos(beta))**2
	term3 = (np.cos(ra)*np.cos(dec)*np.tan(beta)*err_beta/np.cos(beta))**2
	err_lambda = np.sqrt(term1+term2+term3)/np.abs(np.sin(ecl_lambda))
	return ecliptic_skycoord, err_lambda, err_beta
###########################################################################
# Convert ecliptic coorinates (lambda, beta) and their errors to equitorial coordinates.
'''
Inputs:
ecliptic_skycoord = SkyCoord object containing geocentric true ecliptic longitude and latitude information
err_lambda = Error on ecliptic longitude (mas)
err_beta = Error on ecliptic latitude (mas)
epoch = astropy Time object representing equinox at which Earth's obliquity must be determined
equitorial_coord_frame = Coordinate frame for output equitorial coordinates ('icrs', 'fk5', or other)
'''
def ecliptic2equitorial(ecliptic_skycoord, err_lambda, err_beta, epoch, equitorial_coord_frame='fk5'):
	eps = earth_orientation.obliquity(epoch.jd, algorithm=2006)*u.deg # Use IAU 2006 model.
	equitorial_skycoord = ecliptic_skycoord.transform_to(equitorial_coord_frame)
	ecl_lambda = ecliptic_skycoord.lon.rad
	beta = ecliptic_skycoord.lat.rad
	ra = equitorial_skycoord.ra.rad
	dec = equitorial_skycoord.dec.rad
	# Error on dec
	term1 = ((np.cos(beta)*np.cos(eps.to('radian')) - np.sin(beta)*np.sin(eps.to('radian'))*np.sin(ecl_lambda))*err_beta)**2
	term2 = (np.cos(beta)*np.sin(eps.to('radian'))*np.cos(ecl_lambda)*err_lambda)**2
	err_dec = np.sqrt(term1+term2)/np.abs(np.cos(dec))
	# Error on ra
	term1 = (np.sin(ecl_lambda)*np.cos(beta)*err_lambda/np.cos(dec))**2
	term2 = (np.cos(ecl_lambda)*np.sin(beta)*err_beta/np.cos(dec))**2
	term3 = (np.cos(ecl_lambda)*np.cos(beta)*np.tan(dec)*err_dec/np.cos(dec))**2
	err_ra = np.sqrt(term1+term2+term3)/np.abs(np.sin(ra))
	return equitorial_skycoord, err_ra, err_dec
###########################################################################
# Construct covariance matrix from supplied errors and correlation coefficients.
'''
Inputs:
err_RAcosdec          = Error on RA*cos(Dec)   (degrees)
err_DEC               = Error on Dec           (degrees)
err_pmra_cosdec       = Error on pmra_cosdec   (degrees/yr)
err_pmdec             = Error on pmdec         (degrees/yr)
ra_dec_corr           = Corr. coeff. between RA and Dec
ra_pmracosdec_corr    = Corr. coeff. between RA and pmra_cosdec
ra_pmdec_corr         = Corr. coeff. between RA and pmdec
dec_pmracosdec_corr   = Corr. coeff. between Dec and pmra_cosdec
dec_pmdec_corr        = Corr. coeff. between Dec and pmdec
pmracosdec_pmdec_corr = Corr. coeff. between pmra_cosdec and pmdec
'''
def build_cov_matrix(err_RAcosdec, err_DEC, err_pmra_cosdec, err_pmdec, ra_dec_corr, ra_pmracosdec_corr, ra_pmdec_corr, dec_pmracosdec_corr, dec_pmdec_corr, pmracosdec_pmdec_corr):
	C11 = err_RAcosdec**2
	C12 = ra_dec_corr * err_RAcosdec * err_DEC
	C13 = ra_pmracosdec_corr * err_RAcosdec * err_pmra_cosdec
	C14 = ra_pmdec_corr * err_RAcosdec * err_pmdec

	C22 = err_DEC**2
	C23 = dec_pmracosdec_corr * err_DEC * err_pmra_cosdec
	C24 = dec_pmdec_corr * err_DEC * err_pmdec

	C33 = err_pmra_cosdec**2
	C34 = pmracosdec_pmdec_corr * err_pmra_cosdec * err_pmdec

	C44 = err_pmdec**2

	print('Constructing covariance matrix from errors and correlation coefficients between astrometric parameters')
	cov_matrix = np.array([ [C11, C12, C13, C14], [C12, C22, C23, C24], [C13, C23, C33, C34], [C14, C24, C34, C44] ])
	return cov_matrix
###########################################################################
# Correct astrometric parameters and their errors for source proper motion over short time intervals.
# Note: The simple approximation adopted in this function is not suitable for sources that are close to the celestial poles.
'''
Inputs:
input_vector = 1D array of shape (4,) = (alpha*, DEC, mu_alpha*, mu_dec) # Units = (deg, deg, deg/yr, deg/yr)
input_cov_matrix = 2D array of shape (4,4), covariance matrix of parameters at initial epoch
t = Time (years) between epoch of input values to user-suppiled final epoch
'''
def evolve_propermotion_shortintervals(input_vector, input_cov_matrix, t):
	J = np.array([ [1., 0., t, 0.], [0., 1., 0, t], [0., 0., 1., 0.], [0., 0., 0., 1.] ]) # Jacobian
	output_vector = np.matmul(J,input_vector)
	output_cov_matrix = np.matmul(np.matmul(J,input_cov_matrix),J.T)
	return output_vector, output_cov_matrix
#########################################################################
# Correct source position for parallax.
'''
Inputs:
equitorial_coords = RA/Dec SkyCoord object representing true source position
err_ra = Error on right ascension (mas)
err_dec = Error on declination (mas)
datetime = astropy Time object
parallax = annual parallax (mas)
err_parallax = Error on annual parallax (mas)
output_coord_frame = Output coordinate frame for Astropy SkyCoord object ('fk5', 'icrs', or other frames)
'''
def correct_parallax(equitorial_skycoord, err_ra, err_dec, datetime, parallax, err_parallax, output_coord_frame='fk5'):
	parallax_radians = parallax.to(u.rad).value
	# Convert from equitorial coordinates to geocentric ecliptic coordinates.
	ecliptic_skycoord, err_lambda, err_beta = equitorial2ecliptic(equitorial_skycoord, err_ra, err_dec, datetime)
	ecl_lambda = ecliptic_skycoord.lon.radian # Ecliptic longitude (radian)
	ecl_beta = ecliptic_skycoord.lat.radian # Ecliptic latitude (radian)

	# Obtain ecliptic longitude of Sun at observation epoch.
	lambda_sun = get_sun(datetime).transform_to('geocentrictrueecliptic').lon.radian # (radian)

	# Compute annual parallax shifts.
	pshift_lambda = parallax_radians*np.sin(lambda_sun-ecl_lambda)/np.cos(ecl_beta) # radians
	pshift_beta = -parallax_radians*np.cos(lambda_sun-ecl_lambda)*np.sin(ecl_beta) # radians
	# Compute errors on parallax shifts.
	# Error on parallax shift along ecliptic longitude.
	term1 = (err_parallax.to(u.mas).value/parallax.to(u.mas).value)**2
	term2 = (err_lambda.to(u.rad).value/np.tan(lambda_sun-ecl_lambda))**2
	term3 = (np.tan(ecl_beta)*err_beta.to(u.rad).value)**2
	err_pshift_lambda = np.abs(pshift_lambda)*np.sqrt(term1+term2+term3) # radians
	# Error on parallax shift along ecliptic latitude
	term2 = (np.tan(lambda_sun-ecl_lambda)*err_lambda.to(u.rad).value)**2
	term3 = (err_beta.to(u.rad).value/np.tan(ecl_beta))**2
	err_pshift_beta = np.abs(pshift_beta)*np.sqrt(term1+term2+term3) # radians

	# Incorporate parallax corrections into source positions.
	eclip_lon_updated = ecl_lambda + pshift_lambda
	eclip_lat_updated = ecl_beta + pshift_beta
	updated_coords = SkyCoord(lon=eclip_lon_updated*u.rad, lat=eclip_lat_updated*u.rad, frame='geocentrictrueecliptic')
	err_eclip_lon_updated = np.sqrt(err_lambda.to(u.rad).value**2 + err_pshift_lambda**2) # radians
	err_eclip_lat_updated = np.sqrt(err_beta.to(u.rad).value**2 + err_pshift_beta**2) # radians

	# Convert updated ecliptic coordinates back to equitorial coordinates.
	output_skycoord, err_ra_out, err_dec_out = ecliptic2equitorial(updated_coords, err_eclip_lon_updated, err_eclip_lat_updated, datetime, output_coord_frame)
	err_ra_out = (err_ra_out*u.rad).to(u.mas)
	err_dec_out = (err_dec_out*u.rad).to(u.mas)
	return output_skycoord, err_ra_out, err_dec_out
#########################################################################
# Correct source positions and errors for proper motion and parallax.
'''
Inputs:
times = Astropy Time object indicating epochs at which corrected position must be determined
ref_epoch = Astropy Time object representing reference epoch of catalogued astrometric parameters
position = SkyCoord object representing RA and Dec of source at reference epoch
err_RAcosdec = Error on RA * cos(Dec)  (mas)
err_DEC = Error on Dec (mas)
pmra_cosdec = Proper motion in right ascension direction (mas/yr)
pmdec = Proper motion along declination (mas/yr)
err_pmra_cosdec  = Error on pmra_cosdec (mas/yr)
err_pmdec = Error on pmdec (mas/yr)
parallax = Annual parallax (mas)
err_parallax = Error on annual parallax (mas)
ra_dec_corr = Correlation coefficient between RA and Dec
ra_pmracosdec_corr = Correlation coefficient between RA and pmra_cosdec
ra_pmdec_corr = Correlation coefficient between RA and pmdec
dec_pmracosdec_corr = Correlation coefficient between Dec and pmra_cosdec
dec_pmdec_corr = Correlation coefficient between Dec and pmdec
pmracosdec_pmdec_corr = Correlation coefficient between pmra_cosdec and pmdec
do_parallax_correction = Do you want to correct for annual parallax? (True/False)
output_coord_frame = Output coordinate frame for Astropy SkyCoord object ('fk5', 'icrs', or other frames)
'''
def update_positions(times, ref_epoch, position, err_RAcosdec, err_DEC, pmra_cosdec, pmdec, err_pmra_cosdec, err_pmdec, parallax, err_parallax, ra_dec_corr, ra_pmracosdec_corr, ra_pmdec_corr, dec_pmracosdec_corr, dec_pmdec_corr, pmracosdec_pmdec_corr, do_parallax_correction = True, output_coord_frame='fk5'):
	RAcosdec = position.ra.deg*np.cos(position.dec.rad)
	DEC = position.dec.rad
	input_vector = np.array([RAcosdec, position.dec.deg, pmra_cosdec.to(u.deg/u.yr).value, pmdec.to(u.deg/u.yr).value])
	input_cov_matrix = build_cov_matrix(err_RAcosdec.to(u.deg).value, err_DEC.to(u.deg).value, err_pmra_cosdec.to(u.deg/u.yr).value, err_pmdec.to(u.deg/u.yr).value, ra_dec_corr, ra_pmracosdec_corr, ra_pmdec_corr, dec_pmracosdec_corr, dec_pmdec_corr, pmracosdec_pmdec_corr)

	if times.shape:
		N = len(times)
		src_ra = np.zeros(N)
		src_dec = np.zeros(N)
		src_err_ra = np.zeros(N)
		src_err_dec = np.zeros(N)
	else:
		N = 1
		times = np.array([times])

	for i in range(N):
		print('Tranforming astrometric parameters and their errors to epoch J%.4f'% (times[i].jyear))
		t = (times[i].mjd - ref_epoch.mjd)/365.25 # years
		print("Correcting position for proper motion")
		output_vector, output_cov_matrix = evolve_propermotion_shortintervals(input_vector, input_cov_matrix, t)
		RA_updated = output_vector[0]/np.cos(DEC) # Degrees
		DEC_updated = output_vector[1] # Degrees
		err_RA_updated = np.sqrt(output_cov_matrix[0,0])*3.6e6*u.mas/np.cos(DEC) # mas
		err_DEC_updated = np.sqrt(output_cov_matrix[1,1])*3.6e6*u.mas  # mas
		propermotion_pos = SkyCoord(ra=RA_updated*u.deg, dec=DEC_updated*u.deg, frame=output_coord_frame)
		if do_parallax_correction:
			print('Correcting position for parallax')
			updated_coords, err_ra_out, err_dec_out = correct_parallax(propermotion_pos, err_RA_updated, err_DEC_updated, times[i], parallax, err_parallax, output_coord_frame)
		else:
			updated_coords = propermotion_pos
			err_ra_out = err_RA_updated
			err_dec_out = err_DEC_updated
		if N==1:
			final_ra = updated_coords.ra.deg
			final_dec = updated_coords.dec.deg
			return final_ra, final_dec, err_ra_out, err_dec_out # Positions in degrees, errors in mas
		else:
			src_ra[i] = updated_coords.ra.deg
			src_dec[i] = updated_coords.dec.deg
			src_err_ra[i] = err_ra_out.to(u.mas).value
			src_err_dec[i] = err_dec_out.to(u.mas).value
	return src_ra, src_dec, src_err_ra, src_err_dec
#########################################################################
