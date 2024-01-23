############################
# Code developed by Salem Ibrahim SALEM
# For inquiries, please contact: eng.salemsalem@gmail.com  or salem.ibrahim@kuas.ac.jp 
###########################
import netCDF4 as nc
import pandas as pd
import numpy as np

# Path to the NetCDF file
nc_file = 'HICO_LookUpTable_Rrs1nm_Ancillary_6M.nc'

with nc.Dataset(nc_file, mode='r') as dataset:
	# Retrieve all variable names ['Rrs', 'longitude', 'latitude', 'aot_868', 'angstrom', 'chlor_a', 'chl_ocx', 'Kd_490', 'pic', 'poc']
	var_all    = dataset.variables.keys()  
	NonRrs_Hdr = [i for i in var_all if i != 'Rrs']

	# Create headers for Rrs data (from 353 to 719) ['Rrs_353', 'Rrs_354', ...., 'Rrs_718', 'Rrs_719']
	Rrs_Hdr = ['Rrs_' + str(i) for i in range(353, 720)]  

	# Directly create the DataFrame from the dataset
	df_HICO = pd.concat([
		 pd.DataFrame({var: dataset.variables[var][:].filled(np.nan) for var in NonRrs_Hdr }),
		 pd.DataFrame(dataset.variables['Rrs'][:].filled(np.nan), columns=Rrs_Hdr, dtype="float32")],
		 axis=1 )