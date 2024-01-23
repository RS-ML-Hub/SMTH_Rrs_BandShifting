############################
# Code developed by Salem Ibrahim SALEM
# For inquiries, please contact: eng.salemsalem@gmail.com  or salem.ibrahim@kuas.ac.jp 
###########################
import numpy as np
import pandas as pd
import netCDF4 as nc

Inp_Snsr_n = "AERNT"
Out_Snsr_n = "OLCI"

##.. Input Dataset 
Inp_data_fn = "Sample_Input.csv"



# HICO LookUp Table Dataset 
HICO_fn = 'HICO_LookUpTable_Rrs1nm_Ancillary_6M.nc' #"LookUp_HICO_Rrs_1nm_ary_CubicSpline_Interpolation_6M.npz"

# Output file name
Output_fn =  f"Result_HICOMatchup_Resample{Out_Snsr_n}.csv"  

# We recommend using Lsq_angle cost function as it combines two metrics (Least-Square Distance & Angle Factor) to extract HICO matchup 
CostFunction_LUT         = "Lsq_angle"   # there are two options [ "RMSE"  "Lsq_angle" ]

###########################################################################################
##.. Sensors details
sensor_labels = {
	'AERNT'    :'AERONET-OC',
	'AERNT_JP' :'AERONET-OC_JP',
	'OLCI'     :'Sentinel-3',
	'OLI'      :'Landsat-8',
	'MSI'      :'Sentinel-2', 
	'VIIRS'    : 'SNPP', 
	'SGLI'     : 'GCOM-C',
	'MODIS'    : 'Aqua',
	'GOCI2'    : 'Geo-Kompsat-2B',
	'MERIS'    : 'Envisat',
}

sensor_wavelengths = {
    'AERNT'    : [412, 443, 490, 532, 551, 667],  
    'AERNT_JP' : [400, 412, 443, 490, 510, 560, 620, 667],  
    'OLCI'     : [400 ,412, 443, 490, 510, 560 ,620, 665 ,674 ,681 ,709],  
    'OLI'      : [443, 482, 561, 655],  
    'MSI'      : [443, 490, 560, 665, 705],
    'VIIRS'    : [412, 445, 488, 555, 672], 
	'SGLI'     : [380, 412, 443, 490, 530, 565, 674],
	'MODIS'    : [412, 443, 488, 531, 551, 667, 678],	
	'GOCI2'    : [380, 412, 443, 490, 510, 555, 620, 660, 680, 709],
	'MERIS'    : [412, 443, 490, 510, 560, 620, 665, 681, 709],
}


###########################################################################################
###...Spectral Matching Function
def SpectralMatch_Broadcast_Func (YY_, XX_, CostFunction_LUT):
	if CostFunction_LUT == "RMSE":
		# Source:https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/Broadcasting.html
		""" Computing RMSE using memory-efficient vectorization.
			RMSE = [Σ(yy - xx)^2/n]^0.5
			Σ(yy - xx)^2 =  Σyy^2 - 2Σxx.yy + Σxx^2  >> memory-efficient to calculate Σ(y - x)^2  && RMSE
	
		Parameters
		----------
		YY_ : numpy.ndarray, shape=(N, D) 
		XX_ : numpy.ndarray, shape=(M, D) 
		
	
		Returns
		-------
		numpy.ndarray, shape=(N, M)
			The RMSE between each pair of rows between `x` and `y`.
		"""
		No_row_YY_    = YY_.shape[0]
		No_col_YY_ = YY_.shape[1]
		No_row_XX_ = XX_.shape[0]
		
		sqr_dists = np.empty((No_row_YY_, No_row_XX_), dtype="int32")
		
		sqr_dists = -2 * np.matmul(YY_, XX_.T)
		sqr_dists += np.sum(YY_**2, axis=1)[:, np.newaxis]
		sqr_dists += np.sum(XX_**2, axis=1)
		
		return  ( np.sqrt(np.clip(sqr_dists/No_col_YY_, a_min=0, a_max=None)) ).argmin(axis=1)  # np.clip(a, a_min, a_max): Clip limits the values in an array to be within min & max

	if CostFunction_LUT == "Lsq_angle":
		# Source:https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/Broadcasting.html
		""" Computing least-square distance using memory-efficient vectorization.
		Δ = α * Lsq	
		α    = Σ(yy * xx) / [ √ Σ(xx^2)  *  √ Σ(yy^2)]  ## where: √  is a SQRT
		Lsq  = [√ Σ(yy - xx)^2 / Σ(yy) ]
			
		   Σ(yy - xx)^2 =  Σyy^2 - 2Σxx.yy + Σxx^2  >> memory-efficient to calculate Σ(y - x)^2  && RMSE
	
		Parameters
		----------
		YY_ : numpy.ndarray, shape=(N, D)    # e.g., shape (100, 6)       > input stations
		XX_ : numpy.ndarray, shape=(M, D)    # e.g., shape (6 335 988, 6) > HICO LUT Rrs
	
	
		Returns
		-------
		numpy.ndarray, shape=(N, M)
		"""		
	
		No_row_YY_ = YY_.shape[0]    # e.g., 100         >> no. of stations
		No_col_YY_ = YY_.shape[1]    # e.g., 6           >> no. of wavelengths
		No_row_XX_ = XX_.shape[0]    # e.g., 6 335 988   >> no. of HICO stations
		
		sqr_dists = np.empty((No_row_YY_, No_row_XX_), dtype="int32")  # e.g., shape  100 * 6 335 988 
	
		sqr_dists = -2 * np.matmul(YY_, XX_.T)
		sqr_dists += np.sum(YY_**2, axis=1)[:, np.newaxis]
		sqr_dists += np.sum(XX_**2, axis=1)                   # e.g., shape  100 * 6 335 988 
		sqr_dists  = np.clip(sqr_dists, a_min=0, a_max=None)  # np.clip(a, a_min, a_max): Clip limits the values in an array to be within min & max >> important before SQRT
		
		lsq = np.sqrt(sqr_dists) / np.sum(YY_, axis=1)[:, np.newaxis]  # [:, np.newaxis]  required to change shape from "100" to "100*1" to perform broadcast
		
		alpha = np.arccos(  np.matmul(YY_, XX_.T) / (  np.sqrt(np.sum(YY_**2, axis=1))[:, np.newaxis] *  np.sqrt(np.sum(XX_**2, axis=1))[np.newaxis, :]  )  )
		
		return (lsq * alpha).argmin(axis=1)
	
###########################################################################################
###... Resample HICO Hyperspectal Rrs at Nominal bands  
def Resample_HyperSp_to_MultiSp_HICO (df_HICO, Nominal_bands, RSR_sensor, df_Resampled):
	print("Resample Rrs at Nominal band .....")
	for ii8 in Nominal_bands:  # Loop of Nominal bands of input sensor >> e.g., "SGLI" [380, 412, 443, 490, 530, 565, 674] >> no need to worry about float bands "e.g., 674" because RSR file is every 1nm and this bands will be use as name
	
		##*** no need to change any thing from here >>
		Hdr_ = "Rrs_" + str(ii8)      # e.g. ii8=674  >> Hdr_ = Rrs_674
		Hdr_RSR = "RSR_" + Hdr_      #e.g., RSR_Rrs_674

		print(Hdr_ , ">>", Hdr_RSR)
		
		RSR_one_Band_value = RSR_sensor.loc[ ~RSR_sensor[Hdr_RSR].isnull(),  Hdr_RSR].to_numpy()     # for ii8=674 >> select rows is not NaN, for column "RSR_Rrs_674" >> list of RSR 
		RSR_one_band_hdr   = RSR_sensor.loc[ ~RSR_sensor[Hdr_RSR].isnull(),  "wavelength"].to_list() # for ii8=674 >> select rows is not NaN, for column "wavelength"  >> [650, 651,... 673, 674, ... , 694]
		RSR_one_band_hdr = ["Rrs_"+str(ii) for ii in RSR_one_band_hdr]  # add "Rrs_" to each wavelength  >> e.g., ['Rrs_650', 'Rrs_651',... 'Rrs_673','Rrs_674', ... , 'Rrs_694']
		
		#print(RSR_one_band_hdr)
		#*** to here <<
		RSR_hdr_1st = RSR_one_band_hdr[0]
		RSR_hdr_last = RSR_one_band_hdr[-1]
		df_Resampled[Hdr_] = np.sum(df_HICO.loc[:, RSR_hdr_1st:RSR_hdr_last].to_numpy() * RSR_one_Band_value, axis=1) / np.sum (RSR_one_Band_value)	
		# Slice take much memory >> df_Resampled[Hdr_] = np.sum (df_HICO[RSR_one_band_hdr].to_numpy() * RSR_one_Band_value, axis=1) / np.sum (RSR_one_Band_value)  # axis=1 : means calculation in rows direction

		# convert dtype into "float32"
		df_Resampled = df_Resampled.astype("float32")
		
	return df_Resampled

###########################################################################################
###... Resample Hyperspectral Rrs at Nominal bands  
def Resample_HyperSp_to_MultiSp (df_HyperSp_Input, Nominal_bands, RSR_sensor, df_Resampled):
	print("Resample Rrs at Nominal band .....")
	
	for ii8 in Nominal_bands:  # Loop of Nominal bands of input sensor >> e.g., "SGLI" [380, 412, 443, 490, 530, 565, 674] >> no need to worry about float bands "e.g., 674" because RSR file is every 1nm and this bands will be use as name
	
		##*** no need to change any thing from here >>
		Hdr_ = "Rrs_" + str(ii8)      # e.g. ii8=674  >> Hdr_ = Rrs_674
		Hdr_RSR = "RSR_" + Hdr_      #e.g., RSR_Rrs_674

		print(Hdr_ , ">>", Hdr_RSR)
		
		RSR_one_Band_value = RSR_sensor.loc[ ~RSR_sensor[Hdr_RSR].isnull(),  Hdr_RSR].to_numpy()     # for ii8=674 >> select rows is not NaN, for column "RSR_Rrs_674" >> list of RSR 
		RSR_one_band_hdr   = RSR_sensor.loc[ ~RSR_sensor[Hdr_RSR].isnull(),  "wavelength"].to_list() # for ii8=674 >> select rows is not NaN, for column "wavelength"  >> [650, 651,... 673, 674, ... , 694]
		RSR_one_band_hdr = ["Rrs_"+str(ii) for ii in RSR_one_band_hdr]  # add "Rrs_" to each wavelength  >> e.g., ['Rrs_650', 'Rrs_651',... 'Rrs_673','Rrs_674', ... , 'Rrs_694']
		
		#print(RSR_one_band_hdr)
		#*** to here << 

		RSR_hdr_1st  = RSR_one_band_hdr[0]    # for ii8=674 >> RSR_hdr_1st = 650
		RSR_hdr_last = RSR_one_band_hdr[-1]   # for ii8=674 >> RSR_hdr_1st = 694
		df_Resampled[Hdr_] = np.sum(df_HyperSp_Input.loc[:, RSR_hdr_1st:RSR_hdr_last].to_numpy() * RSR_one_Band_value, axis=1) / np.sum (RSR_one_Band_value)	  # axis=1 : means calculation in rows direction
		
		# convert dtype into "float32"
		df_Resampled = df_Resampled.astype("float32")
	
	return df_Resampled

############################################################################
## Cubic Spline interpolation 
def CubicSpline_func( df_inp, hdr_Inp_List, wavlen_Inp_List, hdr_Out_List, wavlen_Out_List):

	# extract unique bands of output sensor >> e.g., inp_ ['Rrs_443', 'Rrs_490', 'Rrs_560'] out_ ["Rrs_412", "Rrs_443", "Rrs_510"]  'Rrs_443' in both Inp_ & Out_ >> unique output ["Rrs_412", "Rrs_510"]
	hdr_Out_unique = [xx for xx in hdr_Out_List if xx not in hdr_Inp_List]
		
	# add the required bands to the input dataframe
	df_inp[hdr_Out_unique] = np.nan   # make the values are NaN
	
	# sort dataframe columns' header
	df_inp = df_inp.reindex(sorted(df_inp.columns), axis=1)  ## e.g., [Rrs_443   Rrs_490  ...  Rrs_412]  >> [Rrs_412   Rrs_443   Rrs_490  ...  ] 
	
	
	hdr_all = df_inp.columns.to_list()  # extract the new columns' header  >> e.g.,  [Rrs_412   Rrs_443   Rrs_490  ...  ] 
	wavlen_all = [float(xx.split("Rrs_")[1]) for xx in hdr_all]  # convert columns' header to float  >> e.g.,  [412   443   490  ...  ] 
	df_inp.columns = wavlen_all    # assign the float list as columns' header  [412   443   490  ...  ] 
	
	###########################################################################################
	###...interpolate using CubicSpline
	# dataframe.interpolate(method='cubicspline')  provide same result as from scipy.interpolate import CubicSpline
	
	df_inp.interpolate(method='cubicspline', axis=1, inplace=True, limit_direction="both") # limit_direction="both": fill NaN in both "forward"  & "backward" direction.	 # axis=1 >> row direction	  
	
	# make the columns' header string again
	df_inp.columns = hdr_all    # assign the string list as columns' header  [Rrs_412   Rrs_443   Rrs_490  ...  ]  
	
	
	df_out = df_inp[hdr_Out_List]
	
	df_out = df_out.astype("float32") 	# convert dtype into "float32"

	return df_out

############################################################################
## Cubic Spline interpolation 
def Linear_func( df_inp, hdr_Inp_List, wavlen_Inp_List, hdr_Out_List, wavlen_Out_List):
	
	# extract unique bands of output sensor >> e.g., inp_ ['Rrs_443', 'Rrs_490', 'Rrs_560'] out_ ["Rrs_412", "Rrs_443", "Rrs_510"]  'Rrs_443' in both Inp_ & Out_ >> unique output ["Rrs_412", "Rrs_510"]
	hdr_Out_unique = [xx for xx in hdr_Out_List if xx not in hdr_Inp_List]
		
	# add the required bands to the input dataframe
	df_inp[hdr_Out_unique] = np.nan   # make the values are NaN
	
	# sort dataframe columns' header
	df_inp = df_inp.reindex(sorted(df_inp.columns), axis=1)  ## e.g., [Rrs_443   Rrs_490  ...  Rrs_412]  >> [Rrs_412   Rrs_443   Rrs_490  ...  ] 
	
	
	hdr_all = df_inp.columns.to_list()  # extract the new columns' header  >> e.g.,  [Rrs_412   Rrs_443   Rrs_490  ...  ] 
	wavlen_all = [float(xx.split("Rrs_")[1]) for xx in hdr_all]  # convert columns' header to float  >> e.g.,  [412   443   490  ...  ] 
	df_inp.columns = wavlen_all    # assign the float list as columns' header  [412   443   490  ...  ] 
	
	###########################################################################################
	###...interpolate using CubicSpline
	# dataframe.interpolate(method='cubicspline')  provide same result as from scipy.interpolate import CubicSpline
	
	###... VIP Note
	# method='linear': Ignore the distance between values and treat the values as equally spaced. 
	# method='values': use the actual numerical values of the columns' header to per
	df_inp.interpolate(method='values', inplace=True, limit_direction="both", axis=1) # limit_direction="both": fill NaN in both "forward"  & "backward" direction.	 # axis=1 >> row direction	  
	
	# make the columns' header string again
	df_inp.columns = hdr_all    # assign the string list as columns' header  [Rrs_412   Rrs_443   Rrs_490  ...  ]  
	
	
	df_out = df_inp[hdr_Out_List]
	
	df_out = df_out.astype("float32") 	# convert dtype into "float32"

	return df_out

############################################################################
###... Correction approach to reflect the difference between the input and Matchup Rrs values
def Rrs_correct_func(df_input_Ref, df_input_MatchUp, df_output_MatchUp):

	# assume wavlen_input_ary:  OLCI  [400, 412.5, 443, 490., 510., 560., 620., 665., 673.75, 681.25] 
	# assume wavlen_output_ary: AERNT [412, 442, 490, 530, 551, 668]
	Hdr_input_ary    = df_input_MatchUp.columns.to_numpy()     #e.g., [Rrs_400, Rrs_412.5, Rrs_443, Rrs_490., Rrs_510., Rrs_560., ...]
	wavlen_input_ary = np.array( [float(xx.strip("Rrs_")) for xx in Hdr_input_ary] )   #e.g., [400, 412.5, 443, 490., 510., 560., ...]

	Hdr_output_ary    = df_output_MatchUp.columns.to_numpy()   #e.g., [Rrs_412, Rrs_442, Rrs_490, Rrs_530, ....]  
	wavlen_output_ary = np.array( [float(xx.strip("Rrs_")) for xx in Hdr_output_ary] ) # e.g., [412, 442, 490, 530, ....] 

	output_Rrs_corrected = []
	for wav_out_one, hdr_out_one in zip(wavlen_output_ary, Hdr_output_ary):
		# Check if hdr_out_one is in Hdr_input_ary
		if hdr_out_one in Hdr_input_ary:
			# Directly use the Rrs values from df_input_MatchUp for this band
			Rrs_fnl = df_input_Ref[hdr_out_one].to_numpy()
			print( f" Input Bands {wav_out_one} considered " ) 
			
		else: 		
			lambda_input_idx_list  = [np.abs(wav_out_one - wavlen_input_ary).argmin()]  # index of closest wavelength from wavlen_input_ary to wav_out_one  (e.g., wav_out_one=530 >> lambda_input_idx_list=[4])
			lambda_input_val_list  = [wavlen_input_ary[lambda_input_idx_list[0]]]   # value nearest wavelength from wavlen_input_ary to wav_out_one  (e.g., wav_out_one=530 >> lambda_input_idx_list=[4] & lambda_input_val_list = 510)
			
			# in case if the Distance is too great between- use weighted average if not first / last index
			# e.g., wav_out_one=530 weighted from [510, 560]  
			if abs(wav_out_one - lambda_input_val_list[0]) > 3 and lambda_input_idx_list[0] not in [0, len(wavlen_input_ary)-1]:  # lambda_input_idx_list[0] not in [0, len(wavlen_input_ary)-1] >>means  [0, idx last element] (e.g., [0, 5]) > to exclude 1st and last element from weight 
				lambda_input_idx_list.append(lambda_input_idx_list[0] + (1 if lambda_input_val_list[0] < wav_out_one else -1)) 
				lambda_input_val_list.append(wavlen_input_ary[lambda_input_idx_list[1]])  
			
			print(wav_out_one, " > Correction Bands> ", lambda_input_val_list) # 530.0  >>  [4, 5] [510.0, 560.0]
			
	
			
			Rrs_es = []  # Rrs_es: 1st estimation >> Rrs @ output wavelength
			#for lambda_inp_idx, lambda_input_val in zip(lambda_input_idx_list, lambda_input_val_list):  # e.g., [4, 5] [510.0, 560.0] >> for wav_out_one=530
			for lambda_inp_idx in lambda_input_idx_list:  # e.g., [4, 5] >> means [510.0, 560.0] >> for wav_out_one=530
				
				Hdr_In = Hdr_input_ary[ lambda_inp_idx ]    # e.g., lambda_inp_idx = 4  >> means 510 nm
				Rrs_Inp_Ref     = df_input_Ref[Hdr_In].to_numpy()             # Rrs_Inp_Ref @ 510nm
				Rrs_Inp_Matchup = df_input_MatchUp[Hdr_In].to_numpy()         # Rrs_Inp_Matchup @ 510nm
				
				Rrs_Out_Matchup = df_output_MatchUp[hdr_out_one].to_numpy()  # Rrs_Out_Matchup @ 530nm
				
				Rrs_es.append( Rrs_Out_Matchup * (Rrs_Inp_Ref / Rrs_Inp_Matchup) )
				
			
			# Rrs_fnl: final estimation >> Rrs @ output wavelength
			if len(lambda_input_val_list) > 1:  # means use weighted average Rrs_es[0] & Rrs_es[1]    e.g., use Rrs(490) & Rrs(530) to calculate Rrs(510)
				Rrs_fnl = np.abs(lambda_input_val_list[1] - wav_out_one) * Rrs_es[0] + np.abs(lambda_input_val_list[0] - wav_out_one) * Rrs_es[1]  # Rrs_fnl is "numpy array"
				Rrs_fnl/= np.abs(lambda_input_val_list[0] - lambda_input_val_list[1])    # compute weighted average from two wavelengths
			else:
				Rrs_fnl = Rrs_es[0]    # save Rrs_es "list" into Rrs_fnl  "numpy array"


		output_Rrs_corrected.append( Rrs_fnl.tolist() )  # save correct Rrs at each wavelength 
		
	df_final = pd.DataFrame(np.transpose( np.array(output_Rrs_corrected) ), columns = Hdr_output_ary)
	df_final = df_final.astype("float32") 	# convert dtype into "float32"
	return df_final

############################################################################
###... Extract selected indices from HICO LookUp Table data
def Extract_MatchUp_HICO_Func(Ind_list):
	global df_HICO
	data_list = []
	for ind_ in Ind_list:
		data_list.append( df_HICO.iloc[ind_] )
	
	df_Extracted = pd.DataFrame( data_list, columns = df_HICO.columns.to_list()) 
	df_Extracted.reset_index( inplace=True, drop=True )  # Reset index  # drop=True: use the drop parameter to avoid the old index being added as a column.
	
	return df_Extracted 


#########################################################################################
###########################################################################################
###########################################################################################
###... Step 1: Load HICO Lookup >> Resample Input sensor

print("Reading LookUp Table dataset...")

with nc.Dataset(HICO_fn, mode='r') as dataset:
	# Retrieve all variable names ['Rrs', 'longitude', 'latitude', 'aot_868', 'angstrom', 'chlor_a', 'chl_ocx', 'Kd_490', 'pic', 'poc']
	var_all    = dataset.variables.keys()  
	HDR_NonRrs = [i for i in var_all if i != 'Rrs']

	# Create headers for Rrs data (from 353 to 719) ['Rrs_353', 'Rrs_354', ...., 'Rrs_718', 'Rrs_719']
	Rrs_Hdr = ['Rrs_' + str(i) for i in range(353, 720)]  

	# Directly create the DataFrame from the dataset
	df_HICO = pd.concat([
		 pd.DataFrame({var: dataset.variables[var][:].filled(np.nan) for var in HDR_NonRrs }),        # Products other than Rrs e.g., ['longitude', 'latitude', 'aot_868', 'angstrom', 'chlor_a', 'chl_ocx', 'Kd_490', 'pic', 'poc']
		 pd.DataFrame(dataset.variables['Rrs'][:].filled(np.nan), columns=Rrs_Hdr, dtype="float32")], # Rrs
		 axis=1 )

Min_Bnd_HICO = 353   
Max_Bnd_HICO = 719 

print(f"HICO Look-Up Table dataset {df_HICO.shape} was loaded.....")

###########################################################################################
###... Step 2:  Read Input dataset

df_Inp = pd.read_csv(Inp_data_fn)	

Hdr_Inp = np.array( [i for i in df_Inp.columns if ("Rrs_" in i ) ] )
Hdr_Inp = np.array( [i for i in Hdr_Inp if (Min_Bnd_HICO <= int(i.strip("Rrs_")) <= Max_Bnd_HICO) ] )
Bnd_Inp = np.array( [int(i.strip("Rrs_")) for i in Hdr_Inp ])   

df_Inp_Rrs = df_Inp[Hdr_Inp]

# remove station with missing data (NaN), zeros or -ve values
df_Inp_Rrs = df_Inp_Rrs.dropna(how='any', axis=0)
df_Inp_Rrs = df_Inp_Rrs[ df_Inp_Rrs.min(axis=1)>0 ]
print("Input dataset was read...")

###########################################################################################
###... Step 3:  read Relative spectral Response (RSR) files for both input & output sensors

f_n = f"Relative_Spectral_Response/{Inp_Snsr_n}_RSR_1nm.csv"  
RSR_Inp = pd.read_csv( f_n )
# filter within min and max wavelengths of HICO
RSR_Inp = RSR_Inp.loc[ (RSR_Inp['wavelength'] >= Min_Bnd_HICO) & (RSR_Inp['wavelength']<= Max_Bnd_HICO) ]		

f_n = f"Relative_Spectral_Response/{Out_Snsr_n}_RSR_1nm.csv"  
RSR_Out_Snsr = pd.read_csv( f_n )
# filter within min and max wavelengths of HICO
RSR_Out_Snsr = RSR_Out_Snsr.loc[ (RSR_Out_Snsr['wavelength']>=Min_Bnd_HICO) & (RSR_Out_Snsr['wavelength']<= Max_Bnd_HICO) ]

###########################################################################################
###... Step 4: Resample HICO LookUp table Rrs at Nominal bands  of >> Input Sensor
print("Resample HICO Look-Up Rrs at %s bands ....." %Inp_Snsr_n)
df_HICO_Rsmpl_Inp = pd.DataFrame(columns = Hdr_Inp)  # empty dataframe
df_HICO_Rsmpl_Inp = Resample_HyperSp_to_MultiSp_HICO(df_HICO , Bnd_Inp, RSR_Inp, df_HICO_Rsmpl_Inp)


###########################################################################################
###... Step 5: Prepare the header and wavelength of Output sensor
Bnd_Out = sensor_wavelengths[Out_Snsr_n]     # e.g., 'SGLI' : [380, 412, 443, 490, 530, 565, 674],
Hdr_Out = ["Rrs_"+str(ii) for ii in Bnd_Out] # e.g., ["Rrs_380", "Rrs_412", "Rrs_443", ... "Rrs_674"]

###########################################################################################
###... Step 6: MatchUp data between (HICO resampled) and (Input_Dataset) 
print("Match up data between Input_Dataset and HICO LookUp Table Resampled at input sensor...")
No_Inp_Data = df_Inp_Rrs.shape[0]
No_Stn_Process_Loop = 100    # in case if shape of [lookUp table] reaches (3,000,000 * 56 ) will be difficult to increase no. > 600

Indx_min_LUT = np.empty( No_Inp_Data, dtype="int32" )

if 	No_Inp_Data <= No_Stn_Process_Loop :
	Indx_min_LUT = SpectralMatch_Broadcast_Func( df_Inp_Rrs.to_numpy(dtype="float32"), df_HICO_Rsmpl_Inp.to_numpy(dtype="float32"), CostFunction_LUT )
else:
	for ii4 in range(0, No_Inp_Data, No_Stn_Process_Loop):
		print(ii4, "   out of  ", No_Inp_Data)
		Indx_min_LUT[ii4:ii4+No_Stn_Process_Loop] = SpectralMatch_Broadcast_Func( df_Inp_Rrs[ii4:ii4+No_Stn_Process_Loop].to_numpy(dtype="float32"), df_HICO_Rsmpl_Inp.to_numpy(dtype="float32"), CostFunction_LUT  )


df_MatchUp_HyperSp = Extract_MatchUp_HICO_Func(Indx_min_LUT)
print("HICO MatchUp Extracted...") 

###########################################################################################
###... Step 6: Calculated Resampled MatchUp Rrs data at Input & Output bands 
df_MatchUp_Rsmpl_Inp = df_HICO_Rsmpl_Inp.loc[Indx_min_LUT, :]

df_MatchUp_Rsmpl_Out = pd.DataFrame(columns = Hdr_Out)  # Create empty dataframe
df_MatchUp_Rsmpl_Out = Resample_HyperSp_to_MultiSp(df_MatchUp_HyperSp.copy() ,       Bnd_Out,   RSR_Out_Snsr,  df_MatchUp_Rsmpl_Out)
print("HICO Resampled to Input & Output bands...") 

###########################################################################################
###... Step 7: Correction approach to reflect the difference between the input and Matchup Rrs values
df_MatchUp_Rsmpl_Out = Rrs_correct_func(df_Inp_Rrs.copy(), df_MatchUp_Rsmpl_Inp.copy(), df_MatchUp_Rsmpl_Out.copy())


###########################################################################################
###... Step 8: Consider the Linear interpolation for 380 nm band
if ("Rrs_380" in df_MatchUp_Rsmpl_Out.columns) :
	LNR_output_df = Linear_func( df_Inp_Rrs.copy(), Hdr_Inp, Bnd_Inp, Hdr_Out, Bnd_Out )
	df_MatchUp_Rsmpl_Out["Rrs_380"] = LNR_output_df["Rrs_380"].to_numpy()

###########################################################################################
###... Step 8: Save  original Input_Dataset & Matchup HICO Hyperspectral data  	
# adjust the header for HICO Matchup. 
Hdr_match_list = np.array([xx + "_HICO" for xx in df_MatchUp_Rsmpl_Out.columns]) # e.g., "SGLI" ["Rrs_380_HICO", "Rrs_412_HICO", "Rrs_443_HICO", ... "Rrs_674_HICO"]
pd.concat([ df_Inp_Rrs, pd.DataFrame(df_MatchUp_Rsmpl_Out.to_numpy(), columns = Hdr_match_list) ], axis=1).to_csv(Output_fn, index=False, float_format='%.10f')
