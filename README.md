# Unique HICO Rrs Dataset
- This script is designed to 

  ## Clone the repository
```
git clone https://github.com/RS-ML-Hub/SMTH_Rrs_BandShifting.git
```

## Dependencies
This code was developed using Windows 11 and Python 3.10. We recommend creating a new conda environment and installing the required dependencies with the following commands:
```
conda env create -f environment.yml
conda activate HICO_env
```

## Read HICO Dataset
```
python Read_HICO_NetCDF_Data.py
```
The script will store HICO Dataset as Dataframe

## Run SMTH to conduct band-shifting

### Script Workflow
1.**Inp_Snsr_n**: Edit the **YYYYMMDD_s** and **YYYYMMDD_e** variables in **HABs_Data_Downloader.py** to set the start and end dates.
- **YYYYMMDD_s**: Start date in "YYYY/MM/DD" format. (e.g., "2023/01/01")
- **YYYYMMDD_e**: End date in "YYYY/MM/DD" format.   (e.g., "2023/09/30")
```
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
```

3. **Run the script**
```
python HABs_Data_Downloader.py
```

3. **CSV Output**: A CSV file named HABs__{YYYYMMDD_s}_{YYYYMMDD_e}.csv (e.g., HABs__20230101_20230930.csv) will be generated in the script's directory.

