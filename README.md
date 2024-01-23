# Unique HICO Rrs Dataset

  ## HICO Dataset Overview:
This dataset encompasses a unique collection of remote-sensing reflectance (Rrs) data, as detailed in the paper "**Spectral band-shifting of multispectral remote-sensing reflectance products: Insights for matchup and cross-mission consistency assessments**" by Salem et al., published in 2023 in the Remote Sensing of Environment Journal. It comprises approximately 6.2 million unique Rrs records spanning the spectral range of 353–719 nm. These records are extracted from 8,893 images captured by the Hyperspectral Imager for the Coastal Ocean (HICO) between 2009 and 2014.

  ## HICO Dataset and Pre-processing:
The HICO sensor, mounted on the International Space Station, collected hyperspectral data specifically for coastal ocean observation. It featured 65 spectral bands between 353 and 719 nm with a 5.7 nm sample interval in the open ocean, coastal waters, estuaries, and shallow regions.

The HICO images were processed to filter out invalid or negative Rrs values across the 65 spectral bands of HICO data. An iterative process was employed to refine and extract unique Rrs data, resulting in a novel look-up table (LUT) consisting of nearly 6.2 million unique Rrs records. The LUT's Rrs spectra were further refined using cubic spline interpolation to adjust the sampling interval from 5.7 nm to 1 nm, resulting in a comprehensive LUT of 6,335,988 records spanning 367 bands (353–719 nm). 

In addition to the Rrs data, the dataset includes 9 ancillary products:

- '**aot_868**': Aerosol optical thickness at 868 nm
- '**angstrom**': Aerosol Angstrom exponent (443 to 865 nm)
- '**chlor_a**': Chlorophyll concentration (mg m^-3) using the OCI Algorithm
- '**chl_ocx**': Chlorophyll concentration (mg m^-3) using the OC4 Algorithm
- '**Kd_490**': Diffuse attenuation coefficient at 490 nm (m^-1) using the KD2 algorithm
- '**pic**': Calcite concentration (mol m^-3) by Balch and Gordon
- '**poc**': Particulate organic carbon (mg m^-3) by D. Stramski, 2007
- '**longitude**': Geographical longitude
- '**latitude**': Geographical latitude

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
1.Configure the following variables:
- **Inp_data_fn**: Edit the file name that include input data for band-shifting.
- **Inp_Snsr_n**: Edit the name of input sensor (e.g., "AERNT").
- **Out_Snsr_n**: Edit the name of output sensor (e.g., "OLCI").

**Note**: The details of sensors names and sensors wavelengths are included in sensor_labels and sensor_wavelengths, respectively.
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

```

2. **Run the script**
```
python SMTH_Rrs_BandShift.py
```

3. The output file name will include the Out_Snsr_n. For instance, if the (Out_Snsr_n = "**OLCI**"), the output file will be (Result_HICOMatchup_Resample**OLCI**.csv)

For inquiries, please contact: eng.salemsalem@gmail.com  or salem.ibrahim@kuas.ac.jp
