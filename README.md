# Unique HICO Rrs Dataset
- This script is designed to download in situ data related to Harmful Algal Blooms (HABs) from the Akashiwo "red tide" network, managed by the Fisheries Agency of Japan.  
- The script automates the process of finding and retrieving available HABs data within a specified date range and collects Ancillary Data from each station.
  
## Dependencies
This code was developed using Windows 11 and Python 3.10. We recommend creating a new conda environment and installing the required dependencies with the following commands:
```
conda env create -f environment.yml
conda activate HICO_env
```

## Running the code
- **Clone the repository**
```
git clone https://github.com/RS-ML-Hub/SMTH_Rrs_BandShifting.git
```

## Script Workflow
1.**Configure Dates**: Edit the **YYYYMMDD_s** and **YYYYMMDD_e** variables in **HABs_Data_Downloader.py** to set the start and end dates.
- **YYYYMMDD_s**: Start date in "YYYY/MM/DD" format. (e.g., "2023/01/01")
- **YYYYMMDD_e**: End date in "YYYY/MM/DD" format.   (e.g., "2023/09/30")

3. **Run the script**
```
python HABs_Data_Downloader.py
```

3. **CSV Output**: A CSV file named HABs__{YYYYMMDD_s}_{YYYYMMDD_e}.csv (e.g., HABs__20230101_20230930.csv) will be generated in the script's directory.

