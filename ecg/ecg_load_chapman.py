import os 
import numpy as np 
import pandas as pd 
import wfdb


work_dir = "C:/Users/user/Dropbox/AMI_EXT_VAL/Data/chapman"
os.chdir(work_dir)
os.getcwd()


# chapman data (raw)
record_list = pd.read_csv('./RECORDS', header=None)
record_list = record_list.rename(columns = {0 : 'path'})

data = [pd.read_csv('./'+f+'RECORDS', header=None) for f in record_list.path]
file_path = []

for i in range(0,len(record_list)):
    file = data[i].rename(columns = {0 : 'file_name'})
    path = record_list.path[i] + file.file_name
    file_path.append(path)

file_path = np.concatenate(file_path)

snomed_ct = pd.read_csv('./ConditionNames_SNOMED-CT.csv')
chapman_diagnostics = pd.DataFrame()

for file in file_path:
    rd_record = wfdb.rdrecord(file)
    # wfdb.plot_wfdb(record=rd_record, figsize=(24,18), title='', ecg_grids='all')

    sig = wfdb.rdsamp(file)
    header = sig[1]
    
    dx = next(item for item in header['comments'] if item.startswith('Dx:'))
    dx = dx.split('Dx: ')[1]
    dx = dx.split(',')
    dx = list(map(int, dx))
    
    diag = snomed_ct[snomed_ct['Snomed_CT'].isin(dx)]
    diag.insert(0, 'Filename', file)
    chapman_diagnostics = pd.concat([chapman_diagnostics, diag])


# save file
chapman_diagnostics.to_csv('chapman_diagnostics.csv', index=False)
