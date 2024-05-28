import os 
import numpy as np 
import xml.etree.ElementTree as ET 
from tqdm import tqdm 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import wfdb


intervals = {
    1: (0, 1250),
    2: (0, 1250),
    3: (0, 1250),
    4: (1250, 2500),
    5: (1250, 2500),
    6: (1250, 2500),
    7: (2500, 3750),
    8: (2500, 3750),
    9: (2500, 3750),
    10: (3750, 5000),
    11: (3750, 5000),
    12: (3750, 5000),
    13: (0, 5000)
}


def _ax_plot(ax, x, y, secs=10, lwidth=0.8, amplitude_ecg = 1.8, time_ticks =0.2): 
    ax.set_xticks(np.arange(0,11,time_ticks))    
    ax.set_yticks(np.arange(-ceil(amplitude_ecg),ceil(amplitude_ecg),1.0))

    #ax.set_yticklabels([])
    #ax.set_xticklabels([])

    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5)) 
    ax.set_ylim(-amplitude_ecg, amplitude_ecg) 
    ax.set_xlim(0, secs) 
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')  
    ax.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))  
    
    
    for spine in ax.spines.values(): 
        spine.set_visible(False)  

    ax.plot(x,y, linewidth=lwidth, color='black')


def plot_ecg(ecg_data, xml_name=None):
    # speed      : signal speed on display, defaults to 50 mm / sec  -> 25 
    # voltage    : signal voltage on display, defaults to 20 mm / mV -> 10 
    subplot_texts1 = ["I", "II", "III", "II"]  
    subplot_texts2 = ["aVR", "aVL", "aVF", ""] 
    subplot_texts3 = ["V1", "V2", "V3", ""] 
    subplot_texts4 = ["V4", "V5", "V6", ""] 

    sample_rate = 500 
    lead_index = ['I', 'II', 'III', 'II']  # 의미 없음
    lead_order = [0,1,2,3] 
    columns=1 
    speed = 25 #mm/sec
    voltage = 10 #mm/mV
    line_width = 0.6 

    leads = len(lead_order)  
    seconds = len(ecg_data[0])/sample_rate  
    
    plt.rcParams.update({'font.size': 8}) 
    fig, ax = plt.subplots( 
        ceil(len(lead_order)/columns),columns, 
        sharex=True, 
        sharey=True,
        figsize=((speed/25.4)*seconds*columns/0.4/2,  # 
                 (4*voltage/25.4)*leads/columns/2)     # 1 subplot -> (-2,2) mV
        )
    fig.subplots_adjust(
        hspace = 0, 
        wspace = 0.04,
        left   = 0.04,  # left side of the subplots 
        right  = 0.98,  # right side of the subplots 
        bottom = 0.06,  # bottom 
        top    = 0.95
        )
    # fig.suptitle(title) 

    step = 1.0/sample_rate 

    for i in range(0, len(lead_order)):  
        if(columns == 1): 
            t_ax = ax[i] 
        else:
            t_ax = ax[i//columns,i%columns] 
        t_lead = lead_order[i]
        t_ax.set_ylabel(lead_index[t_lead])
        t_ax.tick_params(axis='x',rotation=90)
        
        _ax_plot(t_ax, np.arange(0, len(ecg_data[t_lead])*step, step), ecg_data[t_lead], seconds)
        t_ax.text(0.02, 0.85, subplot_texts1[i], transform=t_ax.transAxes, fontsize=12, va='center', ha='center')
        t_ax.text(0.27, 0.85, subplot_texts2[i], transform=t_ax.transAxes, fontsize=12, va='center', ha='center')
        t_ax.text(0.52, 0.85, subplot_texts3[i], transform=t_ax.transAxes, fontsize=12, va='center', ha='center')
        t_ax.text(0.77, 0.85, subplot_texts4[i], transform=t_ax.transAxes, fontsize=12, va='center', ha='center')
    
    # plt.savefig(f'./UKB_np/{xml_name}_plot.png') 
    plt.figure(figsize = (15,6))
    plt.show()


def load_chapman_data(df, path):

    file = df['Filename'].unique()
    data = [wfdb.rdsamp(path+f) for f in file]
    data = np.array([signal for signal, meta in data])

    return data


chapman_rhythm = pd.read_csv('./Data/chapman/chapman_diagnostics.csv')
def to_np_chapman(chapman_df, scale=None, np_save=False, show_plot=False, show_plot_data=None, random_sample=False, fixed=None):

    file_path = chapman_rhythm['Filename'].unique()

    if show_plot_data is not None:
        if random_sample:
            random_file = np.random.choice(file_path, size = show_plot_data)
            indices = [np.where(file_path == value)[0][0] for value in random_file]
            data = chapman_df[indices]
            filename = random_file

        else:
            data = chapman_df[:show_plot_data]
            filename = file_path[:show_plot_data]

    if fixed is not None:
        indices = [np.where(file_path == value)[0][0] for value in fixed]
        data = chapman_df[indices]
        filename = file_path[indices]

    for i in range(0, len(data)): 
        ecg_np = data[i]
        # ecg_np = ecg_np * 0.001
        ecg_np = ecg_np.T  # Shape을 (12, 5000)으로 변환 
        
        ecg_name = filename[i].split('/')[3]
        rhythm_tag = chapman_rhythm.loc[chapman_rhythm['Filename'] == filename[i], 'Acronym Name'].values

        if scale is not None:
            ecg_np = ecg_np * scale
            ecg_name = ecg_name+'_'+str(scale)     

        if np_save:
            np.save(f'./Results/chapman/npy/{ecg_name}.npy', ecg_np) 
        
        if show_plot:
            
            extracted_data = np.zeros((4, 5000))  

            for row in range(1, 14): 
                if row in [1, 4, 7, 10]: 
                    start, end = intervals[row] 
                    extracted_data[0, start:end] = ecg_np[row-1, start:end] 
                elif row in [2, 5, 8, 11]: 
                    start, end = intervals[row] 
                    extracted_data[1, start:end] = ecg_np[row-1, start:end] 
                elif row in [3, 6, 9, 12]: 
                    start, end = intervals[row] 
                    extracted_data[2, start:end] = ecg_np[row-1, start:end] 
                elif row ==13:
                    start, end = intervals[row] 
                    extracted_data[3, start:end] = ecg_np[1, start:end] # lead 2 

            print(f"ECG plot for {ecg_name}")
            print(f"ECG Diagnosed rhythm: {rhythm_tag}") 
            plot_ecg(extracted_data, ecg_name)


def load_ptbxl_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def to_np_ptbxl(X, Y, scale=None, np_save=False, show_plot=False, show_plot_data=None, random_sample=False, fixed=None):

    file_path = Y['filename_hr'].unique()

    if show_plot_data is not None:
        if random_sample:
            random_file = np.random.choice(file_path, size = show_plot_data)
            indices = [np.where(file_path == value)[0][0] for value in random_file]
            data = X[indices]
            filename = random_file

        else:
            data = X[:show_plot_data]
            filename = file_path[:show_plot_data]

    if fixed is not None:
        indices = [np.where(file_path == value)[0][0] for value in fixed]
        data = X[indices]
        filename = file_path[indices]
    
    for i in range(0, len(data)):
        ecg_np = data[i] 
        ecg_np = ecg_np.T  # Shape을 (12, 5000)으로 변환 
        
        ecg_name = filename[i].split('/')[2]
        rhythm_tag = Y[Y.filename_hr == filename[i]].report.values[0]

        if scale is not None:
            ecg_np = ecg_np * scale
            ecg_name = ecg_name+'_'+str(scale)    

        if np_save:
            np.save(f'./Results/ptbxl/npy/{ecg_name}.npy', ecg_np) 

        if show_plot:
            
            extracted_data = np.zeros((4, 5000))  

            for row in range(1, 14): 
                if row in [1, 4, 7, 10]: 
                    start, end = intervals[row] 
                    extracted_data[0, start:end] = ecg_np[row-1, start:end] 
                elif row in [2, 5, 8, 11]: 
                    start, end = intervals[row] 
                    extracted_data[1, start:end] = ecg_np[row-1, start:end] 
                elif row in [3, 6, 9, 12]: 
                    start, end = intervals[row] 
                    extracted_data[2, start:end] = ecg_np[row-1, start:end] 
                elif row ==13:
                    start, end = intervals[row] 
                    extracted_data[3, start:end] = ecg_np[1, start:end] # lead 2 
                
            print(f"ECG plot for {ecg_name}") 
            print(f"ECG Diagnosed rhythm: {rhythm_tag}") 
            plot_ecg(extracted_data, ecg_name) 


def ptbxl_wfdb_plot(path, Y, show_plot_data):
    record_list = Y.iloc[show_plot_data].filename_hr
    rd_record = wfdb.rdrecord(path+record_list)

    print(record_list)
    wfdb.plot_wfdb(record=rd_record, figsize=(24,18), title='', ecg_grids='all')


def chapman_wfdb_plot(path, rhythm, show_plot_data):
    file_path = rhythm['Filename'].unique()
    record_list = file_path[show_plot_data]
    rd_record = wfdb.rdrecord(path+record_list)

    print(record_list)
    wfdb.plot_wfdb(record=rd_record, figsize=(24,18), title='', ecg_grids='all')



