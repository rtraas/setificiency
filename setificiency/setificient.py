import setigen as stg
import os
import pandas as pd
import numpy as np
from blimpy import Waterfall
import time
import astropy.units as u
from turbo_seti.find_doppler.find_doppler import FindDoppler
from turbo_seti.find_event.find_event import read_dat
import logging
import multiprocessing as mp




data_dir_l = '../seti_tess/l_band/'
data_dir_s = '../seti_tess/s_band/'
data_dir_c = '../seti_tess/c_band/'
data_dir_x = '../seti_tess/x_band/'


test_dir_l = './test_files/l_band/'
test_dir_s = './test_files/s_band/'
test_dir_c = './test_files/c_band/'
test_dir_x = './test_files/x_band/'

import signal_funcs 



def get_data_files(band='L'):
    """
    Obtains paths to data files
    """
    if band.upper() == 'L':
        return [data_dir_l+f for f in os.listdir(data_dir_l) if '.0000.h5' in f]
    elif band.upper() == 'S':
        return [data_dir_s+f for f in os.listdir(data_dir_s) if '.0000.h5' in f]
    elif band.upper() == 'C':
        return [data_dir_c+f for f in os.listdir(data_dir_c) if '.0000.h5' in f]
    elif band.upper() == 'X':
        return [data_dir_x+f for f in os.listdir(data_dir_x) if '.0000.h5' in f]

def get_test_files(band):
    """
    Obtains paths to test files
    """
    if band.lower() == 'l':
        return [test_dir_l+f for f in os.listdir(test_dir_l)]
    if band.lower() == 's':
        return [test_dir_s+f for f in os.listdir(test_dir_s)]
    if band.lower() == 'c':
        return [test_dir_c+f for f in os.listdir(test_dir_c)]
    if band.lower() == 'x':
        return [test_dir_x+f for f in os.listdir(test_dir_x)]


def generate_test_file(waterfall_fn, band, power_of_2=9, tchans=None, out_dir='test_files/'):
    """
    Creates a file from an existing file on which further 
    efficiency tests can be conducted.
    
    Creates a generator that returns smaller Waterfall objects by 'splitting'
    an input filterbank file according to the number of frequency samples.
    Since this function only loads in data in chunks according to fchans,
    it handles very large observations well. Specifically, it will not attempt
    to load all the data into memory before splitting, which won't work when
    the data is very large anyway.
    Parameters
    ----------
    waterfall_fn : str
        Filterbank filename with .fil extension
    fchans : int
        Number of frequency samples per new filterbank file
    tchans : int, optional
        Number of time samples to select - will default from start of observation.
        If None, just uses the entire integration time
    f_shift : int, optional
        Number of samples to shift when splitting filterbank. If
        None, defaults to `f_shift=fchans` so that there is no
        overlap between new filterbank files
    Returns
    -------
    """
    fchans=1048576//(2**power_of_2)
    if band.lower() not in ['l','s','c','x']:
        raise ValueError(f"Invalid frequency band '{band.lower()}'")
    out_dir = out_dir + band.lower() + '_band/'

    info_wf = Waterfall(waterfall_fn, load_data=False)
    fch1 = info_wf.header['fch1']
    nchans = info_wf.header['nchans']
    df = info_wf.header['foff']
    tchans_tot = info_wf.container.selection_shape[0]


    if tchans is None:
        tchans = tchans_tot
    elif tchans > tchans_tot:
        raise ValueError('tchans value must be less than the total number of \
                          time samples in the observation')

    # Note that df is negative!
    f_start, f_stop = fch1, fch1 + fchans * df

    # Iterates down frequencies, starting from highest
    if np.abs(f_stop - fch1) <= np.abs(nchans * df):
        fmin, fmax = np.sort([f_start, f_stop])
        waterfall = Waterfall(waterfall_fn,
                              f_start=fmin,
                              f_stop=fmax,
                              t_start=0,
                              t_stop=tchans)
        waterfall.write_to_hdf5(f"{out_dir}_fchans_{fchans}_testing_{waterfall_fn.split('/')[-1]}")

def setificiency(
        filename, 
        fchans=1048576, 
        drift_rate_step=0.5, 
        drift_min=-10.0, 
        drift_max=10.0,
        snr_max=40, 
        snr_min=1, 
        snr_step=0.5, 
        width_min=0.5, 
        width_max=20.0, 
        width_step=0.5, 
        out_dir='./results'):
    """
    turbo_seti efficiency test

    Record all injected signals into a database.  
    Record all detections in a separate database.
    Use matched filtering to determine recovered signals.


    """
    drifts = np.arange(drift_min, drift_max, step=drift_rate_step)
    snrs = np.arange(snr_min, snr_max, step=snr_step)
    widths = np.arange(width_min, width_max, step=width_step)

    signal_params = {
        'InjectedFreq':[],
        'InjectedDriftRate':[],
        'InjectedSNR':[],
        'InjectedIntensity':[],
        'InjectedWidth':[]
                    }

    estimated_num_iterations = len(drifts)*len(snrs)*len(widths)
    time_per_iteration = 0
    results = []
    counter=0

    for i, drift_rate in enumerate(drifts):
        for j, snr in enumerate(snrs):
            for k, width in enumerate(widths):
                counter += 1
                t0 = time.time()
                string = "-"*len(f"Performing iteration {counter}")*5
                print(string)
                print(f"Performing iteration {counter} of {estimated_num_iterations}")
                print(f"Drift iteration {i} of {len(drifts)}, current drift rate: {drift_rate} [Hz/s]")
                print(f"SNR iteration {j} of {len(snrs)}, current SNR: {snr} [arbitrary units]")
                print(f"Width iteration {k} of {len(widths)}, current width: {width} [Hz]")
                if counter != 0:
                    print(f"Estimated time remaining: {((estimated_num_iterations - counter) * (time_per_iteration/60)) : 2.2f} minutes")
                print("\n\n")
                frame = stg.Frame(filename)
                
                f_start = frame.get_frequency(frame.fchans//2)
                


                frame.add_constant_signal(
                        f_start = f_start,
                        drift_rate = drift_rate * u.Hz / u.s,
                        level = frame.get_intensity(snr=snr),
                        width = width * u.Hz)
                
                signal_params['InjectedFreq'].append(f_start)
                signal_params['InjectedDriftRate'].append(drift_rate)
                signal_params['InjectedSNR'].append(snr)
                signal_params['InjectedIntensity'].append(frame.get_intensity(snr=snr))
                signal_params['InjectedWidth'].append(width)
        
                h5_file_save_name = f"{counter}_{filename.split('/')[-1]}"
                frame.save_hdf5(h5_file_save_name)
                FindDoppler(h5_file_save_name, max_drift=abs(drift_rate)+0.5, snr=snr+10, log_level_int=logging.WARNING).search()
                hits_found = read_dat(h5_file_save_name.replace('.h5','.dat'))
                results.append(hits_found)
                os.remove(h5_file_save_name)
    signal_params_df = pd.DataFrame.from_dict(signal_params)
    all_results = pd.concat(results, ignore_index=True)
    signal_params_df.to_csv(f"{out_dir}_new_injected_{filename.split('/')[-1].replace('.h5','.csv')}")
    all_results.to_csv(f"{out_dir}_new_detected_{filename.split('/')[-1].replace('.h5','.csv')}")
    
    return signal_params_df, all_results

def get_signal(frame,frame_start_index, drift_rate, snr, width, signal_path='constant', **kwargs):
    """
    Injects signals based on conditions
    """
    if bool(kwargs):
        if kwargs.haskey('spread') or kwargs.haskey('spread_type') or kwargs.haskey('rfi_type'):
            return signal_funcs.rfi_path(frame, frame_start_index, drift_rate, snr, width, spread=kwargs['spread'], spread_type=['spread_type'], rfi_type=['rfi_type'])
    if signal_path=='constant':
        return signal_funcs.constant_path(frame, frame_start_index, drift_rate, snr, width)




def new_frame(filename, drift_rate, snr, width, iteration, injected_share, share):
    """
    Creates a new Frame and conducts an efficiency test

    Parameters
    ----------
    filename : str
        path to file containing waterfall data
    drift_rate : float
        drift rate of injected signal
    snr : float
        snr of injected signal
    width : float
        width of injected signal
    iteration : int
        iteration number to use for file names
    share : mp.Manager.list
        list to append results to

    Notes
    -----
    Used as a target of a multiprocessing Process
    """
    signal_params = injected_share
    counter = iteration
    frame = stg.Frame(filename)
    #frame.add
    f_start = frame.get_frequency(frame.fchans//2)
    frame.add_constant_signal(
                        f_start = f_start,
                        drift_rate = drift_rate * u.Hz / u.s,
                        level = frame.get_intensity(snr=snr),
                        width = width * u.Hz)
        
    signal_params['InjectedFreq'].append(f_start)
    signal_params['InjectedDriftRate'].append(drift_rate)
    signal_params['InjectedSNR'].append(snr)
    signal_params['InjectedIntensity'].append(frame.get_intensity(snr=snr))
    signal_params['InjectedWidth'].append(width)
        
    h5_file_save_name = f"{counter}_{filename.split('/')[-1]}"
    frame.save_hdf5(h5_file_save_name)
    FindDoppler(h5_file_save_name, max_drift=abs(drift_rate)+0.5, snr=abs(snr)+10, log_level_int=logging.WARNING).search()
    hits_found = read_dat(h5_file_save_name.replace('.h5','.dat'))
    os.remove(h5_file_save_name)
    share.append(hits_found)

def multi_setificiency(
        filename, 
        fchans=1048576, 
        drift_rate_step=0.5, 
        drift_min=-10.0, 
        drift_max=10.0,
        snr_max=40, 
        snr_min=1, 
        snr_step=0.5, 
        width_min=0.5, 
        width_max=20.0, 
        width_step=0.5, 
        out_dir='./results'):
    """
    turbo_seti efficiency test

    Record all injected signals into a database.  
    Record all detections in a separate database.
    Use matched filtering to determine recovered signals.


    """
    manager = mp.Manager()
    injected = manager.dict()
    injected['InjectedFreq']=[]
    injected['InjectedDriftRate']=[]
    injected['InjectedSNR']=[]
    injected['InjectedIntensity']=[]
    injected['InjectedWidth']=[]
    results = manager.list()

    drifts = np.arange(drift_min, drift_max, step=drift_rate_step)
    snrs = np.arange(snr_min, snr_max, step=snr_step)
    widths = np.arange(width_min, width_max, step=width_step)

    signal_params = {
        'InjectedFreq':[],
        'InjectedDriftRate':[],
        'InjectedSNR':[],
        'InjectedIntensity':[],
        'InjectedWidth':[]
                    }

    estimated_num_iterations = len(drifts)*len(snrs)*len(widths)
    time_per_iteration = 0
    counter=0
    procs = []

    for i, drift_rate in enumerate(drifts):
        for j, snr in enumerate(snrs):
            for k, width in enumerate(widths):
                counter += 1
                t0 = time.time()
                string = "-"*len(f"Performing iteration {counter}")*5
                print(string)
                print(f"Performing iteration {counter} of {estimated_num_iterations}")
                print(f"Drift iteration {i} of {len(drifts)}, current drift rate: {drift_rate} [Hz/s]")
                print(f"SNR iteration {j} of {len(snrs)}, current SNR: {snr} [arbitrary units]")
                print(f"Width iteration {k} of {len(widths)}, current width: {width} [Hz]")
                #if counter != 0:
                #    print(f"Estimated time remaining: {((estimated_num_iterations - counter) * (time_per_iteration/60)) : 2.2f} minutes")
                print("\n\n")
                p = mp.Process(target=new_frame, args=(filename, drift_rate, snr, width, counter, injected, results,))
                procs.append(p)
                p.start()
                #t1 = time.time()
                #time_per_iteration = t1 - t0
    for proc in procs:
        proc.join()

    signal_params_df = pd.DataFrame.from_dict(injected)
    all_results = pd.concat(results, ignore_index=True)
    signal_params_df.to_csv(f"{out_dir}injected_{filename.split('/')[-1].replace('.h5','.csv')}")
    all_results.to_csv(f"{out_dir}detected_{filename.split('/')[-1].replace('.h5','.csv')}")
    
    return signal_params_df, all_results


if __name__=='__main__':
    test_file = get_test_files('x')[0]
    setificiency(test_file)
    #_, __ = multi_setificiency(test_file)

