# generator functions to simplify and streamline signal injection and recovery of filterbanks
import setigen as stg
import numpy as np
import pandas as pd
import os
import astropy.units as u
from turbo_seti.find_doppler.find_doppler import FindDoppler
from turbo_seti.find_event.find_event import read_dat
from blimpy import Waterfall
import math
import matplotlib.pyplot as plt

from setificient import get_test_files, get_data_files, generate_test_file

import logging
logger_name = 'multigen'
logger = logging.getLogger(logger_name)

import multigen_analysis

def blueprint():
    s = """#waterfalls = stg.split_waterfall_generator()
        # opt for one small portion of a waterfall file to make this faster!!
        parameters = get_parameters(f_range, d_range, s_range, w_range) DONE
        # waterfalls = (slice_waterfall(w, params) for params in parameters)
        frames = (inject_frame(p) for p in parameters) DONE
        finddoppler_results = (find_doppler_frame(frame) for frame in frames) DONE
        #results = (get_results(fd) for fd in finddopplers)
        final_results = pd.concat(finddoppler_results)
        """
    print(s)

#def uniform_dist(v0, v1, n):
#    """
#    Returns generator of a uniform distribution
#    """
#    return (v for v in np.linspace(v0, v1, n))
def all_dist(
        nsamples,
        f_min, 
        f_max,
        d_min=-10.0,
        d_max=10.0,
        s_min=30.0,
        s_max=30.0,
        w_min=4.0,
        w_max=4.0, 
        nsignals=2):
    """
    Generates values by stepping through the 
    input parameter ranges.  Ensures the entire 
    range is spanned.
    """
    counter = 0
    while counter < nsamples*nsignals:
        f = np.linspace(f_min, f_max, nsamples*(nsignals))[counter:counter+nsignals]
        d = np.linspace(d_min, d_max, nsamples*(nsignals))[counter:counter+nsignals]
        s = np.linspace(s_min, s_max, nsamples*(nsignals))[counter:counter+nsignals]
        w = np.linspace(w_min, w_max, nsamples*(nsignals))[counter:counter+nsignals]
        #f = np.linspace(f_min, f_max, nsamples*nsignals)[nsamples:nsamples-nsignals]
        #d = np.linspace(d_min, d_max, nsamples*nsignals)[nsamples:nsamples-nsignals]
        #s = np.linspace(s_min, s_max, nsamples*nsignals)[nsamples:nsamples-nsignals]
        #w = np.linspace(w_min, w_max, nsamples*nsignals)[nsamples:nsamples-nsignals]
        yield (f, d, s, w)
        #nsamples -= nsignals
        counter += nsignals


def parameters_generator(
        nsamples,
        f_min, 
        f_max, 
        d_min=-10.0, 
        d_max=10.0, 
        s_min=20.0, 
        s_max=40.0,
        w_min=1.0,
        w_max=40.0,
        f_dist='uniform',
        d_dist='uniform',
        s_dist='uniform',
        w_dist='uniform',
        nsignals=2,
        dt=0.0,
        fchans=1024,
        tchans=16):
    """
    Generator of parameter values for signal injection
    
    Parameters
    ----------
    nsamples : int
        Number of times to sample the parameter distributions
    f_min : float
        Minimum injection frequency
    f_max : float
        Maximum injection frequency
    d_min : float, optional
        Minimum injection drift
    d_max : float, optional
        Maximum injection drift
    s_min : float, optional
        Minimum injection SNR
    s_max : float, optional
        Maximum injection SNR
    w_min : float, optional
        Minimum injection width
    w_max : float, optional
        Maximum injection width
    f_dist : str, optional
        Injection frequency distribution ('uniform', 'gaussian', 'random')
    d_dist : str, optional
        Injection drift distribution ('uniform', 'gaussian', 'random')
    s_dist : str, optional
        Injection SNR distribution ('uniform', 'gaussian', 'random')
    w_dist : str, optional
        Injection width distribution ('uniform', 'gaussian', 'random')
    nsignals : int, optional
        Number of signals to inject

    Notes
    -----
    You can replace any distribution given in numpy.random 
    for the frequency, drift rate, snr, and/or width distributions.
    """
    #random.uniform(-1, 1)*(drift*frame.dt*frame.tchans
    #separation = 
    f_dist, d_dist, s_dist, w_dist = [eval("np.random."+dist) for dist in [f_dist, d_dist, s_dist, w_dist]]
    while nsamples > 0:
        f_vals = f_dist(low=f_min, high=f_max, size=nsignals)
        d_vals = d_dist(low=d_min, high=d_max, size=nsignals)
        s_vals = s_dist(low=s_min, high=s_max, size=nsignals)
        w_vals = w_dist(low=w_min, high=w_max, size=nsignals)
        if nsignals > 1:
            f_vals[1] = f_vals[0] + 5000 + np.random.uniform(-2,2)*1000#(np.random.uniform(-1,1)*(d_vals[0]*dt*tchans))
            d_vals[1] = d_vals[0] + (np.random.uniform(-1,1)*5.0)
        yield (
            f_vals,
            d_vals, 
            s_vals,
            w_vals
            )
        nsamples -= 1


def Dtest_generator(
        nsamples_per_drift,
        f_min,
        f_max,
        d_min=-10.0,
        d_max=10.0,
        s_min=30.0,
        s_max=30.0,
        w_min=4.0,
        w_max=4.0,
        f_dist='uniform',
        d_dist='uniform',
        s_dist='uniform',
        w_dist='uniform',
        nsignals=1):
    """
    Generates parameters for drift rate tests
    """
    f = float(np.mean([f_min, f_max]))
    s = float(np.mean([s_min, s_max]))
    w = float(np.mean([w_min, w_max]))
    d_vals = np.array([np.full(nsamples_per_drift, i) for i in np.arange(d_min, d_max, 0.5)]).flatten()
    nsamples = len(d_vals)
    counter=0
    while counter < nsamples:
        yield (f, d_vals[counter], s, w)
        counter += 1
def Stest_generator(
        nsamples_per_drift,
        f_min,
        f_max,
        d_min=0.0,
        d_max=0.0,
        s_min=0.5,
        s_max=100.0,
        w_min=4.0,
        w_max=4.0,
        f_dist='uniform',
        d_dist='uniform',
        s_dist='uniform',
        w_dist='uniform',
        nsignals=1):
    """
    Generates parameters for SNR tests
    """
    f = float(np.mean([f_min, f_max]))
    d = float(np.mean([d_min, d_max]))
    w = float(np.mean([w_min, w_max]))
    #d_vals = np.array([np.full(nsamples_per_drift, i) for i in np.arange(d_min, d_max, 0.5)]).flatten()
    s_vals = np.array([np.full(nsamples_per_drift, i) for i in np.arange(s_min, s_max, 0.5)]).flatten()
    nsamples = len(s_vals)
    counter=0
    while counter < nsamples:
        yield (f, d, s_vals[counter], w) #d_vals[counter], s, w)
        counter += 1



def inject_frame(
        filename,
        parameters, 
        nsignals):
    """
    Returns a frame object with injected signals and 
    a tuple of injected injected signal parameters
    
    Parameters
    ----------
    filename : str
        Path to filterbank file
    parameters : tuple
        Tuple of injection parameter values
    nsignals : int, optional
        Number of signals to inject into data
    
    Notes
    -----
    For each signal the SNR is reduced
    """
    #while niterations > 0:
    frame = stg.Frame(filename)
    frame.add_metadata({'signals':[]})
        
    frequency, drift_rate, snr, width = parameters
    #print(parameters)
    #print(np.shape(parameters))
    #if len(list(np.shape(parameters))) > 1:
    if nsignals > 1:
        for i in range(nsignals):
        #for i in range(np.shape(parameters)[-1]):
        #if len(list(np.shape(parameters)))==1:
        #else:
            f = float(frequency[i])
            d = float(drift_rate[i])
            s = float(snr[i])
            w = float(width[i])

        #logger.debug("f = ",f)
        #logger.debug("d = ",d)
        #logger.debug("s = ",s)
        #logger.debug("w = ",w)
            #print(len(list(np.shape(parameters))))
        #if i > 0 and len(list(np.shape(parameters)))>1:
            s *= .5
            parameters[2][1] = s
            fexs = f - np.copysign(1, d)*w
            fex = (f + (d * frame.dt * frame.tchans)) + np.copysign(1, d)*w
        
        #logger.debug("fexs = ",fexs)
        #logger.debug("fex = ",fex)

            frame.add_signal(
            stg.constant_path(
                f_start=f,
                drift_rate=d*u.Hz/u.s),
            stg.constant_t_profile(level=frame.get_intensity(snr=s)),
            stg.gaussian_f_profile(width=w*u.Hz),
            stg.constant_bp_profile(level=1),
            bounding_f_range=(min(fexs, fex), max(fexs, fex))
            )
            frame.metadata['signals'].append([f, d, s, w])
    else:
        f = float(frequency)
        d = float(drift_rate)
        s = float(snr)
        w = float(width)
        fexs = f - np.copysign(1, d)*w
        fex = (f + (d * frame.dt * frame.tchans)) + np.copysign(1, d)*w
        frame.add_signal(
            stg.constant_path(
                f_start=f,
                drift_rate=d*u.Hz/u.s),
            stg.constant_t_profile(level=frame.get_intensity(snr=s)),
            stg.gaussian_f_profile(width=w*u.Hz),
            stg.constant_bp_profile(level=1),
            bounding_f_range=(min(fexs, fex), max(fexs, fex))
            )
        frame.metadata['signals'].append([f, d, s, w])
    return frame, parameters


def apply_find_doppler(frame, parameters, nsignals):
    """
    Returns results of FindDopplering a frame object
    
    Parameters
    ----------
    frame : stg.Frame
        Frame with injected signals
    parameters : tuple
        Parameters of injected signals
    nsignals : int
        Number of injected signals per frame
    
    """
    #DATA
    #0. num_recovered
    #1. num_inserted
    #2. injected frequency
    #3. injected drift rate
    #4. injected snr
    #5. injected width
    #6. [find_doppler data]
    #7. [2nd sig freq, 2nd sig drift, 2nd sig, 2nd sig snr, 2nd sig, width]

    #RESULTS
    #ratio captured = (0) / (1)
    #injected frequency = (2)
    #detected (captured) frequency = (6)(0)(0)(1)
    control = read_dat('testing_spliced_blc00010203040506o7o0111213141516o021222324252627_guppi_58806_43185_TIC458478250_0127.gpuspec.0000.dat')
    def compare(df_1, df_2, col_):
        """
        Compares DataFrame column values
        """
        def vcompare(df1, df2, col):
            return df1[col] != df2[col]
        #comp = np.vectorize(vcompare)
        #return comp(df_1, df_2, col_)
        return vcompare(df_1, df_2, col_)
    frame.save_hdf5('frame.h5')
    FindDoppler('frame.h5', max_drift=11.0, snr=10).search()
    hits = read_dat('frame.dat')
    os.remove('frame.h5')
    os.remove('frame.dat')
    os.remove('frame.log')

    # remove hits from known RFI
    RFI = read_dat('_fchans_4096_testing_spliced_blc00010203040506o7o0111213141516o021222324252627_guppi_58806_43864_TIC154089169_0129.gpuspec.0000.dat')
    print(hits.columns)
    print(RFI.columns)
    #try: 
    #    hits = hits.loc[(hits['DriftRate'].eq(RFI['DriftRate'])) & (hits['SNR'].eq(RFI['SNR']))]
    #except:
    #    pass
    #hits = hits.loc[compare(hits, control, 'Freq') & compare(hits, control, 'DriftRate') & compare(hits, control, 'SNR')]
    results = {}
    #try:
    results['nInjected'] = [len(frame.metadata['signals'])]
    results['nDetected'] = [len(hits)]
    results['RatioCaptured']=[len(hits) / len(frame.metadata['signals'])]#[len(hits) / np.shape(parameters)[-1]]
    #except:
    #    results['RatioCaptured']=[len(hits) / nsignals]
    if len(hits)==0:
        #hits = {}
        hits['Freq']=[0.0]
        hits['DriftRate']=[0.0]
        hits['SNR']=[0.0]

    #if len(list(np.shape(parameters))) > 1:
    if nsignals > 1:
        results['injFreq'] = [parameters[0][0]/1.0e6]
    else:
        results['injFreq'] = [parameters[0] / 1.0e6]
    #THREW AN ERROR WHEN MORE THAN ONE SIGNAL WAS DETECTED
    #results['detFreq'] = [float(hits['Freq'])]
    results['detFreq'] = [float(hits['Freq'][0])]
    #if len(list(np.shape(parameters))) > 1:
    if nsignals > 1:
        results['injDrift'] = [parameters[1][0]]
    else:
        results['injDrift'] = [parameters[1]]
    results['detDrift'] = [float(hits['DriftRate'][0])]#.tolist()
    if nsignals > 1:#len(list(np.shape(parameters))) > 1:
        results['injSNR'] = [parameters[2][0]]
    else:
        results['injSNR'] = [parameters[2]]
    results['detSNR'] = [float(hits['SNR'][0])]
    if nsignals > 1:#len(list(np.shape(parameters))) > 1:
        results['injWidth'] = [parameters[3][0]]
    else:
        results['injWidth'] = [parameters[3]]
    #results['Separation'] = [(float(np.diff(parameters[0]))*u.MHz).to_value('kHz')]
    if nsignals > 1:#len(list(np.shape(parameters))) > 1:
        results['Separation'] = [float(np.diff(parameters[0]))]#/1.0e3]
        results['secFreq'] = [parameters[0][1]/1.0e6]
        results['secDrift'] = [parameters[1][1]]
        results['secSNR'] = [parameters[2][1]]
        results['secWidth'] = [parameters[3][1]]
        if len(hits)>1:
            results['detsecFreq'] = [float(hits['Freq'][1])]
            results['detsecDrift'] = [float(hits['DriftRate'][1])]
            results['detsecSNR'] = [float(hits['SNR'][1])]

    results['diffFreq'] = float(results['detFreq'][0]) - float(results['injFreq'][0])#.to_numpy()
    results['absdiffFreq'] = abs(results['diffFreq'])
    results['diffDrift'] = results['detDrift'][0]-results['injDrift'][0]
    results['absdiffDrift'] = abs(results['diffDrift'])
    results['diffSNR'] = results['detSNR'][0] - results['injSNR'][0]
    results['absdiffSNR'] = abs(results['diffSNR'])
    results_df = pd.DataFrame.from_dict(results)
    #results_df['injFreq'] = [float(f[1:-1]) for f in results_df['injFreq']]
    #results_df['injDrift'] = [float(d[1:-1]) for d in result_df['injDrift']]
    #results_df['injSNR'] = [float(s[1:-1]) for s in df['injSNR']]
    #results_df['injWidth'] = [float(w[1:-1]) for w in df['injWidth']]
    
    #results_df = pd.concat([results_df, hits], axis=1)
    #for col in ['Hit_ID','status','in_n_ons','RFI_in_range']:
    #    del results_df[col]
    return results_df


def efficiency_pipeline(
        filename, 
        niterations, 
        nsignals=2, 
        d_min=0.0, 
        d_max=0.0, 
        s_min=0.0, 
        s_max=100.0, 
        w_min=4.0, 
        w_max=4.0, 
        f_dist='uniform',
        d_dist='uniform',
        s_dist='uniform',
        w_dist='uniform',
        loglevel=logging.INFO,
        dev=False):
    """
    Top-level efficiency test routine

    Parameters
    ----------
    filename : str
        Path to filterbank file
    nsignals : int, optional
        Number of signals to inject per frame
    niterations : int, optional
        Number of iterations to perform efficiency test

    """
    logger.setLevel(loglevel)
    tchans = stg.Frame(Waterfall(filename, max_load=20)).tchans
    dt = stg.Frame(Waterfall(filename, max_load=20)).dt
    fchans = stg.Frame(Waterfall(filename, max_load=20)).fchans
    f_min, f_max = stg.Frame(Waterfall(filename, max_load=20)).fmin, stg.Frame(Waterfall(filename, max_load=20)).fmax
    if dev:
        frames = (inject_frame(filename, p, nsignals) for p in Stest_generator(
            niterations, 
            f_min,
            f_max,
            d_min=d_min,
            d_max=d_max,
            s_min=s_min,
            s_max=s_max,
            w_min=w_min,
            w_max=w_max,
            nsignals=1))
        #frames = (inject_frame(filename, p) for p in all_dist(
        #    niterations,
        #    f_min,
        #    f_max,
        #    d_min=d_min,
        #    d_max=d_max,
        #    s_min=s_min,
        #    s_max=s_max,
        #    w_min=w_min,
        #    w_max=w_max,
        #    nsignals=nsignals, 
        #    dt=dt,
        #    fchans=fchans,
        #    tchans=tchans))
    else:
        frames = (inject_frame(filename, p, nsignals) for p in parameters_generator(
        niterations, 
        f_min, 
        f_max, 
        d_min=d_min,
        d_max=d_max,
        s_min=s_min, 
        s_max=s_max, 
        w_min=w_min, 
        w_max=w_max, 
        f_dist=f_dist,
        d_dist=d_dist,
        s_dist=s_dist,
        w_dist=w_dist,
        nsignals=nsignals,
        dt=dt,
        fchans=fchans,
        tchans=tchans))
    results = (apply_find_doppler(fr[0], fr[-1], nsignals) for fr in frames)
    results_df = pd.concat(results, ignore_index=True)
    return results_df


def analyze_results(df, twodim=False, dev=True):
    """
    Analyzes results
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame of results

    Notes
    -----
    Quantifies percent recovery as a function of 
    1) signal separation (for 2 signals)
    2) drift rate (histogram)
    """
    # percrec as a function of drift rate
    if not twodim:
        if dev:

            n, bins, patches = plt.hist(df['injDrift'], histtype='step', density=True, bins=np.arange(-10,10,1))#df.plot.hist('injDriftRate')
            #x=df['injDrift'], 
            #bins=np.arange(-10.0, 10.0, 1),
            #histtype='step',
            #facecolor='b', 
            #alpha=0.7, 
            #rwidth=0.85,
            #density=True)

        else:
            n, bins, patches = plt.hist(
            x=df['injDrift'], 
            bins=np.arange(-10.0, 10.0, 1),
            histtype='step',
            facecolor='b', 
            alpha=0.7, 
            rwidth=0.85)
            plt.ylabel('Counts')
    else:
        plt.hist2d(
                x=df['injDrift'],
                y=df['RatioCaptured'],
                bins=np.arange(-10.0,10.0,1),
                #histtype='step',
                #facecolor='b',
                alpha=0.7#,
                #rwidth=0.85
                )
        plt.ylabel("Percent Captured")
    #plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Drift Rate [Hz/s]')
    plt.title('Recovered vs Drift Rate')
    
    #if not twodim:
        #maxcount = n.max()
        # set a clean upper y-axis limit
        #plt.ylim(ymax=np.ceil(maxcount/3)*3 if maxcount %3 else maxcount + 3)
    plt.savefig('Counts_vs_DriftRate.png',bbox_inches='tight')
    print('Saved plot: /home/raffytraas14/seti_efficiency/Counts_vs_DriftRate.png')
    

def test():
    """
    Function to run various ad hoc tests
    """
    filename = get_test_files('x')[0]

    nsamples = 10
    nsignals = 2

    f_min, f_max = stg.Frame(Waterfall(filename, max_load=20)).fmin, stg.Frame(Waterfall(filename, max_load=20)).fmax
    d_min, d_max = -10.0, 10.0
    s_min, s_max = 30.0, 30.0
    w_min, w_max = 4.0, 4.0

    parameters_generator_list = list(parameters_generator(
        nsamples, 
        f_min, 
        f_max,
        d_min=d_min,
        d_max=d_max,
        s_min=s_min,
        s_max=s_max,
        w_min=w_min,
        w_max=w_max, 
        nsignals=nsignals))
    all_dist_list = list(all_dist(
        nsamples, 
        f_min, 
        f_max,
        d_min=d_min,
        d_max=d_max,
        s_min=s_min,
        s_max=s_max,
        w_min=w_min,
        w_max=w_max, 
        nsignals=nsignals))
        
    #print("parameters_generator() == all_dist_list: ", parameters_generator_list == all_dist_list)
    print("parameters_generator():\n", parameters_generator_list)
    print("all_dist_list():\n",all_dist_list)
    print("len(parameters_generator()) == len(all_dist_list()): ", len(parameters_generator_list) == len(all_dist_list))
    print("len(parameters_generator()): ", len(parameters_generator_list))
    print("len(all_dist()): ", len(all_dist_list))

if __name__=='__main__':
    #blueprint()
    #for i in parameters_generator(5, 1.1e10, 7.8e9):
    #    print(i)
    #waterfall_fn = [f for f in get_data_files('x') if 'TIC' in f][2]
    #generate_test_file(waterfall_fn, 'x', power_of_2=8, tchans=None, out_dir='test_files/')
    
    #results = efficiency_pipeline(get_test_files('x')[0], 100, dev=True)
    
    
    
    #results = efficiency_pipeline(get_test_files('x')[0], 1000000, nsignals=1)#, dev=True)#, loglevel=logging.DEBUG)
    results = efficiency_pipeline(get_test_files('x')[0], 10000, nsignals=1)#, dev=True)#, loglevel=logging.DEBUG)
    print(results)
    results.to_csv("10K_snr_no_rfi_filter_signal_efficiency_test.csv")
    #results.to_csv("1M_signal_efficiency_test.csv")
    
    #analyze_results(results)
    
    
    #results.to_csv("100_signals_per_drift_rate_signal_efficiency_test.csv")
    #multigen_analysis.linregress(results,'injDrift', 'absdiffDrift') 
    ####print(results.dropna())
    ####test()
