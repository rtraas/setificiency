import pandas as pd
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from itertools import groupby

# Nature's standard figure dimensions: 89 mm wide (single column) and 183 mm (double column)
u.imperial.enable()
# based off of Nature's "triple column" standards (linearly extrapolated)
figwidth = ((183+(183-89))*u.mm).to_value('inch')

def _bins(low, high, width):
    """
    Returns the centers of the bins
    """
    return np.arange(low, high + width, width)

def _bin_edges(low, high, width):
    """
    Returns the left and right edges of the bins as 
    a generator object
    """
    edges = _bins(low, high, width)
    return (edges[i:i+2] for i in range(len(edges)-1))

def bin_by_col(df, col, width):
    """
    Bins the data based on column values and a width
    
    Returns
    -------
    centers : list
        list of bin centers
    edges : list
        list of left-right bin edge pairs
    data_bins : list
        list of data in each bin
    """
    # get the data into bins
    data_bins = []
    for Ledge, Redge in _bin_edges(df[col].min(), df[col].max(), width):
        data_bins.append(df.loc[(df[col] > Ledge) & (df[col] < Redge)])
    edges = list(_bin_edges(df[col].min(), df[col].max(), width))
    centers = [float(np.mean(e)) for e in edges]
    return centers, edges, data_bins

def analyze_bins(centers:list, edges:list, data_bins:list, bincol:str, cols:list, ops=['mean']):
    """
    Obtains requested column statistics from data bins
    
    Parameters
    ----------
    centers : list
        center value of each bin
    edges : list
        list of left-right bin edge pairs
    data_bins : list
        list of data in each bin
    bincol : str
        the column of the DataFrame to bin the data on
    cols : list of str
        the column(s) to obtain statistics for
    ops : str, optional
        the operation to apply to each bin 
        when obtaining the column value for 
        the entire bin.  (default is to obtain 
        the mean of the bin values.  Any pandas 
        `Series` operation is valid)
    
    Returns
    -------
    analyzed_bins : dict
        dictionary containing the binned statistics for each requested 
        data column
    """
    # ensure number of bins equals actual number of data bins
    if len(edges)!=len(data_bins):
        raise ValueError("`edges` (length: {}) does not match length of `data_bins` (length: {})".format(len(edges, len(data_bins))))
    
    # if number of operations doesn't match number of requested columns, 
    # use same operation to compute statistics for all columns
    if len(ops)!=len(cols):
        # hardcoded to use first operation
        # ideally: repeats last operation
        ops = [ops[0]]*len(cols)
    
    # specify dictionary to hold requested statistics
    analyzed_bins = {col:[] for col in cols}
    
    # add bin centers
    analyzed_bins['centers'] = centers
    
    # add name of column used to bin the data
    analyzed_bins['bincol'] = bincol
    
    # add bin edges
    analyzed_bins['edges'] = edges
    
    # specify a list for number of injected signals per bin
    # -> to be used in estimating average number of signals per bin 
    analyzed_bins['nInjected'] = []
    
    for bin_df in data_bins:
        for i, col in enumerate(cols):
            # get the corresponding operation for the current column
            op = getattr(pd.Series, ops[i])
            
            # apply the operation to the current column
            # assign to var
            statistic = op(bin_df[col])
            analyzed_bins[col].append(statistic)
            
            # append number of injected signals in bin
            analyzed_bins['nInjected'].append(bin_df['nInjected'].sum())
    
    return analyzed_bins
    
#     # obtain statistics for each column
#     analyzed_bins = {col:[]*len(edges) for col in cols}

#     for bin_df in data_bins:
#         for col in cols:
#             # obtain requested statistics 
#             # using getattr() TO CALL A CLASS METHOD BY ITS NAME AS A STRING
#             # source: https://www.kite.com/python/answers/how-to-call-a-function-by-its-name-as-a-string-in-python#:~:text=Use%20getattr()%20to%20call,its%20name%20as%20a%20string&text=Call%20getattr(object%2C%20name),the%20class%20as%20an%20argument.
#             analyzed_bins[col].append(getattr(pd.Series, ops)(bin_df[col]))
#     return analyzed_bins


def plot_bins(xdata, ydata, xlabel:str, ylabel:str, title:str, grid:bool=True, save=None, transparent=False):
    """
    
    Plots input data, formatting as requested
    
    Parameters
    ----------
    ax : matplotlib axis
        axis to plot data onto
    xdata : list, np.array, pd.Series
        x-axis data points
    ydata : list, np.array, pd.Series
        y-axis data points
    xlabel : str
        text to label x-axis
    ylabel : str
        text to label y-axis
    title : str
        text to label title
    grid : bool, optional
        plots on a grid (default is `True`)
    save : str, optional
        name of file to save image to (default is `None` 
        in which plot is not saved)
    transparent : bool, optional
        plotted images are transparent (default is `False`)
    
    Notes
    -----
    Can be used individually or wrapped as in `plot_bins_pipeline`
    
    """
    ax = plt.gca()
    ax.plot(xdata, ydata)
    ax.scatter(xdata, ydata)

    # add a grid (if requested) for user's viewing pleasure
    if grid:
        ax.grid()

    # format the x-axis
    ax.set_xticks(xdata)
    ax.tick_params(axis='x', rotation=70)
    ax.set_xlabel(xlabel)

    # format the y-axis
    ax.set_ylabel(ylabel)

    # format title
    ax.set_title(title)
    
    if save is not None:
        plt.savefig(save, bbox_inches='tight', transparent=transparent)


def plot_bins_pipeline(analyzed_bins, 
              grid:bool=True,
              transparent:bool=False, 
              density:bool=True):
    """
    
    Plots requested data to create a binned distribution of target data
    
    Parameters
    ----------
    analyzed_bins : dict
        dictionary result from `analyze_bins` function
    grid : bool, optional
        plots on a grid (default is `True`)
    transparent : bool, optional
        plotted images are transparent (default is `False`)
    
    """
    # get list of data columns; filter out center values of the bins
    cols = [k for k in analyzed_bins.keys() if k not in ['centers','bincol', 'nInjected', 'edges']]
    
    # get bin width
    width = abs(float(np.diff(analyzed_bins['centers'])[0]))
    
    # estimated number of injected signals per bin
    sigs_per_bin = int(np.array(analyzed_bins['nInjected']).mean())
    
    # define some formatting parameters
    col_to_str = {'injDrift':'Injected Drift [Hz/s]', 'detDrift':'Detected Drift [Hz/s]', 'injSNR':'Injected SNR', 'RatioCaptured':'Ratio Captured', 'nInjected':'Number of injected signals'}
    col_to_title = {'injSNR':'SNR', 'injDrift':'Drift Rate', 'detDrift':'Drift Rate', 'RatioCaptured':'Efficiency', 'nInjected':'Number of injected signals'}
    col_to_csv = {'injDrift':'drift_rate', 'detDrift':'drift_rate', 'injSNR':'snr', 'RatioCaptured':'ratio_captured', 'nInjected':'n_injected'}
    col_to_unit = {'injDrift':'Hz/s', 'detDrift':'Hz/s', 'injSNR':'Arbitrary Units', 'detSNR':'Arbitrary Units', 'injWidth':'Hz', 'detWidth':'Hz'}
    
    fig = plt.figure(figsize=(figwidth, figwidth*len(cols)))
    gs = fig.add_gridspec(nrows=len(cols) + 1, ncols=1, hspace=0.5, wspace=0.0)
    
    # specify the histogram axis
    axis = fig.add_subplot(gs[-1:, :])
    # plot number of injected signals as a function of bin
    plot_bins(
        #ax=axis, 
        xdata=analyzed_bins['centers'], 
        ydata=analyzed_bins['nInjected'], 
        xlabel=col_to_str[analyzed_bins['bincol']], 
        ylabel='Number of injected signals', 
        title="Distribution of injected signals"
    )
    
    for gs_idx, col in enumerate(cols):
        ax = fig.add_subplot(gs[gs_idx:gs_idx+1, :], sharex=axis)
        plot_bins(
            #ax=ax, 
            xdata=analyzed_bins['centers'], 
            ydata=analyzed_bins[col], 
            xlabel=col_to_str[analyzed_bins['bincol']], 
            ylabel=col_to_str[col], 
            title="{} by {}".format(col_to_title[col], col_to_title[analyzed_bins['bincol']])
        )
    
    plt.suptitle("{} Efficiency Test: Bin Width = {:.2f} [{}]".format(col_to_title[analyzed_bins['bincol']], width, col_to_unit[analyzed_bins['bincol']]))
    cols_str = cols[0]
    for c in cols[1:]:
        cols_str+= "_{}_".format(c)
    fig.savefig("{}_eff_test_by_{}.png".format(col_to_csv[analyzed_bins['bincol']], cols_str), transparent=transparent, bbox_inches='tight')
    
        
def binalyze(df:pd.core.frame.DataFrame, bincol:str, width:float, outcols:list, ops:list=['mean']):
    """
    
    Top-level bin analysis function.
    
    Parameters
    ----------
    df : pd.core.frame.DataFrame
        DataFrame of efficiency test results
    bincol : str
        the column of the DataFrame to bin the data on
    width : float
        the bin width to be applied uniformly
    outcols : list of str
        the column(s) to obtain statistics for
    ops : str, optional
        the operation to apply to each bin 
        when obtaining the column value for 
        the entire bin.  (default is to obtain 
        the mean of the bin values.  Any pandas 
        `Series` operation is valid)
    
    Notes
    -----
    `binalyze` is the compressed form of the 
    words 'bin analysis'
    
    """
    centers, edges, bins = bin_by_col(df, bincol, width)
    
    bin_analysis = analyze_bins(centers, edges, bins, cols=outcols, ops=ops, bincol=bincol)
    
    plot_bins_pipeline(bin_analysis)