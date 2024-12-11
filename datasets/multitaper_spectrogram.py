# Analysis Imports
import math
import numpy as np
from scipy.signal.windows import dpss
from scipy.signal import detrend
# Logistical Imports
import warnings
import timeit
from joblib import Parallel, delayed, cpu_count
# Visualization imports
# noinspection PyUnresolvedReferences
# import colorcet  # this import is necessary to add rainbow colormap to matplotlib
import matplotlib.pyplot as plt
import torch

def multitaper_spectrogram_torch(data, fs, frequency_range=None, time_bandwidth=5, num_tapers=None,
                                 window_params=None, min_nfft=0, detrend_opt='linear', weighting='unity',
                                 device='cpu', **kwargs):
    """
    Compute multitaper spectrogram using PyTorch tensors.

    Args:
        data (torch.Tensor or np.ndarray): Tensor or array of shape (batch_size, signal_length) or (signal_length,).
        fs (float): Sampling frequency in Hz.
        frequency_range (list): [min_freq, max_freq], default is [0, fs/2].
        time_bandwidth (float): Time-half bandwidth product.
        num_tapers (int): Number of DPSS tapers to use.
        window_params (list): [window_size (s), step_size (s)].
        min_nfft (int): Minimum NFFT size.
        detrend_opt (str): Detrend option ('linear', 'constant', 'off').
        weighting (str): Taper weighting option ('unity', 'eigen').
        device (str): Device to use ('cpu' or 'cuda').
        **kwargs: Additional arguments.

    Returns:
        mt_spectrogram (torch.Tensor): Spectrogram tensor.
        stimes (torch.Tensor): Time points corresponding to spectrogram windows.
        sfreqs (torch.Tensor): Frequency bins in the spectrogram.
    """
    # Convert data to torch tensor if not already
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(np.ascontiguousarray(data), dtype=torch.float32)
    data = data.to(device)

    # Ensure data has batch dimension
    if data.dim() == 1:
        data = data.unsqueeze(0)  # Shape: (1, signal_length)
    batch_size, signal_length = data.shape

    # Process inputs
    if frequency_range is None:
        frequency_range = [0, fs / 2]
    if num_tapers is None:
        num_tapers = int(math.floor(2 * time_bandwidth) - 1)
    if window_params is None:
        window_params = [5, 1]
    win_size_samples = int(window_params[0] * fs)
    win_step_samples = int(window_params[1] * fs)
    nfft = max(2 ** int(np.ceil(np.log2(win_size_samples))), win_size_samples)

    # Check data length
    if signal_length < win_size_samples:
        raise ValueError("Data length is less than window size")

    # Compute number of windows
    num_windows = int(np.floor((signal_length - win_size_samples) / win_step_samples) + 1)

    # Use torch.unfold to get data segments
    data_segments = data.unfold(dimension=1, size=win_size_samples, step=win_step_samples)
    # Shape: (batch_size, num_windows, win_size_samples)

    # Compute DPSS tapers using scipy and convert to torch tensor
    dpss_tapers, dpss_eigen = dpss(win_size_samples, time_bandwidth, num_tapers, return_ratios=True)
    dpss_tapers = torch.tensor(np.ascontiguousarray(dpss_tapers), dtype=torch.float32).to(device)  # (num_tapers, win_size_samples)
    dpss_eigen = torch.tensor(np.ascontiguousarray(dpss_eigen), dtype=torch.float32).view(num_tapers, 1).to(device)

    # Precompute weights
    if weighting == 'eigen':
        wt = dpss_eigen / num_tapers
    elif weighting == 'unity':
        wt = torch.ones(num_tapers, 1, device=device) / num_tapers
    else:
        raise ValueError("Invalid weighting option")

    # Detrend data segments if necessary
    if detrend_opt != 'off':
        if detrend_opt == 'constant':
            data_segments = data_segments - data_segments.mean(dim=2, keepdim=True)
        elif detrend_opt == 'linear':
            # Implement linear detrending
            x = torch.arange(win_size_samples, dtype=torch.float32, device=device)
            x_mean = x.mean()
            x_var = ((x - x_mean) ** 2).sum()

            y_mean = data_segments.mean(dim=2, keepdim=True)
            xy_cov = ((x - x_mean) * (data_segments - y_mean)).sum(dim=2, keepdim=True)

            a = xy_cov / x_var  # Slope
            b = y_mean - a * x_mean  # Intercept

            trend = a * x + b  # Shape: (batch_size, num_windows, win_size_samples)
            data_segments = data_segments - trend
        else:
            raise ValueError("Invalid detrend option")
    # If detrend_opt is 'off', do nothing

    # Multiply data segments by tapers
    data_segments = data_segments.unsqueeze(2)  # (batch_size, num_windows, 1, win_size_samples)
    dpss_tapers = dpss_tapers.unsqueeze(0).unsqueeze(0)  # (1, 1, num_tapers, win_size_samples)
    tapered_data = data_segments * dpss_tapers  # Broadcasting
    # Shape: (batch_size, num_windows, num_tapers, win_size_samples)

    # Compute FFT
    fft_data = torch.fft.fft(tapered_data, n=nfft, dim=3)  # (batch_size, num_windows, num_tapers, nfft)

    # Compute power spectrum
    spower = fft_data.real**2 + fft_data.imag**2  # (batch_size, num_windows, num_tapers, nfft)

    # Precompute frequencies
    df = fs / nfft
    sfreqs = torch.arange(0, fs, df, device=device)
    freq_inds = (sfreqs >= frequency_range[0]) & (sfreqs <= frequency_range[1])
    sfreqs = sfreqs[freq_inds]

    # Select frequencies
    spower = spower[..., freq_inds]  # (batch_size, num_windows, num_tapers, num_freqs)

    # Compute weighted sum over tapers
    wt = wt.view(1, 1, num_tapers, 1)  # (1, 1, num_tapers, 1)
    mt_spectrogram = (spower * wt).sum(dim=2)  # (batch_size, num_windows, num_freqs)

    # Compute one-sided PSD
    dc_select = (sfreqs == 0)
    nyquist_select = (sfreqs == fs / 2)
    select = ~(dc_select | nyquist_select)
    mt_spectrogram[..., select] *= 2
    mt_spectrogram = mt_spectrogram / fs

    # Generate time vector
    stimes = (torch.arange(num_windows, device=device) * win_step_samples + win_size_samples // 2) / fs

    return mt_spectrogram, stimes, sfreqs

def multitaper_spectrogram_multiple(signals, fs, **kwargs):
    """
    Compute multitaper spectrograms for multiple signals.

    Args:
        signals (list or np.ndarray): List or 2D numpy array where each row corresponds to a signal.
        fs (float): Sampling frequency in Hz.
        **kwargs: Additional keyword arguments to pass to multitaper_spectrogram.

    Returns:
        results (list): A list of tuples where each tuple contains:
                        (mt_spectrogram, stimes, sfreqs) for each signal.
    """
    results = []

    def process_signal(signal):
        return multitaper_spectrogram(signal, fs, **kwargs)

    if kwargs.get("multiprocess", False):
        n_jobs = kwargs.get("n_jobs", cpu_count() - 1)
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_signal)(signals[i, :]) for i in range(signals.shape[0])
        )
    else:
        for i, signal in enumerate(signals):
            results.append(process_signal(signal))
    
    multi_taper_spec = np.stack([r[0] for r in results], axis=0)
    # multi_taper_spec = torch.tensor(multi_taper_spec)
    stimes = results[0][1]
    sfreqs = results[0][2]
    
    return multi_taper_spec, stimes, sfreqs



# MULTITAPER SPECTROGRAM #
def multitaper_spectrogram(data, fs, frequency_range=None, time_bandwidth=5, num_tapers=None, window_params=None,
                           min_nfft=0, detrend_opt='linear', multiprocess=False, n_jobs=None, weighting='unity',
                           plot_on=True, return_fig=False, clim_scale=True, verbose=True, xyflip=False, ax=None):
    """ Compute multitaper spectrogram of timeseries data
    Usage:
    mt_spectrogram, stimes, sfreqs = multitaper_spectrogram(data, fs, frequency_range=None, time_bandwidth=5,
                                                            num_tapers=None, window_params=None, min_nfft=0,
                                                            detrend_opt='linear', multiprocess=False, cpus=False,
                                                            weighting='unity', plot_on=True, return_fig=False,
                                                            clim_scale=True, verbose=True, xyflip=False):
        Arguments:
                data (1d np.array): time series data -- required
                fs (float): sampling frequency in Hz  -- required
                frequency_range (list): 1x2 list - [<min frequency>, <max frequency>] (default: [0 nyquist])
                time_bandwidth (float): time-half bandwidth product (window duration*half bandwidth of main lobe)
                                        (default: 5 Hz*s)
                num_tapers (int): number of DPSS tapers to use (default: [will be computed
                                  as floor(2*time_bandwidth - 1)])
                window_params (list): 1x2 list - [window size (seconds), step size (seconds)] (default: [5 1])
                detrend_opt (string): detrend data window ('linear' (default), 'constant', 'off')
                                      (Default: 'linear')
                min_nfft (int): minimum allowable NFFT size, adds zero padding for interpolation (closest 2^x)
                                (default: 0)
                multiprocess (bool): Use multiprocessing to compute multitaper spectrogram (default: False)
                n_jobs (int): Number of cpus to use if multiprocess = True (default: False). Note: if default is left
                            as None and multiprocess = True, the number of cpus used for multiprocessing will be
                            all available - 1.
                weighting (str): weighting of tapers ('unity' (default), 'eigen', 'adapt');
                plot_on (bool): plot results (default: True)
                return_fig (bool): return plotted spectrogram (default: False)
                clim_scale (bool): automatically scale the colormap on the plotted spectrogram (default: True)
                verbose (bool): display spectrogram properties (default: True)
                xyflip (bool): transpose the mt_spectrogram output (default: False)
                ax (axes): a matplotlib axes to plot the spectrogram on (default: None)
        Returns:
                mt_spectrogram (TxF np array): spectral power matrix
                stimes (1xT np array): timepoints (s) in mt_spectrogram
                sfreqs (1xF np array)L frequency values (Hz) in mt_spectrogram

        Example:
        In this example we create some chirp data and run the multitaper spectrogram on it.
            import numpy as np  # import numpy
            from scipy.signal import chirp  # import chirp generation function
            # Set spectrogram params
            fs = 200  # Sampling Frequency
            frequency_range = [0, 25]  # Limit frequencies from 0 to 25 Hz
            time_bandwidth = 3  # Set time-half bandwidth
            num_tapers = 5  # Set number of tapers (optimal is time_bandwidth*2 - 1)
            window_params = [4, 1]  # Window size is 4s with step size of 1s
            min_nfft = 0  # No minimum nfft
            detrend_opt = 'constant'  # detrend each window by subtracting the average
            multiprocess = True  # use multiprocessing
            cpus = 3  # use 3 cores in multiprocessing
            weighting = 'unity'  # weight each taper at 1
            plot_on = True  # plot spectrogram
            return_fig = False  # do not return plotted spectrogram
            clim_scale = False # don't auto-scale the colormap
            verbose = True  # print extra info
            xyflip = False  # do not transpose spect output matrix

            # Generate sample chirp data
            t = np.arange(1/fs, 600, 1/fs)  # Create 10 min time array from 1/fs to 600 stepping by 1/fs
            f_start = 1  # Set chirp freq range min (Hz)
            f_end = 20  # Set chirp freq range max (Hz)
            data = chirp(t, f_start, t[-1], f_end, 'logarithmic')
            # Compute the multitaper spectrogram
            spect, stimes, sfreqs = multitaper_spectrogram(data, fs, frequency_range, time_bandwidth, num_tapers,
                                                           window_params, min_nfft, detrend_opt, multiprocess,
                                                           cpus, weighting, plot_on, return_fig, clim_scale,
                                                           verbose, xyflip):

        This code is companion to the paper:
        "Sleep Neurophysiological Dynamics Through the Lens of Multitaper Spectral Analysis"
           Michael J. Prerau, Ritchie E. Brown, Matt T. Bianchi, Jeffrey M. Ellenbogen, Patrick L. Purdon
           December 7, 2016 : 60-92
           DOI: 10.1152/physiol.00062.2015
         which should be cited for academic use of this code.

         A full tutorial on the multitaper spectrogram can be found at: # https://www.sleepEEG.org/multitaper

        Copyright 2021 Michael J. Prerau Laboratory. - https://www.sleepEEG.org
        Authors: Michael J. Prerau, Ph.D., Thomas Possidente, Mingjian He

  __________________________________________________________________________________________________________________
    """

    #  Process user input
    [data, fs, frequency_range, time_bandwidth, num_tapers,
     winsize_samples, winstep_samples, window_start,
     num_windows, nfft, detrend_opt, plot_on, verbose] = process_input(data, fs, frequency_range, time_bandwidth,
                                                                       num_tapers, window_params, min_nfft,
                                                                       detrend_opt, plot_on, verbose)

    # Set up spectrogram parameters
    [window_idxs, stimes, sfreqs, freq_inds] = process_spectrogram_params(fs, nfft, frequency_range, window_start,
                                                                          winsize_samples)
    # Display spectrogram parameters
    if verbose:
        display_spectrogram_props(fs, time_bandwidth, num_tapers, [winsize_samples, winstep_samples], frequency_range,
                                  nfft, detrend_opt)

    # Split data into segments and preallocate
    data_segments = data[window_idxs]

    # COMPUTE THE MULTITAPER SPECTROGRAM
    #     STEP 1: Compute DPSS tapers based on desired spectral properties
    #     STEP 2: Multiply the data segment by the DPSS Tapers
    #     STEP 3: Compute the spectrum for each tapered segment
    #     STEP 4: Take the mean of the tapered spectra

    # Compute DPSS tapers (STEP 1)
    dpss_tapers, dpss_eigen = dpss(winsize_samples, time_bandwidth, num_tapers, return_ratios=True)
    dpss_eigen = np.reshape(dpss_eigen, (num_tapers, 1))

    # pre-compute weights
    if weighting == 'eigen':
        wt = dpss_eigen / num_tapers
    elif weighting == 'unity':
        wt = np.ones(num_tapers) / num_tapers
        wt = np.reshape(wt, (num_tapers, 1))  # reshape as column vector
    else:
        wt = 0

    tic = timeit.default_timer()  # start timer

    # Set up calc_mts_segment() input arguments
    mts_params = (dpss_tapers, nfft, freq_inds, detrend_opt, num_tapers, dpss_eigen, weighting, wt)

    if multiprocess:  # use multiprocessing
        n_jobs = max(cpu_count() - 1, 1) if n_jobs is None else n_jobs
        mt_spectrogram = np.vstack(Parallel(n_jobs=n_jobs)(delayed(calc_mts_segment)(
            data_segments[num_window, :], *mts_params) for num_window in range(num_windows)))

    else:  # if no multiprocessing, compute normally
        mt_spectrogram = np.apply_along_axis(calc_mts_segment, 1, data_segments, *mts_params)

    # Compute one-sided PSD spectrum
    mt_spectrogram = mt_spectrogram.T
    dc_select = np.where(sfreqs == 0)[0]
    nyquist_select = np.where(sfreqs == fs/2)[0]
    select = np.setdiff1d(np.arange(0, len(sfreqs)), np.concatenate((dc_select, nyquist_select)))

    mt_spectrogram = np.vstack([mt_spectrogram[dc_select, :], 2*mt_spectrogram[select, :],
                               mt_spectrogram[nyquist_select, :]]) / fs

    # Flip if requested
    if xyflip:
        mt_spectrogram = mt_spectrogram.T

    # End timer and get elapsed compute time
    toc = timeit.default_timer()
    if verbose:
        print("\n Multitaper compute time: " + "%.2f" % (toc - tic) + " seconds")

    if np.all(mt_spectrogram.flatten() == 0):
        print("\n Data was all zeros, no output")

    # Plot multitaper spectrogram
    if plot_on:
        # convert from power to dB
        spect_data = nanpow2db(mt_spectrogram)

        # Set x and y axes
        dx = stimes[1] - stimes[0]
        dy = sfreqs[1] - sfreqs[0]
        extent = [stimes[0]-dx, stimes[-1]+dx, sfreqs[-1]+dy, sfreqs[0]-dy]

        # Plot spectrogram
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        im = ax.imshow(spect_data, extent=extent, aspect='auto')
        fig.colorbar(im, ax=ax, label='PSD (dB)', shrink=0.8)
        ax.set_xlabel("Time (HH:MM:SS)")
        ax.set_ylabel("Frequency (Hz)")
        im.set_cmap(plt.cm.get_cmap('cet_rainbow4'))
        ax.invert_yaxis()

        # Scale colormap
        if clim_scale:
            clim = np.percentile(spect_data, [5, 98])  # from 5th percentile to 98th
            im.set_clim(clim)  # actually change colorbar scale

        fig.show()
        if return_fig:
            return mt_spectrogram, stimes, sfreqs, (fig, ax)

    return mt_spectrogram, stimes, sfreqs


# Helper Functions #

# Process User Inputs #
def process_input(data, fs, frequency_range=None, time_bandwidth=5, num_tapers=None, window_params=None, min_nfft=0,
                  detrend_opt='linear', plot_on=True, verbose=True):
    """ Helper function to process multitaper_spectrogram() arguments
            Arguments:
                    data (1d np.array): time series data-- required
                    fs (float): sampling frequency in Hz  -- required
                    frequency_range (list): 1x2 list - [<min frequency>, <max frequency>] (default: [0 nyquist])
                    time_bandwidth (float): time-half bandwidth product (window duration*half bandwidth of main lobe)
                                            (default: 5 Hz*s)
                    num_tapers (int): number of DPSS tapers to use (default: None [will be computed
                                      as floor(2*time_bandwidth - 1)])
                    window_params (list): 1x2 list - [window size (seconds), step size (seconds)] (default: [5 1])
                    min_nfft (int): minimum allowable NFFT size, adds zero padding for interpolation (closest 2^x)
                                    (default: 0)
                    detrend_opt (string): detrend data window ('linear' (default), 'constant', 'off')
                                          (Default: 'linear')
                    plot_on (True): plot results (default: True)
                    verbose (True): display spectrogram properties (default: true)
            Returns:
                    data (1d np.array): same as input
                    fs (float): same as input
                    frequency_range (list): same as input or calculated from fs if not given
                    time_bandwidth (float): same as input or default if not given
                    num_tapers (int): same as input or calculated from time_bandwidth if not given
                    winsize_samples (int): number of samples in single time window
                    winstep_samples (int): number of samples in a single window step
                    window_start (1xm np.array): array of timestamps representing the beginning time for each window
                    num_windows (int): number of windows in the data
                    nfft (int): length of signal to calculate fft on
                    detrend_opt ('string'): same as input or default if not given
                    plot_on (bool): same as input
                    verbose (bool): same as input
    """

    # Make sure data is 1 dimensional np array
    if len(data.shape) != 1:
        if (len(data.shape) == 2) & (data.shape[1] == 1):  # if it's 2d, but can be transferred to 1d, do so
            data = np.ravel(data[:, 0])
        elif (len(data.shape) == 2) & (data.shape[0] == 1):  # if it's 2d, but can be transferred to 1d, do so
            data = np.ravel(data.T[:, 0])
        else:
            raise TypeError("Input data is the incorrect dimensions. Should be a 1d array with shape (n,) where n is \
                            the number of data points. Instead data shape was " + str(data.shape))

    # Set frequency range if not provided
    if frequency_range is None:
        frequency_range = [0, fs / 2]

    # Set detrending method
    detrend_opt = detrend_opt.lower()
    if detrend_opt != 'linear':
        if detrend_opt in ['const', 'constant']:
            detrend_opt = 'constant'
        elif detrend_opt in ['none', 'false', 'off']:
            detrend_opt = 'off'
        else:
            raise ValueError("'" + str(detrend_opt) + "' is not a valid argument for detrend_opt. The choices " +
                             "are: 'constant', 'linear', or 'off'.")
    # Check if frequency range is valid
    if frequency_range[1] > fs / 2:
        frequency_range[1] = fs / 2
        warnings.warn('Upper frequency range greater than Nyquist, setting range to [' +
                      str(frequency_range[0]) + ', ' + str(frequency_range[1]) + ']')

    # Set number of tapers if none provided
    if num_tapers is None:
        num_tapers = math.floor(2 * time_bandwidth) - 1

    # Warn if number of tapers is suboptimal
    if num_tapers != math.floor(2 * time_bandwidth) - 1:
        warnings.warn('Number of tapers is optimal at floor(2*TW) - 1. consider using ' +
                      str(math.floor(2 * time_bandwidth) - 1))

    # If no window params provided, set to defaults
    if window_params is None:
        window_params = [5, 1]

    # Check if window size is valid, fix if not
    if window_params[0] * fs % 1 != 0:
        winsize_samples = round(window_params[0] * fs)
        warnings.warn('Window size is not divisible by sampling frequency. Adjusting window size to ' +
                      str(winsize_samples / fs) + ' seconds')
    else:
        winsize_samples = window_params[0] * fs

    # Check if window step is valid, fix if not
    if window_params[1] * fs % 1 != 0:
        winstep_samples = round(window_params[1] * fs)
        warnings.warn('Window step size is not divisible by sampling frequency. Adjusting window step size to ' +
                      str(winstep_samples / fs) + ' seconds')
    else:
        winstep_samples = window_params[1] * fs

    # Get total data length
    len_data = len(data)

    # Check if length of data is smaller than window (bad)
    if len_data < winsize_samples:
        raise ValueError("\nData length (" + str(len_data) + ") is shorter than window size (" +
                         str(winsize_samples) + "). Either increase data length or decrease window size.")

    # Find window start indices and num of windows
    window_start = np.arange(0, len_data - winsize_samples + 1, winstep_samples)
    num_windows = len(window_start)

    # Get num points in FFT
    if min_nfft == 0:  # avoid divide by zero error in np.log2(0)
        nfft = max(2 ** math.ceil(np.log2(abs(winsize_samples))), winsize_samples)
    else:
        nfft = max(max(2 ** math.ceil(np.log2(abs(winsize_samples))), winsize_samples),
                   2 ** math.ceil(np.log2(abs(min_nfft))))

    return ([data, fs, frequency_range, time_bandwidth, num_tapers,
             int(winsize_samples), int(winstep_samples), window_start, num_windows, nfft,
             detrend_opt, plot_on, verbose])


# PROCESS THE SPECTROGRAM PARAMETERS #
def process_spectrogram_params(fs, nfft, frequency_range, window_start, datawin_size):
    """ Helper function to create frequency vector and window indices
        Arguments:
             fs (float): sampling frequency in Hz  -- required
             nfft (int): length of signal to calculate fft on -- required
             frequency_range (list): 1x2 list - [<min frequency>, <max frequency>] -- required
             window_start (1xm np array): array of timestamps representing the beginning time for each
                                          window -- required
             datawin_size (float): seconds in one window -- required
        Returns:
            window_idxs (nxm np array): indices of timestamps for each window
                                        (nxm where n=number of windows and m=datawin_size)
            stimes (1xt np array): array of times for the center of the spectral bins
            sfreqs (1xf np array): array of frequency bins for the spectrogram
            freq_inds (1d np array): boolean array of which frequencies are being analyzed in
                                      an array of frequencies from 0 to fs with steps of fs/nfft
    """

    # create frequency vector
    df = fs / nfft
    sfreqs = np.arange(0, fs, df)

    # Get frequencies for given frequency range
    freq_inds = (sfreqs >= frequency_range[0]) & (sfreqs <= frequency_range[1])
    sfreqs = sfreqs[freq_inds]

    # Compute times in the middle of each spectrum
    window_middle_samples = window_start + round(datawin_size / 2)
    stimes = window_middle_samples / fs

    # Get indexes for each window
    window_idxs = np.atleast_2d(window_start).T + np.arange(0, datawin_size, 1)
    window_idxs = window_idxs.astype(int)

    return [window_idxs, stimes, sfreqs, freq_inds]


# DISPLAY SPECTROGRAM PROPERTIES
def display_spectrogram_props(fs, time_bandwidth, num_tapers, data_window_params, frequency_range, nfft, detrend_opt):
    """ Prints spectrogram properties
        Arguments:
            fs (float): sampling frequency in Hz  -- required
            time_bandwidth (float): time-half bandwidth product (window duration*1/2*frequency_resolution) -- required
            num_tapers (int): number of DPSS tapers to use -- required
            data_window_params (list): 1x2 list - [window length(s), window step size(s)] -- required
            frequency_range (list): 1x2 list - [<min frequency>, <max frequency>] -- required
            nfft(float): number of fast fourier transform samples -- required
            detrend_opt (str): detrend data window ('linear' (default), 'constant', 'off') -- required
        Returns:
            This function does not return anything
    """

    data_window_params = np.asarray(data_window_params) / fs

    # Print spectrogram properties
    print("Multitaper Spectrogram Properties: ")
    print('     Spectral Resolution: ' + str(2 * time_bandwidth / data_window_params[0]) + 'Hz')
    print('     Window Length: ' + str(data_window_params[0]) + 's')
    print('     Window Step: ' + str(data_window_params[1]) + 's')
    print('     Time Half-Bandwidth Product: ' + str(time_bandwidth))
    print('     Number of Tapers: ' + str(num_tapers))
    print('     Frequency Range: ' + str(frequency_range[0]) + "-" + str(frequency_range[1]) + 'Hz')
    print('     NFFT: ' + str(nfft))
    print('     Detrend: ' + detrend_opt + '\n')


# NANPOW2DB
def nanpow2db(y):
    """ Power to dB conversion, setting bad values to nans
        Arguments:
            y (float or array-like): power
        Returns:
            ydB (float or np array): inputs converted to dB with 0s and negatives resulting in nans
    """

    if isinstance(y, int) or isinstance(y, float):
        if y == 0:
            return np.nan
        else:
            ydB = 10 * np.log10(y)
    else:
        if isinstance(y, list):  # if list, turn into array
            y = np.asarray(y)
        y = y.astype(float)  # make sure it's a float array so we can put nans in it
        y[y == 0] = np.nan
        ydB = 10 * np.log10(y)

    return ydB


# Helper #
def is_outlier(data):
    smad = 1.4826 * np.median(abs(data - np.median(data)))  # scaled median absolute deviation
    outlier_mask = abs(data-np.median(data)) > 3*smad  # outliers are more than 3 smads away from median
    outlier_mask = (outlier_mask | np.isnan(data) | np.isinf(data))
    return outlier_mask


# CALCULATE MULTITAPER SPECTRUM ON SINGLE SEGMENT
def calc_mts_segment(data_segment, dpss_tapers, nfft, freq_inds, detrend_opt, num_tapers, dpss_eigen, weighting, wt):
    """ Helper function to calculate the multitaper spectrum of a single segment of data
        Arguments:
            data_segment (1d np.array): One window worth of time-series data -- required
            dpss_tapers (2d np.array): Parameters for the DPSS tapers to be used.
                                       Dimensions are (num_tapers, winsize_samples) -- required
            nfft (int): length of signal to calculate fft on -- required
            freq_inds (1d np array): boolean array of which frequencies are being analyzed in
                                      an array of frequencies from 0 to fs with steps of fs/nfft
            detrend_opt (str): detrend data window ('linear' (default), 'constant', 'off')
            num_tapers (int): number of tapers being used
            dpss_eigen (np array):
            weighting (str):
            wt (int or np array):
        Returns:
            mt_spectrum (1d np.array): spectral power for single window
    """

    # If segment has all zeros, return vector of zeros
    if all(data_segment == 0):
        ret = np.empty(sum(freq_inds))
        ret.fill(0)
        return ret

    if any(np.isnan(data_segment)):
        ret = np.empty(sum(freq_inds))
        ret.fill(np.nan)
        return ret

    # Option to detrend data to remove low frequency DC component
    if detrend_opt != 'off':
        data_segment = detrend(data_segment, type=detrend_opt)

    # Multiply data by dpss tapers (STEP 2)
    tapered_data = np.multiply(np.mat(data_segment).T, np.mat(dpss_tapers.T))

    # Compute the FFT (STEP 3)
    fft_data = np.fft.fft(tapered_data, nfft, axis=0)

    # Compute the weighted mean spectral power across tapers (STEP 4)
    spower = np.power(np.imag(fft_data), 2) + np.power(np.real(fft_data), 2)
    if weighting == 'adapt':
        # adaptive weights - for colored noise spectrum (Percival & Walden p368-370)
        tpower = np.dot(np.transpose(data_segment), (data_segment/len(data_segment)))
        spower_iter = np.mean(spower[:, 0:2], 1)
        spower_iter = spower_iter[:, np.newaxis]
        a = (1 - dpss_eigen) * tpower
        for i in range(3):  # 3 iterations only
            # Calc the MSE weights
            b = np.dot(spower_iter, np.ones((1, num_tapers))) / ((np.dot(spower_iter, np.transpose(dpss_eigen))) +
                                                                 (np.ones((nfft, 1)) * np.transpose(a)))
            # Calc new spectral estimate
            wk = (b**2) * np.dot(np.ones((nfft, 1)), np.transpose(dpss_eigen))
            spower_iter = np.sum((np.transpose(wk) * np.transpose(spower)), 0) / np.sum(wk, 1)
            spower_iter = spower_iter[:, np.newaxis]

        mt_spectrum = np.squeeze(spower_iter)

    else:
        # eigenvalue or uniform weights
        mt_spectrum = np.dot(spower, wt)
        mt_spectrum = np.reshape(mt_spectrum, nfft)  # reshape to 1D

    return mt_spectrum[freq_inds]