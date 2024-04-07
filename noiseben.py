import os
import glob
import scipy
import pandas as pd
import numpy as np
from numba import jit
from scipy.signal import hilbert
from obspy.signal.util import _npts2nfft
from obspy.signal.invsim import cosine_taper
from scipy.fftpack import fft,ifft,next_fast_len
from obspy.signal.filter import bandpass,lowpass
from obspy.signal.regression import linear_regression
from obspy.core.util.base import _get_function_from_entry_point
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from matplotlib import pyplot as plt
from obspy import Trace
import re
import matplotlib.pyplot as plt
import shutil
from glob import glob
import obspy
from obspy import UTCDateTime
from scipy.optimize import curve_fit


'''
--------------------------(c) 2023 ICTP-XCORR-RMA- Modified--------------------------
'''


def cross_correlation_bigdata(spectra1,spectra2,para):
    '''
    this function correlates 2 clean traces

    PARAMETERS:
    spectra1: numpy object #1, containing whitened data
    spectra1: numpy object #1, containing whitened data
    para: dict containing period for band-pass filter
    to strengthen earthquake signal then remove to have pure noise,
    whitening windown length and frequency range of interest

    RETURNS:
    xcorr_tr: cross-correlation in range of maxlag
    '''
    freqmin     = para['freqmin']
    freqmax     = para['freqmax']
    samp_freq   = para['samp_freq']
    maxlag      = para['maxlag']
    delta       = 1/samp_freq
    conj_spectra2=np.conj(spectra2)
    xcorr=np.multiply(spectra1,conj_spectra2)
    xcorr_time=np.fft.fftshift(np.fft.irfft(xcorr,n=2*len(xcorr)))
    xcorr_tr=Trace(header={'npts': len(xcorr_time), 'delta': delta}, data=xcorr_time)
    xcorr_tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=2, zerophase=True)
    xcorr_tr.taper(max_percentage=0.01,type='cosine')
    t1=(xcorr_tr.stats.npts*xcorr_tr.stats.delta/2) - maxlag
    t2=(xcorr_tr.stats.npts*xcorr_tr.stats.delta/2) + maxlag
    xcorr_tr.trim(starttime=xcorr_tr.stats.starttime+t1,
                  endtime=xcorr_tr.stats.starttime+t2)
    return xcorr_tr

def new_spectral_whitening_bigdata(tr, para):
    '''
    this function is used to whiten data

    PARAMETERS:
    trace: obspy trace object, containing normalized data
    para: dict containing period for band-pass filter
    to strengthen earthquake signal then remove to have pure noise
    and whitening windown length

    RETURNS:
    whitened_spectrum: obspy trace object whitened data
    '''
    smooth_N    = para['smooth_N']
    zero_pad    = tr.stats.npts*tr.stats.delta/2
    tr.trim(starttime=tr.stats.starttime-zero_pad,
            endtime=tr.stats.endtime+zero_pad,
            pad=True, nearest_sample=True,
            fill_value=0)
    tr.taper(max_percentage=0.001,type='cosine')
    npts = len(tr.data)
    carr = np.fft.fftshift(np.fft.fft(tr.data, 2*npts-1))
    fft_result = carr[npts-1:2*npts]
    amplitude_spectrum = np.abs(fft_result)
    smoothed_spectrum = np.convolve(amplitude_spectrum,
                                    np.ones(smooth_N)/smooth_N,
                                    mode='same')
    whitened_spectrum = fft_result/smoothed_spectrum
    return whitened_spectrum

def time_domain_normalization_bigdata(tr, para): # running-absolute-mean normalization
    '''
    this function is modified
    this function is used to normalize data in time domain by
    running-absolute-mean normalization

    PARAMETERS:
    tr:  obspy trace object, containing clean data
    para: dict containing frequency for band-pass filter
    to strengthen earthquake signal then remove to have pure noise

    RETURNS:
    tr_norm: obspy trace object normalized data
    '''
    smooth_N    = para['smooth_N']
    tr_norm     = tr.copy()

    window_length=2*smooth_N+1
    weights=[]

    # finding weight to remove signal from earthquake
    for n in range(len(tr_norm.data)):
        start_index = max(0, n - smooth_N)
        end_index   = min(len(tr_norm.data) - 1, n + smooth_N)
        w=np.sum(np.abs(tr_norm.data)[start_index : end_index + 1])/window_length
        weights.append(w)
    tr_norm.data/=np.asarray(weights)
    return tr_norm


'''
--------------------------(c) 2023 ICTP-XCORR-RMA--------------------------
'''

def time_domain_normalization(tr, para): # running-absolute-mean normalization
    '''
    this function is modified
    this function is used to normalize data in time domain by
    running-absolute-mean normalization

    PARAMETERS:
    tr:  obspy trace object, containing clean data
    para: dict containing frequency for band-pass filter
    to strengthen earthquake signal then remove to have pure noise

    RETURNS:
    tr_norm: obspy trace object normalized data
    '''
    smooth_N    = para['smooth_N']
    tr_norm     = tr.copy()
    tr1         = tr.copy() # do not contaminate the tr after filtering

    freqmin_rma = para['freqmin_rma']
    freqmax_rma = para['freqmax_rma']
    tr1.filter('bandpass',freqmin=freqmin_rma, freqmax=freqmax_rma, corners=2, zerophase=True)
    
    window_length=2*smooth_N+1
    weights=[]

    # finding weight to remove signal from earthquake
    for n in range(len(tr1.data)):
        start_index = max(0, n - smooth_N)
        end_index   = min(len(tr1.data) - 1, n + smooth_N)
        w=np.sum(np.abs(tr1.data)[start_index : end_index + 1])/window_length
        weights.append(w)
    tr1 = []
    tr_norm.data/=np.asarray(weights)
    return tr_norm


def new_spectral_whitening(trace,para):
    '''
    this function is used to whiten data

    PARAMETERS:
    trace: obspy trace object, containing clean data
    para: dict containing period for band-pass filter
    to strengthen earthquake signal then remove to have pure noise
    and whitening windown length

    RETURNS:
    whitened_spectrum: obspy trace object whitened data
    '''
    smooth_N    = para['smooth_N']
    tr = time_domain_normalization(trace, para)
    zero_pad = tr.stats.npts*tr.stats.delta/2
    tr.trim(starttime=tr.stats.starttime-zero_pad,endtime=tr.stats.endtime+zero_pad,pad=True, nearest_sample=True, fill_value=0)
    tr.taper(max_percentage=0.001,type='cosine')
    npts = len(tr.data)
    carr = np.fft.fftshift(np.fft.fft(tr.data, 2*npts-1))
    fft_result=carr[npts-1:2*npts]
    amplitude_spectrum = np.abs(fft_result)
    smoothed_spectrum = np.convolve(amplitude_spectrum, np.ones(smooth_N)/smooth_N, mode='same')
    whitened_spectrum = fft_result/smoothed_spectrum
    return whitened_spectrum

def cross_correlation(tr1,tr2,para):
    '''
    this function correlates 2 clean traces

    PARAMETERS:
    tr1: obspy trace object #1, containing clean data
    tr2: obspy trace object #2, containing clean data
    para: dict containing period for band-pass filter
    to strengthen earthquake signal then remove to have pure noise,
    whitening windown length and frequency range of interest

    RETURNS:
    whitened_spectrum: obspy trace object whitened data
    '''
    freqmin  = para['freqmin']
    freqmax  = para['freqmax']
    delta=tr1.stats.delta
    spectra1=new_spectral_whitening(tr1,para)
    spectra2=new_spectral_whitening(tr2,para)
    conj_spectra2=np.conj(spectra2)
    xcorr=np.multiply(spectra1,conj_spectra2)
    xcorr_time=np.fft.fftshift(np.fft.irfft(xcorr,n=2*len(xcorr)))
    xcorr_tr=Trace(header={'npts': len(xcorr_time), 'delta': delta}, data=xcorr_time)
    xcorr_tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=2, zerophase=True)
    xcorr_tr.taper(max_percentage=0.01,type='cosine')
    return xcorr_tr


def xcorr(tr1, tr2, para):
    '''
    this function checking data and making x-correlation

    PARAMETERS:
    tr1: obspy trace object #1, containing clean data
    tr2: obspy trace object #2, containing clean data
    para: dict containing period for band-pass filter
    to strengthen earthquake signal then remove to have pure noise,
    whitening windown length, frequency range of interest
    maxlag for x-correlation
    chunklength and portrion_data for controlling process

    RETURNS:
    whitened_spectrum: obspy trace object whitened data
    '''

    maxlag          = para['maxlag']
    chunklength     = para['chunklength']
    portion_data    = para['portion_data']

    # this part is not important because the input is clean data
    startime1 = tr1.stats.starttime.timestamp
    startime2 = tr2.stats.starttime.timestamp
    t11 = max(startime1,startime2)
    t11_utc = UTCDateTime(t11)
    endtime1 = tr1.stats.endtime.timestamp
    endtime2 = tr2.stats.endtime.timestamp
    t22 = min(endtime1,endtime2)
    t22_utc = UTCDateTime(t22)
    record_length = t22 - t11
    x_corr = Trace()
    
    if record_length > portion_data*chunklength:
        tr1.trim(starttime=t11_utc,endtime=t22_utc)
        tr2.trim(starttime=t11_utc,endtime=t22_utc)
        x_corr=cross_correlation(tr1, tr2, para)
        t1=(x_corr.stats.npts*x_corr.stats.delta/2) - maxlag
        t2=(x_corr.stats.npts*x_corr.stats.delta/2) + maxlag
        x_corr.trim(starttime=x_corr.stats.starttime+t1, endtime=x_corr.stats.starttime+t2)
        return x_corr
    else:
        return [] # return empty trace


'''
--------------------------(c) 2023 ICTP-XCORR-1BIT--------------------------
'''

# def normalize(tr, norm_method="1bit"):

#     if norm_method == "1bit":
#         tr.data = np.sign(tr.data)
#         tr.data = np.float32(tr.data)
#     else:
#         raise ValueError('choose 1 bit! abort!')
#     return tr

# def spectral_whitening(tr,para):
#     lc  = para['lc']
#     hc = para['hc']
#     smooth_N    = para['smooth_N']

#     npts = len(tr.data)
#     Nyfreq=0.5/tr.stats.delta
#     tr_norm=normalize(tr, norm_method="1bit")
#     # zeropad(1/2)*number of points
#     window = np.hamming(len(tr_norm.data))
#     tapered_data=tr_norm.data*window
#     zero_pad=tr_norm.stats.npts*tr_norm.stats.delta/2
#     tr_norm.trim(starttime=tr_norm.stats.starttime-zero_pad,endtime=tr_norm.stats.endtime+zero_pad,pad=True,nearest_sample=True, fill_value=0)
#     #tr_norm=time_domain_normalization(tr,smoothing_window,lc,hc)
#     carr=np.fft.fftshift(np.fft.fft(tapered_data, 2*npts-1))
#     fft_result=carr[npts-1:2*npts]
#     amplitude_spectrum=np.abs(fft_result)
#     #smoothing_window=50
#     smoothed_spectrum=np.convolve(amplitude_spectrum,np.ones(smooth_N)/smooth_N, mode='same')
#     whitened_spectrum=fft_result/smoothed_spectrum
#     return whitened_spectrum

# def correlate(tr1,tr2,para):
#     lc  = para['lc']
#     hc = para['hc']
#     smooth_N    = para['smooth_N']
#     tr1_spectrum=spectral_whitening(tr1,para)
#     tr2_spectrum=spectral_whitening(tr2,para)
#     xcorr=np.multiply(np.conj(tr1_spectrum),tr2_spectrum )   # correlate tr1 and tr2
#     delta=tr1.stats.delta                              ##tr is original data
#     xcorr_time=np.fft.fftshift(np.fft.irfft(xcorr,n=2*len(xcorr)))
#     xcorr_tr=Trace(header={'npts': len(xcorr_time), 'delta': delta},data=xcorr_time)
#     return xcorr_tr

# def xcorr(tr1, tr2, para):
#     maxlag     = para['maxlag']
#     chunklength = para['chunklength']
#     # this part is not important because the input is clean data
#     startime1 = tr1.stats.starttime.timestamp
#     startime2 = tr2.stats.starttime.timestamp
#     t11 = max(startime1,startime2)
#     t11_utc = UTCDateTime(t11)
#     endtime1 = tr1.stats.endtime.timestamp
#     endtime2 = tr2.stats.endtime.timestamp
#     t22 = min(endtime1,endtime2)
#     t22_utc = UTCDateTime(t22)
#     record_length = t22 - t11
#     x_corr = Trace()
#     if record_length > 0.9*chunklength:
#         tr1.trim(starttime=t11_utc,endtime=t22_utc)
#         tr2.trim(starttime=t11_utc,endtime=t22_utc)
#         x_corr=correlate(tr1, tr2, para)
#         t1=(x_corr.stats.npts*x_corr.stats.delta/2) - maxlag
#         t2=(x_corr.stats.npts*x_corr.stats.delta/2) + maxlag
#         x_corr.trim(starttime=x_corr.stats.starttime+t1, endtime=x_corr.stats.starttime+t2)
#         return x_corr
#     else:
#         return []


'''
--------------------------(c) 2023-Ben-ICTP-Preprocess-NoisePy--------------------------
'''

def sta_info_from_inv(inv):
    '''
    this function outputs station info from the obspy inventory object
    (used in S0B)
    PARAMETERS:
    ----------------------
    inv: obspy inventory object
    RETURNS:
    ----------------------
    sta: station name
    net: netowrk name
    lon: longitude of the station
    lat: latitude of the station
    elv: elevation of the station
    location: location code of the station
    '''
    # load from station inventory
    sta = inv[0][0].code
    net = inv[0].code
    lon = inv[0][0].longitude
    lat = inv[0][0].latitude
    if inv[0][0].elevation:
        elv = inv[0][0].elevation
    else: elv = 0.

    if inv[0][0][0].location_code:
        location = inv[0][0][0].location_code
    else: location = '00'

    return sta,net,lon,lat,elv,location

def preprocess_raw(st,inv,para,date_info):
    '''
    this function is modified from NOISEPY
    this function pre-processes the raw data stream by:
        1) check samping rate and gaps in the data;
        2) remove sigularity, trend and mean of each trace
        3) filter and correct the time if integer time are between sampling points
        4) remove instrument responses with selected methods including:
            "inv"   -> using inventory information to remove_response;
            "spectrum"   -> use the inverse of response spectrum. (a script is provided in additional_module to estimate response spectrum from RESP files)
            "RESP_files" -> use the raw download RESP files
            "polezeros"  -> use pole/zero info for a crude correction of response
        5) trim data to a day-long sequence and interpolate it to ensure starting at 00:00:00.000

    PARAMETERS:
    -----------------------
    st:  obspy stream object, containing noise data to be processed
    inv: obspy inventory object, containing stations info
    para: dict containing fft parameters, such as frequency bands and selection for instrument response removal etc.
    date_info:   dict of start and end time of the stream data

    RETURNS:
    -----------------------
    ntr: obspy stream object of cleaned, merged and filtered noise data
    '''
    # load paramters from fft dict
    rm_resp       = para['rm_resp']
    if 'rm_resp_out' in para.keys():
        rm_resp_out   = para['rm_resp_out']
    else:
        rm_resp_out   = 'VEL'
    # respdir       = para['respdir']
    freqmin       = para['freqmin']
    freqmax       = para['freqmax']
    samp_freq     = para['samp_freq']
    portion_data  = para['portion_data']
    max_files     = para['max_files']

    # parameters for butterworth filter
    f1 = 0.9*freqmin;f2=freqmin
    if 1.1*freqmax > 0.45*samp_freq:
        f3 = 0.4*samp_freq
        f4 = 0.45*samp_freq
    else:
        f3 = freqmax
        f4= 1.1*freqmax
    pre_filt  = [f1,f2,f3,f4]
    # prior_filt=[0.0067,0.01,8,10]
    # check sampling rate and trace length
    st = check_sample_data(st,date_info, portion_data, max_files)
    if len(st) == 0:
        print('RETURN: No traces in Stream: Continue!', end=' | ');return st
    sps = int(st[0].stats.sampling_rate)
    station = st[0].stats.station

    # remove nan/inf, mean and trend of each trace before merging
    for ii in range(len(st)):

        #-----set nan/inf values to zeros (it does happens!)-----
        tttindx = np.where(np.isnan(st[ii].data))
        if len(tttindx) >0:st[ii].data[tttindx]=0
        tttindx = np.where(np.isinf(st[ii].data))
        if len(tttindx) >0:st[ii].data[tttindx]=0

        st[ii].data = np.float32(st[ii].data)
        st[ii].data = scipy.signal.detrend(st[ii].data,type='constant')
        st[ii].data = scipy.signal.detrend(st[ii].data,type='linear')

    # merge, taper and filter the data
    if len(st)>1:st.merge(method=1,fill_value=0)
    st[0].taper(max_percentage=0.05,max_length=50)	# taper window
    st[0].data = np.float32(bandpass(st[0].data,pre_filt[0],pre_filt[-1],df=sps,corners=4,zerophase=True))

    # make downsampling if needed
    if abs(samp_freq-sps) > 1E-4:
        # downsampling here
        st.interpolate(samp_freq,method='weighted_average_slopes')
        delta = st[0].stats.delta

        # when starttimes are between sampling points
        fric = st[0].stats.starttime.microsecond%(delta*1E6)
        if fric>1E-4:
            st[0].data = segment_interpolate(np.float32(st[0].data),float(fric/(delta*1E6)))
            #--reset the time to remove the discrepancy---
            st[0].stats.starttime-=(fric*1E-6)

    # remove traces of too small length

    # options to remove instrument response
    if rm_resp == 'inv':
        #----check whether inventory is attached----
        if not inv[0][0][0].response:
            raise ValueError('no response found in the inventory! abort!')
        else:
            try:
                print('removing response for %s using inv'%st[0])
                st[0].attach_response(inv)
                st[0].remove_response(output=rm_resp_out,pre_filt=pre_filt,water_level=60)
            except Exception:
                print('cannot remove response')
                st = []
                return st
    else:
        raise ValueError('no such option for rm_resp! please double check!')
    ntr = obspy.Stream()
    # trim a continous segment into user-defined sequences
    st[0].trim(starttime=date_info['starttime'],endtime=date_info['endtime'],pad=True,fill_value=0)
    ntr.append(st[0])

    return ntr

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

@jit('float32[:](float32[:],float32)')
def segment_interpolate(sig1,nfric):
    '''
    this function interpolates the data to ensure all points located on interger times of the
    sampling rate (e.g., starttime = 00:00:00.015, delta = 0.05.)
    PARAMETERS:
    ----------------------
    sig1:  seismic recordings in a 1D array
    nfric: the amount of time difference between the point and the adjacent assumed samples

    RETURNS:
    ----------------------
    sig2:  interpolated seismic recordings on the sampling points
    '''
    npts = len(sig1)
    sig2 = np.zeros(npts,dtype=np.float32)

    #----instead of shifting, do a interpolation------
    for ii in range(npts):

        #----deal with edges-----
        if ii==0 or ii==npts-1:
            sig2[ii]=sig1[ii]
        else:
            #------interpolate using a hat function------
            sig2[ii]=(1-nfric)*sig1[ii+1]+nfric*sig1[ii]

    return sig2


def check_sample_data(stream,date_info,portion_data, max_files):
    """
    this function checks sampling rate and find gaps of all traces in stream.
    PARAMETERS:
    -----------------
    stream: obspy stream object.
    date_info: dict of starting and ending time of the stream

    RETURENS:
    -----------------
    stream: List of good traces in the stream
    """
    # remove empty/big traces
    if len(stream)==0 or len(stream) > max_files:
        print("maxfiles {:d} > {:d}".format(len(stream), max_files), end=' | ')
        stream = []
        return stream

    # remove traces with big gaps
    # modified by ben
    # if portion_gaps(stream,date_info)>0.3:
    pdata = portion_dataset(stream,date_info)
    if pdata < portion_data:
        print("proportion data {:4.2f} < 0.95".format(pdata), end=' | ')
        stream = []
        return stream

    freqs = []
    for tr in stream:
        freqs.append(int(tr.stats.sampling_rate))
    freq = max(freqs)
    for tr in stream:
        if int(tr.stats.sampling_rate) != freq:
            stream.remove(tr)
        if tr.stats.npts < 10:
            stream.remove(tr)

    return stream


def portion_dataset(stream,date_info):
    '''
    this function tracks the data (npts) from the accumulated
    difference between starttime and endtime of each stream trace
    PARAMETERS:
    -------------------
    stream: obspy stream object
    date_info: dict of starting and ending time of the stream

    RETURNS:
    -----------------
    pdata: proportion of data/all_pts in stream
    '''
    # ideal duration of data
    starttime = date_info['starttime']
    endtime   = date_info['endtime']

    npts = 0
    for tr in stream:
        npts = npts + tr.stats.npts
    pdata = npts/((endtime-starttime)*stream[0].stats.sampling_rate)
    return pdata


'''
--------------------------(c) 2023-Ben-ICTP-Utilities--------------------------
'''
#

# Define the spline function
def spline_function(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def spline(x, y, x_smooth):
    # Fit the spline regression
    popt, _ = curve_fit(spline_function, x, y)

    # Generate points for the fitted spline curve
    y_smooth = spline_function(x_smooth, *popt)
    return y_smooth


def atoi(text):
    return int(text) if text.isdigit() else text

def natural(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def data_stack(data, order):
    '''
    this function is used to stack data
    PARAMETERS:
    -----------------------
    data: the list of array, each array contains the data of trace
    order:
    RETURNS:
    -----------------------
    stack: 1 array of final stack
    '''
    stack = 0
    phase = 0j
    if order == 0: # linear
        for acorr in data:
            stack += acorr
        stack /= len(data)
        return stack
    else:
        for acorr in data:
            stack += acorr

            asig = hilbert(acorr)
            phase += asig / np.abs(asig)
        stack /= len(data)
        weight = np.abs(phase / len(data))
        stack = stack * weight**order
        return stack


def get_timelist(starttime , endtime, chunklength):
    # old name: get_event_list
    """
    this function calculates the event list between times d1 and d2 by increment of chunklength
    PARAMETERS:
    ----------------
    str1: string of the starting time -> 2010_01_01_0_0
    str2: string of the ending time -> 2010_10_11_0_0
    chunklength: in second -> 1 day = 86400
    RETURNS:
    ----------------
    event: a numpy character list
    """
    dt = chunklength
    d1 = starttime
    d2 = endtime
    event = []
    while d1 < d2:
        event.append(d1.strftime("%Y_%m_%d_%H_%M_%S"))
        d1 += dt
    event.append(d2.strftime("%Y_%m_%d_%H_%M_%S"))
    return event

def basename(path):
    return [os.path.basename(x) for x in path]

def copy_group_daily(pathin = './waveforms-daily',
                     pathout = './waveforms-group/',
                     station   = 'station-name',
                     starttime_str = '2014-12-01T00:00:00',
                     endtime_str   = '2016-12-31T00:00:00',
                     chunklength = 86400):
    '''
    copy the data mseed-daily data to group-daily data
    because there are discontinuity in the daily data
    that means there are more than one waveform in one day (differnt hours)
    PARAMETERS:
    ----------------------
    station: list of station
    starttime_str: string of the starttime
    endtime_str: string of end time
    chunklength: it should be 86400 - 1 day
    RETURNS:
    ----------------------
    void
    '''
    # minus/plus 1 day because some some waveforms have starttime = 23:59:59/00:00:00
    # to make sure to have ability to deal with a massive data set
    starttime = obspy.UTCDateTime(starttime_str) - 86400
    endtime   = obspy.UTCDateTime(endtime_str) + 86400
    nday = int((endtime-starttime)/chunklength)
    currenttime_str = ['']*nday
    currenttime = starttime

    makefolder(pathout+station+'-group', force_overmake=True)
    for j in range(nday):
        currenttime_str[j] = currenttime.strftime('%y%m%d')
        makefolder(pathout+station+'-group'+'/'+station+currenttime_str[j], force_overmake=True)
        currenttime = currenttime + chunklength
    waveformname = [os.path.basename(x) for x in glob(pathin+station+'.mseed/*')]
    waveformname.sort()
    for x in waveformname:
        for y in currenttime_str:
            z = x[-16:len(x)-10]  # 'akat141203052400.bhz' => 141203
            if y == z:
                shutil.copy2(pathin+station+'.mseed/'+x,
                        pathout+station+'-group/'+station+y)


def mplplot(stream, figsize=[12.0, 8.0], linewidth=1):
    '''
    This function is used to plot waveforms
    '''
    if len(stream)==1:
        fig, ax = plt.subplots(nrows=len(stream), ncols=1, figsize=figsize)
        ax.plot(stream[0].times("matplotlib"), stream[0].data, "k-", linewidth=linewidth, label=stream[0].stats.station+'.'+stream[0].stats.channel)
        ax.legend(loc=2)
    else:
        fig, ax = plt.subplots(nrows=len(stream), ncols=1, figsize=figsize)
        for i, tr in zip(range(0,len(stream)), stream):
            ax[i].plot(tr.times("matplotlib"), tr.data, "k-", linewidth=linewidth, label=tr.stats.station+'.'+tr.stats.channel)
            ax[i].legend(loc=2)

def makefolder(fname="New Folder", force_overmake = False):
    '''
    '''
    # Remove the folder to FORCE-CREATE the new one
    if force_overmake == True:
        if not os.path.isdir(fname):
            os.mkdir(fname)
        else:
            shutil.rmtree(fname)
            os.mkdir(fname)
    if force_overmake == False:
        try:
            os.mkdir(fname)
        except:
            pass

def chunkwaveform_sh(finput='input-directory',
                     foutput='output-directory',
                     chunklength=86400,
                     force_overwrite=False):
    """
    This function is used to create the mscut.sh to chunk the data
    REQUIREMENT:
    -------------------------
    gipp + Java JRE
    check the link:
    https://www.gfz-potsdam.de/en/section/geophysical-imaging/infrastructure/geophysical-instrument-pool-potsdam-gipp/software/gipptools
    PARAMETERS:
    -------------------------
    finput: the directory of folder that include all the continous mseed file
    foutput: the directory that you want to stores chukced file
    chunktime: duration of chunktime
    force_overwrite: already existing files in the output directory will be overwritten without mercy!
    force_concat: concat the discontinuous waveform
    RETURNS:
    -------------------------
    the mscut.sh file to execute the mseedcut (gipp)
    """
    waveformname = [os.path.basename(x) for x in glob(finput+'/*')]
    waveformname.sort()
    f = open("mscut.sh",'w')

    # create folders for daily data
    makefolder(foutput, force_overmake=True)
    fw = ''
    if force_overwrite:
        fw = ' --force-overwrite'
    f.write("echo INFO:\n")
    f.write("echo -----------------------------------\n")

    for x in waveformname:
        makefolder(foutput+'/'+x, force_overmake=True)
        # write the bash file for executing mseedcut
        # echo the name of waveform
        f.write("echo Processing waveform: "+x+"\n")
        # do mseedcut
        f.write("mseedcut --file-length="+str(chunklength)+" --output-dir="+
                foutput+'/'+x+" "+
                finput+'/'+x+fw+"\n")
    f.write("echo Finised!\n")
    f.close()

def gfzdownload_sh(fintput = 'gmap-stations-gfz.txt',
                     foutputwf = 'gfz-waveforms',
                     foutputsta= 'gfz-stations',
                     starttime = '2014-12-01T00:00:00',
                     endtime   = '2014-12-01T00:00:10',
                     channel   = 'BHZ'):
    '''
    This function is used to create the sh file to dowload gfz data by fdsnws_fetch
    https://geofon.gfz-potsdam.de/software/fdsnws_fetch/
    -----------------------------
    Return:
    gfz-waveform.sh for downloading waveform from gfz
    gfz-station.sh for downloading station from gfz

    '''
    df = pd.read_csv(fintput, sep="|")
    df.columns = df.columns.str.replace(' ', '')
    df.columns = df.columns.str.lower()
    f = open('gfz-waveform.sh','w')
    makefolder(foutputwf, force_overmake=True)
    makefolder(foutputsta, force_overmake=True)
    # print the fdsnws_fetch for downloading waveform
    for i in range(df.shape[0]):
        f.write('fdsnws_fetch'
                + ' -N \'' + df.network[i] + '\''
                + ' -S \'' + df.station[i] + '\''
                + ' -L \'*\''
                + ' -C \'' + channel       + '\''
                + ' -s \'' + starttime     + '\''
                + ' -e \'' + endtime       + '\''
                + ' -v -o ' + foutputwf +'/' + df.station[i] + '.mseed'
                + '\n')
    f.close()
    f = open('gfz-station.sh','w')
    # print the fdsnws_fetch for downloading station
    for i in range(df.shape[0]):
        f.write('fdsnws_fetch'
                + ' -N \'' + df.network[i] + '\''
                + ' -S \'' + df.station[i] + '\''
                + ' -L \'*\''
                + ' -C \'' + channel       + '\''
                + ' -s \'' + starttime     + '\''
                + ' -e \'' + endtime       + '\''
                + ' -y station -q level=response'
                + ' -v -o ' + foutputsta +'/' + df.station[i] + '.xml'
                + '\n')
    f.close()
