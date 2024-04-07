import matplotlib.pyplot as plt
import os
import shutil
from glob import glob
import pandas as pd
import obspy
from obspy import UTCDateTime
from scipy.signal import hilbert
import numpy as np    

"""
Copyright (c) 2023 Ben
"""
def data_stack(data, order):
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

            ## calculate phase
            asig = hilbert(acorr)
            phase += asig / np.abs(asig)
        stack /= len(data)
        weight = np.abs(phase / len(data))
        return stack * weight**order


def get_event_list(starttime , endtime, chunklength):
    """
    this function calculates the event list between times d1 and d2 by increment of inc_hours
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
    ----------------------
    return: void
    '''
    # minus/plus 1 day because some some waveforms have starttime = 23:59:59/00:00:00
    # to make sure to have ability to deal with a massive data set
    starttime = obspy.UTCDateTime(starttime_str) - 86400
    endtime   = obspy.UTCDateTime(endtime_str) + 86400
    nday = int((endtime-starttime)/chunklength)
    currenttime_str = ['']*nday
    currenttime = starttime
    
    makefolder(pathout+station+'-group')
    for j in range(nday):
        currenttime_str[j] = currenttime.strftime('%y%m%d')
        makefolder(pathout+station+'-group'+'/'+station+currenttime_str[j])
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

def check_wavecontent_inc_hours(stream, inc_hours):
    '''
    This function return True if the stream of one
    day have more than inc_hours data.
    ----------------------------------
    stream: stream
    inc_hours: incremental hours for checking quantity of data
    '''
    # npts: number of points
    npts = 0
    for tr in stream:
        npts = npts + tr.stats.npts
    sps  = int(stream[0].stats.sampling_rate)
    # if the number of points is less than the data that can be saved
    # in the incremental hours
    # remove the trace
    if npts < sps*inc_hours*3600 * 90.0/100:
        print("bad chunked data: " + stream[0].stats.station + "-"
        + str(stream[0].stats.starttime.strftime("%y%m%d")))
        return False
    return True

def makefolder(fname="New Folder"):
    '''
    Remove the folder to create the new one
    '''
    if not os.path.isdir(fname):
        os.mkdir(fname)
    else:
        shutil.rmtree(fname)
        os.mkdir(fname)

def chunkwaveform_sh(finput='input-directory',
                     foutput='output-directory',
                     chunklength=86400,
                     force_overwrite=False):
    """
    This function is used to create the mscut.sh to chunk the data
    Requirement: gipp + Java JRE
    https://www.gfz-potsdam.de/en/section/geophysical-imaging/infrastructure/geophysical-instrument-pool-potsdam-gipp/software/gipptools
    -------------------------
    finput: the directory of folder that include all the continous mseed file
    foutput: the directory that you want to stores chukced file
    chunktime: duration of chunktime
    force_overwrite: already existing files in the output directory will be overwritten without mercy!
    force_concat: concat the discontinuous waveform
    -------------------------
    return: the mscut.sh file to execute the mseedcut (gipp)
    """
    waveformname = [os.path.basename(x) for x in glob(finput+'/*')]
    waveformname.sort()
    f = open("mscut.sh",'w')

    # create folders for daily data
    makefolder(foutput)
    fw = ''
    if force_overwrite:
        fw = ' --force-overwrite'
    f.write("echo INFO:\n")
    f.write("echo -----------------------------------\n")

    for x in waveformname:
        makefolder(foutput+'/'+x)
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
    makefolder(foutputwf)
    makefolder(foutputsta)
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
