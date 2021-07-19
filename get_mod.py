import sys, os
import shutil
import subprocess
import glob

#import datetime
#import arrow
#import requests
#from bs4 import BeautifulSoup
#import wget


#####################################################

arg = sys.argv
if len(arg) < 3:
    print('######################################################')
    print('    Usage: Pass line arguments as: ')
    print('       python3 get_mod.py <RUNdir>')
    print('       <RUNdir> : path to run folder ')
    print('       <ModelDomain> :  icogs2d')
    print('######################################################')
    sys.exit()
# Get args
RUNdir = arg[1]
ModelDomain = arg[2]

########################################################################
# Read text file
# Read text file
#base_url = f'http://ccrm.vims.edu/yinglong/Cui/adcirc/'
path = f'/sciclone/schism10/hyu05/NOAA_NWM/oper_2D/fcst'
fcst=glob.glob(f'{path}/2021*')
fcst.sort()
fdate=fcst[-1][-10:]
Date=fdate[:8]
FCT=fdate[-2:]
print(Date)
print(FCT)

url=f'{path}/{fdate}'

fmaxele='maxelev.gr3.gz'
fpoints='staout_1'

work_dir = os.path.join(RUNdir, Date, FCT)

if not os.path.exists(work_dir):
    os.makedirs(work_dir)

#for fname in fnames:
#    floc = os.path.join(work_dir, fname)
#    wget.download(url + fname, out=floc)

    for fname in [fmaxele, fpoints]:
        #floc = os.path.join(work_dir)
        #print(floc)
        if not os.path.exists(url + fname):
            try:
                shutil.copy(f'{url}/{fname}', work_dir)
                # prepare ini file
                fnew = os.path.join(RUNdir, 'new.ini')
                f = open(fnew, 'w')
                f.write('ModelDomain    ' + ModelDomain + ' \n')
                f.write('Date           ' + Date + ' \n')
                f.write('FCT            ' + FCT + ' \n')
                f.write('RUNdir         ' + os.path.abspath(RUNdir) + ' \n')
                f.close()
            except:
                print('ERROR: fetch latest data  \n {} \n was not successful ...')
                os.system('rm -rf ' + work_dir)
                sys.exit()
    cmd=f'gunzip {work_dir}/maxelev.gr3.gz'
    subprocess.check_call(cmd, shell=True)

