# coding: utf-8


"""
Dynamic map hindcast implementation 
"""
__author__ = 'Saeed Moghimi'
__copyright__ = 'Copyright 2020, UCAR/NOAA'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'moghimis@gmail.com'

# import cheetah
from glob import glob
import os
from pathlib import Path
from string import Template
import sys

######## process command line args
arg = sys.argv
if len(arg) < 3:
    print('######################################################')
    print('Usage: Pass line arguments as: ')
    print('python get_mod.py <RUNdir> ')
    print('<RUNdir> : path to run folder ')
    print('<TMPdir> : path to run folder ')
    print('######################################################')
    sys.exit()
# Get args
RUNdir = arg[1]
TMPdir = arg[2]

# RUNdir='/data01/data01/01-projects/07-Maryland/02-working/02-hurricane/hurr_web_auto/run/'
# TMPdir='/data01/data01/01-projects/07-Maryland/02-working/02-hurricane/hurr_web_auto/estofs_static_html/tmpdir/'

print('  > Start generating index.html ...')

out_dir = os.path.join(RUNdir, 'html_out')
if not os.path.exists(out_dir):
    sys.exit('ERROR: pass correct pass to already generated static html files ...')

if not os.path.exists(out_dir):
    sys.exit('ERROR: pass correct pass to already generated static html files ...')


def read_file2str(TMPdir, fname):
    tmpfilen = os.path.join(TMPdir, fname)
    if not os.path.exists(tmpfilen):
        sys.exit(
            'ERROR: pass correct pass TMPdir as second argument and make sure you have "{}" inside it ...'.format(
                fname
            )
        )
    tmpf = open(tmpfilen, 'r')
    tmpstr = tmpf.read()
    tmpf.close()
    return tmpstr


def write_str2html(RUNdir, html_txt, fhtml):
    f_new_html = os.path.join(RUNdir, 'html_out', fhtml)
    tmpf = open(f_new_html, 'w')
    tmpf.write(html_txt)
    tmpf.close()


########################################
####       MAIN CODE from HERE     #####
########################################
tmpstr_azure = read_file2str(TMPdir=TMPdir, fname='html_block_az.temp')
tmpstr_local = read_file2str(TMPdir=TMPdir, fname='html_block.temp')
startstr = read_file2str(TMPdir=TMPdir, fname='html_start.txt')
endstr = read_file2str(TMPdir=TMPdir, fname='html_end.txt')

tmpstr_local = tmpstr_azure

all_blocks_az = []

htmls = glob(out_dir + '/data/*20*html*')

# to get last FCT up
htmls.sort(reverse=True)

for html in htmls:
    html = Path(html).as_posix()
    # build dict for tmp
    fname = html.split('/')[-1].split('.')[0].split('_')
    d = dict(
        ModelDomain=fname[0],
        Date=fname[1],
        FCT='t' + fname[2] + 'z',
        FileName=html.split('/')[-1],
    )
    block_az = Template(tmpstr_azure).safe_substitute(d)

    all_blocks_az.append(block_az)

# generate index.html for Azure website
all_blocks_az_txt = ' \n '.join(all_blocks_az)
new_html_txt = ' \n '.join([startstr, all_blocks_az_txt, endstr])
write_str2html(RUNdir, new_html_txt, 'index.html')

print('  > Finished generating index_az.html')

#################################################
