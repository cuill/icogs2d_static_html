#!/bin/csh

#hp desktop
#export python3=/home/moghimi/opt/all_conda/02_miniconda2/envs/hurricane/bin/python3
#export rundir=/data01/data01/01-projects/07-Maryland/02-working/02-hurricane/hurr_web_auto/run/
#export tmpdir=/data01/data01/01-projects/07-Maryland/02-working/02-hurricane/hurr_web_auto/estofs_static_html/tmpdir
#export ncmeshfile=/data01/data01/01-projects/07-Maryland/02-working/02-hurricane/hurr_web_auto/estofs_static_html/inpdir/hsofs_atl_mesh.nc

conda activate estofs_static_html_wsl

#base=/mnt/c/Users/Saeed.Moghimi/Documents/work/07-estofs-web/estofs_static_html
#base=/sciclone/data10/lcui01/estofs_static_html
setenv base /sciclone/data10/lcui01/icogs2d_static_html
#NOAA laptop linux sub system
setenv python3 python3
#export rundir=/mnt/c/Users/Saeed.Moghimi/Documents/work/07-estofs-web/run/
setenv rundir ${base}/run/
setenv tmpdir ${base}/tmpdir/
#export ncmeshfile=${base}/inpdir/hsofs_atl_mesh.nc
setenv ncmeshfile ${base}/inpdir/hgrid.gr3

cd ${base}
${python3} get_mod.py ${rundir} icogs2d
${python3} gen_sta_html.py ${rundir} icogs2d
${python3} gen_index_html.py ${rundir} ${tmpdir}
#${python3} write_azure.py

#https://ocsofsviewersa.blob.core.windows.net/$web
#https://ocsofsviewersa.blob.core.windows.net/$web/estofs/data/
#https://ocsofsviewersa.blob.core.windows.net/$web/estofs/

#commmit and push
#cd ${rundir}/html_out
#git add data/*.html
#txt=`ls *.html`
#git commit -m "${txt}"
#git push origin master

#* 9 * * *  /sciclone/data10/lcui01/Post_processing/09_static_html/drive_estofs_atl.sh
#check cron >                /etc/init.d/cron status
