#!/usr/bin/python

# Loads a gdbc file into the object of class rgis.grid
# Call testGridClass.py with the following options:
#   <gdbc filename> -V -T
# Test GDBCs:
# Daily:
#   /asrc/ecr/NEWS/LoopingPaper/noresm1-m/rcp8p5/cap/v001/RGISresults/noresm1-m_rcp8p5_cap_v001/USA/qxt_watertemp/Pristine/Static/Daily/USA_qxt_watertemp_Pristine_Static_dTS2040.gdbc.gz
# Monthly:
#   /asrc/ecr/NEWS/LoopingPaper/noresm1-m/rcp8p5/cap/v001/RGISresults/noresm1-m_rcp8p5_cap_v001/USA/qxt_watertemp/Pristine/Static/Monthly/USA_qxt_watertemp_Pristine_Static_mTS2012.gdbc.gz



import os,mmap
#from subprocess import Popen, PIPE
import subprocess as sp
import struct
import numpy as np
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt
import rgis as rg
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
import argparse
from argparse import RawTextHelpFormatter
import calendar
import random


# The following is a small routine that can help with debugging
def print_if(value):
    if print_if_flag:
        print(value)

# For debug purposes, set to True
print_if_flag=True

# We first read the command line arguments and check that we have all the info needed
# to run the routine

parser = argparse.ArgumentParser(description='Calculates seasonal HDD and CDD',
                                 formatter_class=RawTextHelpFormatter)

parser.add_argument('Grid',
    help="gdbc file",
    default='')
#    default='./test2000.gdbc')

parser.add_argument('-o', '--output',
                    dest='OutFile',
                    help="Out gdbc file",
                    default="/asrc/ecr/fabio/NEWS/Tests/pippo_test.gdbc.gz")

parser.add_argument('-V', '--verbose',
    dest='print_if_flag',
    action='store_true',
    help=    "Prints some debugging info",
    default=False)

parser.add_argument('-T', '--timeseries',
    action='store_true',
    help=    "Define if Grid is timeseries",
    default=True)

args = parser.parse_args()

# For debug purposes, set the Verbose flag (e.g True)
print_if_flag=args.print_if_flag


print_if(args)

# Save the start time to monitor how much time it takes...
start_time = time.time()

IsTimeSeries=args.timeseries

# Load data from gdbc file
rg.grid(args.Grid)
rgGrid = rg.grid(args.Grid,args.timeseries) #clip2.gdbc' # test2000b.gdbc'
rgGrid.Load()

# Change the option below to add a layer to the dataset (can save to the GDBC file)
AddLayer=0

print_if("--- %s minutes for READ---" % ((time.time() - start_time) / 60))

print_if('Number of rows {}, cols {}, layers {} and bytes {}'.format(rgGrid.nRows,rgGrid.nCols,rgGrid.nLayers,rgGrid.nByte))

print_if('LLx {}, LLy {}, URx {} URy {}'.format(rgGrid.LLx,rgGrid.LLy,rgGrid.URx,rgGrid.URy))

print_if('Grid data shape {}'.format(rgGrid.Data.shape))

#for Layer in rgGrid.Layers['ID']:
#        i = Layer # .ID
#        print('Day {}: Min {}, Max {}, Avg {}, SDev {}'.format(i,np.nanmin(rgGrid.Data[i,:,:]),np.nanmax(rgGrid.Data[i,:,:]),np.nanmean(rgGrid.Data[i,:,:]),np.nanstd(rgGrid.Data[i,:,:])))
"""
#TEST FOR THE NEWS POSTPROCESSING CODE (AFTER AddLayer METHOD REWRITE)
rgGrid.Layers['Name']='1960-' + '{num:02d}'.format(num=1)
NewNames=['1960-' + '{num:02d}'.format(num=i) for i in range(2, 10)]
#for i in range(2, 13):
#    NewNames.append('XXXX-' + '{num:02d}'.format(num=i))
rgGrid.AddLayer(NewNames,8)
"""

if AddLayer > 0:
    CurLayers=rgGrid.nLayers
    print_if('Adding {} layers'.format(AddLayer))
#    rgGrid.Data[0,:,:]=random.randint(0,10)
    NewNames=[]
    for i in range(0,AddLayer):
        NewNames.append(str(i+1))
#        rgGrid.AddLayer(str(i))
#        rgGrid.Data[i,:,:]=random.randint(0,10)
    if rgGrid.TimeSeries:
        Step='MS'
    else:
        Step=None
    rgGrid.AddLayer(NewNames,AddLayer) #,Step)
    for i in range(CurLayers, rgGrid.nLayers):
        rgGrid.Data[i, :, :] = random.randint(0, 10)

    print_if('Saving file {}'.format(args.OutFile))
    rgGrid.SaveAs(args.OutFile,rgGrid.Name,'pluto',True,'Month', rgGrid.Year , 1) # 'Day',0,1,1) # ,True,
#/asrc/ecr/fabio/NEWS/Tests/pippo_new.gdbc

#for i in range(0,rgGrid.nRows):
#    rgGrid.Data[0,i,:]=(rgGrid.URy - (rgGrid.ResY / 2 + rgGrid.ResY * i))
#rgGrid.Layers.loc[rgGrid.Layers.index==0,'Name'] = 'Latitude'
#LonLayer=rgGrid.AddLayer('Longitude')
#for i in range(0,rgGrid.nCols):
#    rgGrid.Data[LonLayer,:,i]=(rgGrid.LLx + rgGrid.ResX / 2 + rgGrid.ResX * i)

"""
Rows=[]
for i in range(0,rgGrid.nRows):
    Rows.append('{:8.3f}'.format(rgGrid.LLy + rgGrid.ResY / 2 + rgGrid.ResY * i))
Rows.sort(reverse=True)
Cols=[]
for i in range(0,rgGrid.nCols):
    Rows.append('{:8.3f}'.format(rgGrid.LLx + rgGrid.ResX / 2 + rgGrid.ResX * i))
df_tmp=pd.DataFrame(data=rgGrid.Data[0,:,:],index=Rows,columns=Cols)
"""



#rgGrid.Recast(np.float32)
#/asrc/ecr/NEWS/Runs/Network/NAmerica_HydroSTNv100pre_03min_Static_v2.gdbn
#/asrc/ecr/fabio/NEWS/Tests/USA_airtemperature_Pristine_Static_dTS2099.gdbc
#/asrc/ecr/NEWS/PowerPlants/creating_files/Version_1/fuel_1.gdbc
#/asrc/NEWSisimip/tas/gfdl-esm2m/rcp8p5/3min/tas_gfdl-esm2m_rcp8p5_2099.gdbc.gz
#/asrc/ecr/NEWS/PowerPlants/PPtemplate.gdbc
#rgGrid.Save("/asrc/ecr/NEWS/PowerPlants/TestIndex.gdbc","/asrc/ecr/NEWS/PowerPlants/creating_files/Version_1/fuel_1.gdbc","TestIndex")
print_if("-- %s minutes --" % ((time.time() - start_time) / 60))
