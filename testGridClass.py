#!/usr/bin/python

# Loads a gdbc file into the object of class rgis.grid
# Call testGridClass.py with the following options:
#   -i ./USA_WaterTemp_MonthlyAverages2012.gdbc.gz -V -T


import time
import rgis as rg
import argparse
from argparse import RawTextHelpFormatter
import random


# The following is a small routine that can help with debugging
def print_if(value):
    if print_if_flag:
        print(value)

# For debug purposes, set to True
print_if_flag=True

# We first read the command line arguments and check that we have all the info needed
# to run the routine

parser = argparse.ArgumentParser(description='Tests reading of GDBC files into python',
                                 formatter_class=RawTextHelpFormatter)

parser.add_argument('-i', '--input',
                    dest='Grid',
                    help="gdbc file",
                    default='./USA_WaterTemp_MonthlyAverages2012.gdbc.gz')

parser.add_argument('-o', '--output',
                    dest='OutFile',
                    help="Out gdbc file",
                    default="./TestOutput.gdbc.gz")

parser.add_argument('-V', '--verbose',
                    dest='print_if_flag',
                    action='store_true',
                    help=    "Prints some debugging info",
                    default=False)

parser.add_argument('-T', '--timeseries',
                    action='store_true',
                    help=    "Define if Grid is timeseries",
                    default=False)

args = parser.parse_args()

# For debug purposes, set the Verbose flag (e.g True)
print_if_flag=args.print_if_flag


print_if(args)

# Save the start time to monitor how much time it takes...
start_time = time.time()

IsTimeSeries=args.timeseries

# This is where we catually test the grid class
# Initialize the class with:
#       the name of the GDBC file
#       the optional timeseries flag (if odmitted assumed False, dataset is not a time series)
rgGrid = rg.grid(args.Grid,args.timeseries)
# Now we can load the data from gdbc file
rgGrid.Load()

print_if("--- %s minutes for READ---" % ((time.time() - start_time) / 60))

print_if('Number of rows {}, cols {}, layers {} and bytes {}'.format(rgGrid.nRows,rgGrid.nCols,rgGrid.nLayers,rgGrid.nByte))

print_if('LLx {}, LLy {}, URx {} URy {}'.format(rgGrid.LLx,rgGrid.LLy,rgGrid.URx,rgGrid.URy))

print_if('Grid data shape {}'.format(rgGrid.Data.shape))


# Change the option below to add a layer to the dataset (can save to the GDBC file)
AddLayer=2

if AddLayer > 0:
    # We Add layers to the dataset before saving it
    CurLayers=rgGrid.nLayers
    print_if('Adding {} layers'.format(AddLayer))
    NewNames=[]
    for i in range(0,AddLayer):
        NewNames.append(str(i+1))
    if rgGrid.TimeSeries:
        Step='MS'
    else:
        Step=None
    rgGrid.AddLayer(NewNames,AddLayer) #,Step)
    for i in range(CurLayers, rgGrid.nLayers):
        rgGrid.Data[i, :, :] = random.randint(0, 10)

    print_if('Saving file {}'.format(args.OutFile))
    rgGrid.SaveAs(args.OutFile,rgGrid.Name,'Out title',True,'Month', rgGrid.Year , 1)
else:
    # We just save the input GDBC file to the output file
    print_if('Saving file {}'.format(args.OutFile))
    rgGrid.SaveAs(args.OutFile, rgGrid.Name, 'Out title', True, 'Month', rgGrid.Year, 1)

print_if("-- %s minutes --" % ((time.time() - start_time) / 60))
