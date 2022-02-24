"""
Handles the interface between Python and the rgis_package framework
The goal is to expose the directory structure of the rgis_package
framework to Python to allow easy access to the RGISarchives
and the results of the WBM model runs

Requires:
rgis_package to be installed on the system (https://github.com/bmfekete/RGIS)

Contains:

class grid()
Loads a rgis_package grid file into a numpy multidimensional array and
an XArray data structure

class wbm()
Reads the WBM script and loads the configuration variables,
which define the output variables produced and the directory
structure of the WBM results.

function RGISfunction()
It exposes the functions of the RGISfunctions.sh script.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!! The following section kept for historical reference !!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!! NO LONGER NEEDED  !!!!!!!!!!!!!!!!!!!!!!!!!!
! RGISfunctions.sh now exposes the functions even without sourcing it !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
To access the RGISfunctions.sh script it used to require a a small
wrapper script named RGISpython.sh to be placed in the same
folder as the RGISfunctions.sh
RGISfunctions.sh is generally locatd in the following folder:
/usr/local/share/ghaas/Scripts
The content of the RGISpython.sh script is as follows
------------ Start of Script ------------------
#!/bin/bash
# This is a wrapper script to expose the functions in the RGISfunctions.sh
# script to python
#
# Requires two parameters:
#    the Function being invoked
#    the required arguments
#
# Returns the result evaluated by the RGIS function
#
# exits if there are less than 2 parameters passed
if [ "${GHAASDIR}" == "" ]; then GHAASDIR="/usr/local/share/ghaas"; fi
source "${GHAASDIR}/Scripts/RGISfunctions.sh"
if (( $# < 2)); then
    exit 0
else
    FUNCTION="$1"; shift
    ARGUMENTS="$@"
fi
${FUNCTION} ${ARGUMENTS}
------------- End of Script ------------------
!!!!!!!!!!!!!!!!! END OF THE PART NO LONGER NEEDED !!!!!!!!!!!!!!!!!!!!
"""

# Various import statements, commets added where needed...

# ray is used for parallel processing with shared memory
# this code is stuck with an early version of the library
# plus it's likely an overkill...
#import ray

import time
from ctypes import *
from copy import deepcopy
import sys
import gzip
import struct
# Subprocess is used to spwan the call to the respective RGIS commands
import subprocess as sp
import os
import numpy as np
import pandas as pd
import random
import xarray as xr
import psutil
#import pickle
#import multiprocessing as mp # import Process,Pool,Manager,Queue
#from .pickle2reducer import Pickle2Reducer
#from pathos.multiprocessing import ProcessingPool as Pool
#import dill
#import threading

#ctx = mp.get_context()
#ctx.reducer = Pickle2Reducer()

#dill.settings['protocol'] = 4

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

if 'GHAASDIR' in os.environ:
    Dir2Ghaas = os.environ['GHAASDIR']
else:
    Dir2Ghaas = '/usr/local/share/ghaas'

# The following are two data structures converted from the RGIS C code,
# and are used to parse the GDBC file header

class MFmissing(Union):
    _fields_ = [("Int", c_int),
                ("Float", c_double)]

class MFdsHeader(Structure):
    _fields_ = [("Swap", c_short),
                ("Type", c_short),
                ("ItemNum", c_int),
                ("Missing", MFmissing),
                ("Date", c_char * 24)]

# This is translation of the GDBC data type codes into standard
# Numpy type codes
def _npType(nType):
    # nType values: 5=short,6=long,7=float,8=double
    if nType == 5:
        return np.int16
    elif nType == 6:
        return np.int32
    elif nType == 7:
        return np.float32
    elif nType ==  8:
        return np.float64
    else:
        raise Exception('Unknown value format: type {}'.format(nType))


####################################################
# Shared functions
def _FindFile(Name):
    # Checkes if the River GIS grid file exists and if it is
    # compressed or a data stream file

    NameTest = Name.lower()
    if os.path.isfile(Name):
        if Name[len(NameTest) - 3:len(NameTest)] == '.ds':
            Result = {'Name': Name, 'Compressed': False, 'Datastream': True, 'Network': False}
        elif Name[len(NameTest) - 5:len(NameTest)] == '.gdbn':
            Result = {'Name': Name, 'Compressed': False, 'Datastream': False, 'Network': True}
        else:
            if Name[len(NameTest) - 3:len(NameTest)] == '.gz':
                Result = {'Name': Name, 'Compressed': True, 'Datastream': False, 'Network': False}
            else:
                Result = {'Name': Name, 'Compressed': False, 'Datastream': False, 'Network': False}
    elif os.path.isfile(Name + '.gz'):
        Result = {'Name': Name + '.gz', 'Compressed': True, 'Datastream': False, 'Network': False}
    else:
        raise Exception('File {} not found'.format(Name))
    return Result


####################################################

def _ReadDBLayers(inFile, TimeSeriesFlag):
    # Loads the DBLayers table of the RiverGIS file. This table can be
    # loaded into a pandas DataFrame and the parameters are used to read
    # the rest of the file
    cmd = [Dir2Ghaas + '/bin/rgis2table', '-a', 'DBLayers', inFile]
    proc = sp.Popen(cmd, stdout=sp.PIPE)  # , shell=True) +inFile
    data1 = StringIO(bytearray(proc.stdout.read()).decode("utf-8"))
    if TimeSeriesFlag:
        Layers = pd.read_csv(data1, sep='\t',
                             parse_dates=['Name'],
                             infer_datetime_format=True,
                             index_col=['Name'],
                             dtype={'ID': 'int',
                                    'RowNum': 'int',
                                    'ColNum': 'int',
                                    'ValueType': 'int',
                                    'ValueSize': 'int',
                                    'CellWidth': 'float',
                                    'CellHeight': 'float'})
        Layers.index = pd.DatetimeIndex(Layers.index.values, freq=pd.infer_freq(Layers.index))
        ts = pd.to_datetime(str(Layers.index.values[0]))
        Year = ts.strftime('%Y')
    else:
        Layers = pd.read_csv(data1, sep='\t',
                             dtype={'ID': 'int',
                                    'RowNum': 'int',
                                    'ColNum': 'int',
                                    'ValueType': 'int',
                                    'ValueSize': 'int',
                                    'CellWidth': 'float',
                                    'CellHeight': 'float'})
        Year = None
    Layers['ID'] = Layers['ID'] - 1

    return Layers, Year


####################################################

def _ReadDBItems(inFile):  # ,TimeSeriesFlag):
    # Loads the DBLayers table of the RiverGIS file. This table can be
    # loaded into a pandas DataFrame and the parameters are used to read the rest
    # of the file

    # Dict_types={'ID' : 'int',
    #            'Name' : 'str',
    #            'GridValue' : 'float',
    #            'SymbolFLD' : 'str',
    #            'GridArea' : 'float',
    #            'GridPercent' : 'float'}

    cmd = [Dir2Ghaas + '/bin/rgis2table', '-a', 'DBItems', inFile]
    proc = sp.Popen(cmd, stdout=sp.PIPE)  # , shell=True) +inFile
    data1 = StringIO(bytearray(proc.stdout.read()).decode("utf-8"))

    Items = pd.read_csv(data1, sep='\t')  # ,
    # dtype= Dict_types)
    Items['ID'] = Items['ID'] - 1

    return Items


####################################################

def _ReadRawData(Name, template):
    # Loads a grid data from the DataStream into a array
    # cmd = [Dir2Ghaas + '/bin/rgis2ds']
    if template is None:
        cmd = [Dir2Ghaas + '/bin/rgis2ds']
    else:
        cmd = [Dir2Ghaas + '/bin/rgis2ds', '-m ', template]
    cmd.append(Name)

    proc = sp.Popen(cmd, stdout=sp.PIPE)
    RawData = proc.stdout.read()  # (perLayer*self.nLayers+40*self.nLayers)

    return RawData

###################################################

def _LoadDBCells(inFile):
    # Loads a network DBCells table into a Pandas DataFrame
    cmd = Dir2Ghaas + '/bin/rgis2table -a DBCells '
    proc = sp.Popen(cmd + inFile, stdout=sp.PIPE, shell=True)

    data1 = StringIO(bytearray(proc.stdout.read()).decode("utf-8"))

    return pd.read_csv(data1, sep='\t')

####################################################

def runInParallel(*fns):
    proc = []
    #manager = mp.Manager()
    #return_dict = manager.dict()
    for fn in fns:
        p = mp.Process(target=fn) # ,args=return_dict)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()

#@ray.remote
def _LoadData_m(name,template,verbose=False):
    data, type, nodata, nptype, dummy = _LoadData(name,template,verbose=verbose)
    #ray.put(data, type, nodata, nptype)
    return [data, type, nodata, nptype]
#@ray.remote
def _LoadDBLayers_m(name,timeseries,verbose=False):
    layers, year,dummy = _LoadDBLayers(name,timeseries,verbose=verbose)
    #ray.put(layers, year)
    return [layers, year]
#@ray.remote
def _LoadGeo_m(name,compressed,verbose=False):
    llx,lly,metadata,dummy= _LoadGeo(name,compressed,verbose=verbose)
    #ray.put(llx,lly,metadata)
    return [llx,lly,metadata]
#@ray.remote
def _LoadDBItems_m(name,verbose=False):
    items,dummy=_LoadDBItems(name,verbose=verbose)
    #ray.put(items)
    return [items]

def _LoadDBLayers(name,timeseries,verbose=False): #, q): #,pippo,return_dict):
    if verbose:
        print('Started _LoadDBLayers')
        start_time = time.time()
    #return_dict['Layers'], return_dict['Year'] = _ReadDBLayers(self.Name, self.TimeSeries)
    Layers, Year = _ReadDBLayers(name, timeseries)
    #self.Layers, self.Year = _ReadDBLayers(self.Name, self.TimeSeries)

    #if q is not None:
    #    print("Loading queue in _LoadDBLayers")
    #    q.put([self.Layers, self.Year])

    if verbose:
        print('Finished _LoadDBLayers in {} minutes'.format((time.time() - start_time) / 60))
    return Layers, Year,'Layers'

def _LoadDBItems(name,verbose=False): #, q): #,pippo,return_dict):
    if verbose:
        print('Started _LoadDBItems')
        start_time = time.time()
    #return_dict['Layers'], return_dict['Year'] = _ReadDBLayers(self.Name, self.TimeSeries)
    Items = _ReadDBItems(name)
    #self.Layers, self.Year = _ReadDBLayers(self.Name, self.TimeSeries)
    #if q is not None:
    #    print("Loading queue in _LoadDBItems")
    #    q.put([self.Items])
    if verbose:
        print('Finished _LoadDBItems in {} minutes'.format((time.time() - start_time) / 60))
    return Items,'Items'

def _LoadData(name,template,verbose=False): # q, template=''):

    if verbose:
        print('Started _LoadData')
        start_time = time.time()

    RawData=_ReadRawData(name, template)

    if verbose:
        print("--- %s minutes for actual pipe transfer ---" % ((time.time() - start_time) / 60))

    #q.put = ([self.RawData])

    dump40 = MFdsHeader.from_buffer_copy(RawData[0:40])
    Type = dump40.Type
    # Type values 5:short,6:long,7:float,8:double
    if Type > 6:
        NoData = dump40.Missing.Float
    else:
        NoData = dump40.Missing.Int

    npType = _npType(Type)

    Data1 = np.frombuffer(RawData, dtype=npType)
    Data = np.copy(Data1)
    #self.Data.reshape([-1, 1])

    #print(type(self.Data),self.npType,self.NoData)
    #print(len(self.Data))

    #if q is not None:
    #    print("Loading queue in _LoadData")
    #    ttt=len(self.Data)-1
    #    q.put([self.Data[0:10000], self.Type, self.NoData])
    if verbose:
        print('Finished _LoadData in {} minutes'.format((time.time() - start_time) / 60))
    return Data, Type, NoData, npType, 'Data'

    #return_dict['RawData'] = data


    """
    for i in range(0, self.nLayers):
        dump40 = MFdsHeader()
        proc.stdout.readinto(dump40)
        self.Type = dump40.Type
        dataparts.append(bytearray(proc.stdout.read(perLayer)))
    data = b"".join(dataparts)
    """

def _LoadGeo(name,compressed,verbose=False): #, q):
    if verbose:
        print('Started _LoadGeo')
        start_time = time.time()
    if compressed:
        ifile=gzip.open(name, "rb") # as ifile
    else:
        ifile=open(name, "rb")  # as ifile:
    ifile.seek(40)
    LLx = struct.unpack('d', ifile.read(8))[0]
    LLy = struct.unpack('d', ifile.read(8))[0]
    ifile.read(8)
    titleLen = struct.unpack('h', ifile.read(2))[0]
    title = ifile.read(titleLen).decode()
    MetaData={"title":title}
    ifile.read(9)
    docLen = struct.unpack('h', ifile.read(2))[0]
    docRec = ifile.read(docLen).decode()
    ifile.read(25)
    readMore=True
    while readMore:
        infoLen=struct.unpack('h', ifile.read(2))[0]
        infoRec=ifile.read(infoLen).decode()
        if infoRec=="Data Records":
            readMore=False
            break
        ifile.read(1)
        valLen=struct.unpack('h', ifile.read(2))[0]
        if valLen == 44:
            ifile.read(26)
        elif valLen == 48:
            ifile.read(30)
        valLen=struct.unpack('h', ifile.read(2))[0]
        valRec=ifile.read(valLen).decode()
        MetaData[infoRec.lower()]=valRec
        ifile.read(1)

    #if q is not None:
    #    print("Loading queue in _LoadGeo")
    #    q.put([self.LLx,self.LLy,self.MetaData])
    if verbose:
        print(f'Finished _LoadGeo in {(time.time() - start_time) / 60} minutes')
    return LLx,LLy,MetaData,'Geo'


####################################################
####################################################

class grid():
    """
        Properties defined here:
            Name:     the fully qualifies name of the file to be loaded
            Layers:   the dictionary with the grid settings read from the above
            Data:     the list of variables generated by the WBM run
    """

    def __init__(self, Name=None, TimeSeries=False, verbose=False):
        """
            Initializes the class and sets the GHAASDIR variable
        """
        self.verbose=verbose
        # We check for the existence of both the uncompressed and the compressed version of the
        # file
        self.DataStream = False

        if Name == None:
            self.NewGrid = True
        else:
            self.NewGrid = False
            FindFileResult = _FindFile(Name)
            self.Name = FindFileResult['Name']
            self.Compressed = FindFileResult['Compressed']
            self.DataStream = FindFileResult['Datastream']

        self.TimeSeries = TimeSeries
        if self.NewGrid:
            self.Layers = pd.DataFrame(columns=['Name'])
        else:
            self.Layers = None
        self.Items = None
        self.RawData = None
        self.Data = None
        self.Template = None
        self.TemplNet = None
        self.Cells = None
        self.Year = None
        self.nLayers = None
        self.nRows = None
        self.nCols = None
        self.nByte = None
        self.nType = None
        self.ResX = None
        self.ResY = None
        self.Type = None
        self.npType = None
        self.Projection = None
        self.LLx = None
        self.LLy = None
        self.URx = None
        self.URy = None
        self.NoData = None
        self.MetaData = {'title': None,
                         'geodomain': None,
                         'subject': None,
                         'version': None}
        self.LayerHeaderLen = None


    def _LoadDBCells(self, inFile):
        # Loads a network DBCells table into a Pandas DataFrame
        cmd = Dir2Ghaas + '/bin/rgis2table -a DBCells '
        proc = sp.Popen(cmd + inFile, stdout=sp.PIPE, shell=True)

        data1 = StringIO(bytearray(proc.stdout.read()).decode("utf-8"))

        return pd.read_csv(data1, sep='\t')

    def _LoadParallel(self,which):
        if which == 1:
            return self._LoadData()
        elif which == 2:
            return self._LoadDBLayers()
        elif which == 3:
            return self._LoadGeo()
        else:
            raise Exception('Unknonw option in _LoadParallel: {}'.format(which))

#    def _SortReturn(self, res):
#        for i in range(0,3):
#            ReturnType=res[i][len(res[i])-1]
#            if ReturnType == 'Data':
#                self.RawData=res[i][0]
#                self.Type=res[i][1]
#                self.NoData=res[i][2]
#            elif ReturnType == 'Layers':
#                self.Layers = res[i][0]
#                self.Year = res[i][1]
#            elif ReturnType == 'Geo':
#                self.LLx = res[i][0]
#                self.LLy = res[i][1]
#            else:
#                raise Exception('Unknonw return option in _SortReturn: {}'.format(ReturnType))

    def _SortReturn(self, data,layers,geo,items):
        self.Data=data[0]
        self.Type=data[1]
        self.NoData=data[2]
        self.Layers = layers[0]
        self.Year = layers[1]
        self.LLx = geo[0]
        self.LLy = geo[1]
        self.MetaData = geo[3]
        self.Items = items[0]

    def Load(self, template=None, MultiThread=False):
        if MultiThread:
            print("Error: MultiThread option currently not supposrteded")
            exit(1)
        if not self.DataStream:  # If NOT a DataStream then expect GDBC and load DBLayers table
            #runInParallel(self._LoadData,self._LoadDBLayers)

            #p=Pool(nodes=3)
            if MultiThread:
                # MultiThread not working yet....
                if self.verbose:
                    print("MultiThread option select")
                ncpu = int(psutil.cpu_count() *.75)
                ray.init(num_cpus=ncpu,# include_webui=False,
                         ignore_reinit_error=True,
                         memory=60000 * 1024 * 1024,
                         object_store_memory=20000 * 1024 * 1024,
                         driver_object_store_memory=10000 * 1024 * 1024)

                name_id=ray.put(self.Name)
                timeseries_id=ray.put(self.TimeSeries)
                compressed_id=ray.put(self.Compressed)
                template_id=ray.put(self.Template)
                verbose=ray.put(self.verbose)

                load_out = [_LoadDBLayers_m.remote(name_id,timeseries_id,verbose),
                            _LoadGeo_m.remote(name_id,compressed_id,verbose),
                            _LoadDBItems_m.remote(name_id,verbose),
                            _LoadData_m.remote(name_id,template_id,verbose)]

                loadlayers, loadgeo,loaditems,loaddata = ray.get(load_out)

                self.Layers = loadlayers[0]
                self.Year = loadlayers[1]
                self.LLx = loadgeo[0]
                self.LLy = loadgeo[1]
                self.MetaData = loadgeo[2]
                self.Items = loaditems[0]
                self.Data = loaddata[0].copy()
                self.Type = loaddata[1]
                self.NoData = loaddata[2]
                self.npType = loaddata[3]

                ray.shutdown()
                if self.verbose:
                    print("Finished MultiThread data reading")


                #p=mp.Pool(processes=3)
                #res=p.map(self._LoadParallel,[1,2,3])

                #print(len(res[0]))

                #manager = mp.Manager()
                #self.return_dict = manager.dict()

                #q1 = mp.Queue()
                #q2 = mp.Queue()
                #q3 = mp.Queue()
                #q4 = mp.Queue()

                #p1 = mp.Process(target=self._LoadData,args=(q1,))
                #p2 = mp.Process(target=self._LoadDBLayers,args=(q2,))
                #p3 = mp.Process(target=self._LoadGeo,args=(q3,))
                #p4 = mp.Process(target=self._LoadDBItems,args=(q4,))

                #p1.start()
                #p2.start()
                #p3.start()
                #p4.start()

                #p1.join()
                #p2.join()
                #p3.join()
                #p4.join()

                #print("Starting sort")
                #self._SortReturn(q1.get_nowait(),q2.get_nowait(),q3.get_nowait(),q4.get_nowait())
                #print("Ended sort")

                """
                p1 = threading.Thread(target=self._LoadData)
                p2 = threading.Thread(target=self._LoadDBLayers)
                p3 = threading.Thread(target=self._LoadGeo)
                p1.start()
                p2.start()
                p3.start()
                #out1 = q1.get()
                #out2 = q2.get()
                p1.join()
                p2.join()
                p3.join()
                """
            else:
                if self.verbose:
                    print("SingleThread option select")

                self.Layers, self.Year,dummy=_LoadDBLayers(self.Name,self.TimeSeries,verbose=self.verbose)
                self.Data, self.Type, self.NoData, self.npType, dummy=_LoadData(self.Name,self.Template,verbose=self.verbose)
                self.LLx,self.LLy,self.MetaData,dummy=_LoadGeo(self.Name,self.Compressed,verbose=self.verbose)
                self.Items,dummy=_LoadDBItems(self.Name,verbose=self.verbose)

                if self.verbose:
                    print("Finished SingleThread data reading")

            self.Data.setflags(write=1)

            self.nLayers = int(len(self.Layers.index))
            firstLayer=self.Layers.iloc[0,:]
            self.nByte = int(firstLayer['ValueSize'])
            self.nRows = int(firstLayer['RowNum'])
            self.nCols = int(firstLayer['ColNum'])
            self.ResX = float(firstLayer['CellWidth'])
            self.ResY = float(firstLayer['CellHeight'])
            self.nType = int(firstLayer['ValueType'])
            #self.npType = self._npType(self.Type)
            #self.Items = _LoadDBItems(self.Name)
            self._ReshapeData()
            self._AdjustGeo()
        elif not template == None:  # If DataStream and Template not None
            Result = self.FindFile(template)
            self.Template = Result['Name']
            self.TemplNet = Result['Network']
            if self.TemplNet:  # We have a network as template need to load the mapping
                self.Cells = self._LoadDBCells(self.Template)
                # self.Cells = _LoadDBCells(self.Template)
            self._LoadDBLayers()
            self.nLayers = int(len(self.Layers.index))
            firstLayer=self.Layers.iloc[0,:]
            self.nByte = int(firstLayer['ValueSize'])
            self.nRows = int(firstLayer['RowNum'])
            self.nCols = int(firstLayer['ColNum'])
            self.ResX = float(firstLayer['CellWidth'])
            self.ResY = float(firstLayer['CellHeight'])
            self.nType = int(firstLayer['ValueType'])
            self.npType = self._npType(self.nType)
            self._Load_ds()
        elif (self.nLayers is not None and
              self.nRows is not None and
              self.nCols is not None and
              self.nByte is not None and
              self.nType is not None and
              self.ResX is not None and
              self.ResY is not None):  # If a DataStream and template None, check basic paramenters are set before loading the DS
            self.npType = self._npType(self.nType)
            self._Load_ds()
        else:
            raise Exception('Not enough parametes defined to load DS file')


    def _npType(self, nType):
        # nType values: 2=int,3=float
        if nType == 2:
            if self.nByte == 2:
                return np.int16
            elif self.nByte == 4:
                return np.int32
        elif nType == 3:
            if self.nByte == 4:
                return np.float32
            elif self.nByte == 8:
                return np.float64
        else:
            raise Exception('Unknown value format: type {}, size {}'.format(nType, self.nByte))


    def SetType(self, nType):
        # print(nType)
        if nType == np.int16:
            # print('Type set to 5')
            self.Type = 5
        elif nType == np.int32:
            # print('Type set to 6')
            self.Type = 6
        elif nType == np.float32:
            # print('Type set to 7')
            self.Type = 7
        elif nType == np.float64:
            # print('Type set to 8')
            self.Type = 8
        else:
            raise Exception('Unknown value format: type {}'.format(nType))

    def _Load_ds(self):

        if self.TemplNet:
            perLayer = self.Cells.shape[0]
        else:
            perLayer = int(self.nRows * self.nCols * self.nByte)

        data = bytearray()

        with open(self.Name, "rb") as ifile:
            data = bytearray()
            for i in range(0, self.nLayers):
                dump40 = MFdsHeader()
                ifile.readinto(dump40)
                self.Type = dump40.Type
                if dump40.Type > 6:
                    self.NoData = dump40.Missing.Float
                else:
                    self.NoData = dump40.Missing.Int
                data = data + bytearray(ifile.read(perLayer))
        self.Data = np.frombuffer(data, dtype=self.npType)
        self.Data.shape = (self.nLayers, self.nRows, self.nCols)
        for i in range(0, self.nLayers):
            self.Data[i, :, :] = np.flipud(self.Data[i, :, :])

    def _ReshapeData(self):

        if self.verbose:
            print('Started _ReshapeData')
            start_time = time.time()

        #headLen=int(40/self.Type)
        headLen=int((len(self.Data)/self.nLayers) - (self.nRows * self.nCols))

        self.LayerHeaderLen = headLen

        if self.Type > 6:
            self.Data[self.Data == self.NoData] = np.nan
        else:
            self.Data[self.Data == self.NoData] = 0

        self.Data.shape = (self.nLayers, headLen + self.nRows * self.nCols)
        self.Data=self.Data[:,headLen:].reshape(self.nLayers, self.nRows, self.nCols)

        #self.Data.shape = (self.nLayers, self.nRows, self.nCols)
        for i in range(0, self.nLayers):
            self.Data[i, :, :] = np.flipud(self.Data[i, :, :])

        if self.verbose:
            print('Finished _ReshapeData in {} minutes'.format((time.time() - start_time) / 60))

    def _AdjustGeo(self):
        self.URx = self.LLx + (self.nCols * self.ResY)
        self.URy = self.LLy + (self.nRows * self.ResY)
        if ((self.LLx >= -180.0) and (self.URx <= 180.0) and (self.LLy >= -90.) and (self.URy <= 90.)):
            self.Projection = "Spherical"
        else:
            self.Projection = "Cartesian"

    def Save(self, OutFile, Template):
        cmd = Dir2Ghaas + '/bin/ds2rgis ' + \
        '-m ' + Template + \
        ' -t {} '.format(self.MetaData['title']) + \
        ' -d {} '.format(self.MetaData['geodomain']) + \
        ' -u {} '.format(self.MetaData['subject']) + \
        ' -v {} '.format(self.MetaData['version'])

        if self.verbose:
            print(cmd)
        with open(OutFile, "wb") as ifile:
            p = sp.Popen(cmd, stdout=ifile, stdin=sp.PIPE, stderr=sp.STDOUT, shell=True)
            for i in range(0, self.nLayers):
                h = MFdsHeader()
                h.Swap = 1
                h.Type = self.Type
                h.ItemNum = self.nRows * self.nCols
                if self.Type > 6:
                    h.Missing.Float = float(self.NoData)
                else:
                    h.Missing.Int = int(self.NoData)
                if self.TimeSeries:
                    h.Date = pd.to_datetime(str(self.Layers.index[i]))
                else:
                    h.Date = self.Layers.Name[i].encode()
                MakeGDBC = p.communicate(input=bytearray(h) + bytearray(np.flipud(self.Data[i, :, :])))[0]

    def SaveAs(self, OutFile, Template, Title="", Date=False, step='', Y=0, M=0, D=0, R=0, I=0):
        if OutFile[len(OutFile) - 3:len(OutFile)] == '.gz':
            Compress = True
        else:
            Compress = False
        if Title == "":
            Title = self.MetaData['title']
        cmd = [Dir2Ghaas + '/bin/ds2rgis',
               '-m', Template,
               '-t', Title,
               '-d',self.MetaData['geodomain'],
               '-u',self.MetaData['subject'],
               '-v',self.MetaData['version']
               ]  # - ' + OutFile

        OutData = bytearray()
        for i in range(0, self.nLayers):
            h = MFdsHeader()
            h.Swap = 1
            h.Type = self.Type
            h.ItemNum = self.nRows * self.nCols
            if self.Type > 6:
                h.Missing.Float = float(self.NoData)
            else:
                h.Missing.Int = int(self.NoData)
            if self.TimeSeries:
                h.Date = self.Layers.index[i].strftime("%Y-%m-%d").encode()
                if step == '':
                    # If it is a time series, then we need to always save with
                    # the Date option on
                    Date = True
                    # And we need to convert Pandas date information
                    # to rgis_package step...
            else:
                if type(self.Layers.Name[0]) == 'str':
                    h.Date = self.Layers.Name[i].encode()
                else:
                    h.Date = str(self.Layers.Name[i]).encode()
            bHeader = bytearray(h)
            flipped = np.flipud(self.Data[i, :, :]).flatten()
            mask = np.isnan(flipped)
            flipped[mask] = self.NoData
            bData = bHeader + flipped.tobytes()
            OutData.extend(bData)
        if Date or Compress:
            FinalFile = OutFile
            OutFile = '/tmp/tmp' + str(random.randint(10000, 99999)) + '.gdbc'
        with open(OutFile, "wb") as ifile:
            p = sp.Popen(cmd, stdout=ifile, stdin=sp.PIPE, stderr=sp.STDOUT, bufsize=1)  # 2000000)
            p.communicate(input=OutData)[0]
            p.terminate()
        if Date:
            self.DateLayer(OutFile, FinalFile, step, Y, M, D, R, I)
            # If we have to run through DateLayer, then the RiverGIS command grdDateLayer will
            # take care of the compression, so we don't need to run through the compression
            # option
            Compress = False
            if os.path.isfile(OutFile):
                os.remove(OutFile)

        if Compress:
            with open(OutFile, 'rb') as f_in, gzip.open(FinalFile, 'wb') as f_out:
                f_out.write(f_in.read())
            if os.path.isfile(OutFile):
                os.remove(OutFile)
        return

    def DateLayer(self, InFile, OutFile, step, Y=0, M=0, D=0, R=0, I=0):
        cmd = Dir2Ghaas + '/bin/grdDateLayers '
        if step.lower() == 'minute':
            cmd = cmd + ' -y ' + str(Y)
            cmd = cmd + ' -m ' + str(M)
            cmd = cmd + ' -d ' + str(D)
            cmd = cmd + ' -r ' + str(R)
            cmd = cmd + ' -i ' + str(I)
            cmd = cmd + ' -e minute '
        elif step.lower() == 'hour':
            cmd = cmd + ' -y ' + str(Y)
            cmd = cmd + ' -m ' + str(M)
            cmd = cmd + ' -d ' + str(D)
            cmd = cmd + ' -r ' + str(R)
            cmd = cmd + ' -e hour '
        elif step.lower() == 'day':
            cmd = cmd + ' -y ' + str(Y)
            cmd = cmd + ' -m ' + str(M)
            cmd = cmd + ' -d ' + str(D)
            cmd = cmd + ' -e day '
        elif step.lower() == 'month':
            cmd = cmd + ' -y ' + str(Y)
            cmd = cmd + ' -m ' + str(M)
            cmd = cmd + ' -e month '
        elif step.lower() == 'year':
            cmd = cmd + ' -y ' + str(Y)
            cmd = cmd + ' -e year '
        else:
            raise Exception('Unknown option in grdDateLayers: {}'.format(step))
        cmd = cmd + InFile + ' ' + OutFile
        sp.call(cmd, shell=True)

    def Recast(self, AsType):
        self.npType = AsType
        if self.npType == np.int16:
            self.nByte = 2
            self.Type = 5
        elif self.npType == np.float64:
            self.nByte = 8
            self.Type = 8
        else:
            self.nByte = 4
            if self.npType == np.float32:
                self.Type = 7
            else:
                self.Type = 6
        if self.verbose:
            print(AsType, self.nByte, self.Type)
        tmp = np.empty(self.Data.shape)
        tmp = self.Data.astype(AsType)
        if self.verbose:
            print(self.Data.dtype)
            print(tmp.dtype)
        self.Data = tmp
        if self.verbose:
            print(self.Data)

    def AddLayer(self, Name, Num=1, Frequency=None):
        # Depending on the value of Num (e.g. the number of layers to add)
        # Name is either a string or a list of names
        #
        # So here we test if we have a list of names that matches the number
        # of layers to add
        #
        if Num > 1:
            if not isinstance(Name, list):
                raise Exception('Adding multiple layers. Expected a list of names')
            else:
                Names = Name
        else:
            Names = [Name]
        # Now we define the index for the new layers
        MaxIndex = self.Layers.index.max()
        if self.TimeSeries:
            Frequency = self.Layers.index.freqstr
            ind = pd.date_range(MaxIndex, periods=Num + 1, freq=Frequency)[1:]
            TableItems = 7
        else:
            Frequency = None
            ind = [MaxIndex + x + 1 for x in range(0, Num)]
            TableItems = 8
        addLayers = pd.DataFrame([[np.nan] * TableItems] * Num, columns=self.Layers.columns, index=ind)
        firstLayer = self.Layers.iloc[0, :]
        addLayers['ValueSize'] = int(firstLayer['ValueSize'])
        addLayers['RowNum'] = int(firstLayer['RowNum'])
        addLayers['ColNum'] = int(firstLayer['ColNum'])
        addLayers['CellWidth'] = float(firstLayer['CellWidth'])
        addLayers['CellHeight'] = float(firstLayer['CellHeight'])
        addLayers['ValueType'] = int(firstLayer['ValueType'])
        if not self.TimeSeries:
            addLayers['ID'] = addLayers.index
            NameI = 0
            for i in ind:
                addLayers['Name'] = Names[NameI]
                NameI += 1
        else:
            IDnew = self.Layers['ID'].max() + 1
            for i in ind:
                addLayers['ID'] = IDnew
                IDnew += 1

        self.Layers = self.Layers.append(addLayers)
        if self.TimeSeries:
            self.Layers.index = pd.DatetimeIndex(self.Layers.index.values, freq=Frequency)
        firstLayer=self.Layers.iloc[0,:]
        tmp = np.full(int(firstLayer['RowNum']) * int(firstLayer['ColNum']) * Num, np.nan)
        self.Data = np.concatenate((self.Data.flatten(), tmp))
        self.Data.shape = (self.nLayers + Num, self.nRows, self.nCols)
        self.nLayers += Num
        return

    def Copy(self):
        return deepcopy(self)

    def Xarray(self):
        inX = self.LLx + self.ResX / 2.
        inY = self.LLy + self.ResY / 2.
        xcoords = [inX + self.ResX * n for n in range(0, self.nCols)]
        ycoords = [inY + self.ResY * n for n in range(0, self.nRows)]
        ycoords.reverse()
        if self.TimeSeries:
            out_xr = xr.DataArray(self.Data, dims=['time','latitude','longitude'],
                                  coords={'time':self.Layers.index, 'latitude':ycoords, 'longitude':xcoords})
                                  #coords=[('time', self.Layers.index), ('latitude', ycoords), ('longitude', xcoords)])
        else:
            # out_xr = xr.DataArray(self.Data, coords=[
            #    ('time', pd.DatetimeIndex(self.Layers['Name'], freq=pd.infer_freq(self.Layers['Name']))),
            #    ('latitude', ycoords), ('longitude', xcoords)])
            out_xr = xr.DataArray(self.Data, dims=['time', 'latitude', 'longitude'],
                                  coords = {'time': self.Layers['Name'].values,
                                  'latitude': ycoords, 'longitude': xcoords})
        if "subject" in self.MetaData.keys():
            out_xr.name=self.MetaData["subject"].lower()
        out_xr.attrs=self.MetaData
        out_xr.attrs["actual_range"]=[np.nanmin(self.Data),np.nanmax(self.Data)]
        return out_xr


####################################################
####################################################

class network():
    """
        Properties defined here:
            Name:     the fully qualifies name of the network to be loaded
            Data:     the list of variables generated by the WBM run
    """

    def __init__(self, Name=None):
        """
            Initializes the class and sets the GHAASDIR variable
        """
        # We check for the existence of both the uncompressed and the compressed version of the
        # file
        self.DataStream = False

        if Name == None:
            self.NewGrid = True
        else:
            self.NewGrid = False

        Result = _FindFile(Name)
        self.Name = Result['Name']
        self.Compressed = Result['Compressed']
        self.DataStream = Result['Datastream']

        #            if os.path.isfile(Name):
        #                self.Name=Name
        #                if Name[len(Name)-3:len(Name)]=='.ds':
        #                    self.DataStream=True
        #                else:
        #                    if Name[len(Name)-3:len(Name)]=='.gz':
        #                        self.Compressed=True
        #                    else:
        #                        self.Compressed=False
        #            elif os.path.isfile(Name+'.gz'):
        #                self.Name=Name+'.gz'
        #                self.Compressed=True
        #            else:
        #                raise Exception('File {} not found'.format(Name))

        self.Layers = None
        self.Cells = None
        self.nLayers = None
        self.nRows = None
        self.nCols = None
        self.nByte = None
        self.nType = None
        self.ResX = None
        self.ResY = None
        self.Type = None
        self.npType = None
        self.Projection = None
        self.LLx = None
        self.LLy = None
        self.URx = None
        self.URy = None
        self.NoData = None


####################################################
####################################################

class wbm():
    """
        Properties defined here:
            envFile:     the script that is used to run WBM from which the
                        environemental settings for the run are read
            EnvDict:    the dictionary with the setting read from the above
            OutDict:    the list of variables generated by the WBM run
    """

    def __init__(self, envFile, template=None):
        """
            Initializes the class and sets the GHAASDIR variable
        """
        if os.path.isfile(envFile):
            self.Exists = True
        else:
            self.Exists = False
            if os.path.isfile(template):
                print('Loading defautls from template: '.format(template))
            else:
                raise Exception(
                    'File {} does not exist, you need to specify a template to generate a new WBM script'.format(
                        template))

        self.envFile = envFile
        #        print(envFile)
        #        print(self.envFile)
        self.Template = template
        self.EnvDict = {}

        self.EnvDict['GHAASDIR'] = Dir2Ghaas

        self.funct = None
        self.args = None
        self.OutDict = None

    def WBMenvironment(self, projectDir=None, splitOutput=False):
        """
            Reads the WBM model environment from the WBM script file
            defined by envFile property
            Initializes both EnvDict and OutDict property
        """
        if self.Exists:
            loadFile = self.envFile
        else:
            loadFile = self.Template
        #        print(loadFile)
        with open(loadFile, 'rb') as source:
            InScript = self._clean_envFile(source)
        self.EnvDict = self._merge_two_dicts(self.EnvDict, InScript['Environment'])
        if 'PROJECTDIR' not in self.EnvDict:
            self.EnvDict['PROJECTDIR'] = os.path.dirname(self.envFile).replace('/Scripts', '')
        self.EnvDict['NEWSLINKDIR'] = self.EnvDict['PROJECTDIR']
        if projectDir is not None:
            if splitOutput:
                self.EnvDict['NEWSLINKDIR'] = projectDir
            else:
                self.EnvDict['PROJECTDIR'] = projectDir
                self.EnvDict['NEWSLINKDIR'] = self.EnvDict['PROJECTDIR']
        #        print(self.EnvDict)
        self._expandEnv()
        self.OutDict = InScript['Outputs']
        self._LoadOutputs()

    def GetFiles(self, Variable, timeStepType, timeStep):
        """
            Returns a list of all the files for a given output
            variable and timestep
        """
        if Variable not in self.OutDict:
            raise Exception('Variable {} not in current list of outputs'.format(Variable))
        VarDict = {Variable: {}}
        start = int(self.EnvDict['STARTYEAR'])
        end = int(self.EnvDict['ENDYEAR']) + 1
        for year in range(start, end):
            param = []
            #            param.append(self.EnvDict['RGISRESULTS'])
            #            param.append(self.EnvDict['DOMAIN'])
            param.append(self.EnvDict['OUTPUTPATH'])
            param.append(self.EnvDict['GEODOMAIN'])
            param.append(Variable)
            param.append(self.EnvDict['CONFIGURATION'])
            param.append('Static')
            param.append(timeStepType)  # self.EnvDict['STEPTYPE'])
            param.append(timeStep)  # self.EnvDict['TIMESTEP'])
            param.append(str(year))
            VarDict[Variable][year] = RGISfunction("RGISfilePath", param)
        return

    def _LoadOutputs(self):
        for i in self.OutDict:
            param = []
            #            param.append(self.EnvDict['RGISRESULTS'])
            #            param.append(self.EnvDict['DOMAIN'])
            param.append(self.EnvDict['OUTPUTPATH'])
            param.append(self.EnvDict['GEODOMAIN'])
            param.append(i)
            param.append(self.EnvDict['CONFIGURATION'])
            param.append('Static')
            param.append('TS')  # self.EnvDict['STEPTYPE'])
            param.append('daily')  # self.EnvDict['TIMESTEP'])
            path = RGISfunction("RGISdirectoryPath", param)
            param.append('0000')
            file = RGISfunction("RGISfilePath", param)
            file = file.replace(path, '')
            file = file.replace('_dTS0000', '_TimeStepTS0000')
            path = path.replace('/Daily', '/TimeStep')
            #            print(file,path)
            self.OutDict[i] = [path, file]

    def _clean_envFile(self, infile):
        """
            Actual read of the WBM script file
            Cleans all the non relevant lines of the script
            Works without defining special sections in the WBM file
        """
        Env = {}
        Out = {}
        copyOut = False
        copyEnv = False
        for line in infile:
            #            print line.strip()[0:19]
            # Skip any empty line
            line = line.decode('UTF-8')
            if len(line.strip()) != 0:
                # Skip any comment line and lines that start with ( which define the
                # array's counters
                if (line.lstrip()[0] != '#' and line.lstrip()[0] != '('):
                    # Now we read the requested output variables which can be used for
                    # further processing
                    if line.strip()[0:19] == 'OUTPUTS[${OutNum}]=':
                        #                        print('Found one output')
                        tmp = line.strip().replace('OUTPUTS[${OutNum}]="', '')
                        tmp = tmp[0:tmp.find('"')]
                        Out[tmp] = []
                    # This section reads the datasources (if ever needed)
                    elif line.strip()[0:24] == 'DATASOURCES[${DataNum}]=':
                        pass
                    # This section reads the options (if ever needed)
                    elif line.strip()[0:19] == 'OPTIONS[${OptNum}]=':
                        pass
                    # If we have a exactly one equal sign in the line we assume it's a
                    # environmental variable assignment
                    elif line.strip().count("=") == 1:
                        tmp = line.strip().split('=')
                        if '#' in tmp[1]:
                            tmp[1] = tmp[1][0:tmp[1].find('#')]
                        tmp[1] = tmp[1].replace("'", "").strip()
                        Env[tmp[0]] = tmp[1].replace('"', '').strip()
                        # The FwInit line contains the ultimate info for the following environment settings:
                        #    Model executable
                        #    Geographic domain of the analysis
                        #    Domain file - Generally set to the network file for the analysis
                        #    Path for GDS file
                        #    Path for output files
                        #    Path to RGIS binaries
                        #
                        # so we read the info and check if we need to update any of the environmental variables
                        # in our dictionary
                    elif line.strip()[0:6] == 'FwInit':
                        tmp = line.strip().split()
                        Env['EXECUTABLE'] = tmp[1].replace('"', '').strip()
                        Env['GEODOMAIN'] = tmp[2].replace('"', '').strip()
                        Env['ANALYSIDOMAIN'] = tmp[3].replace('"', '').strip()
                        Env['GDSPATH'] = tmp[4].replace('"', '').strip()
                        Env['OUTPUTPATH'] = tmp[5].replace('"', '').strip()

        return {'Err': 0, 'Environment': Env, 'Outputs': Out}

    def write(self):
        print('To be written')

    def _expandEnv(self):
        """
            Expands the shell variables defined in the WBM script
        """
        found = False
        for i in self.EnvDict:
            self.EnvDict[i] = self.EnvDict[i].replace('//', '/')
            item = self.EnvDict[i]
            if '${' in item:
                found = True
                ini = item.find('${')
                fin = item.find('}')
                rkey = item[(ini + 2):(fin)]
                self.EnvDict[i] = item.replace('${' + rkey + '}', self.EnvDict[rkey])
        if found:
            self._expandEnv()
        else:
            return

    def _merge_two_dicts(self, x, y):
        """
            Merges two dictionary and returns the resulting dictionary
        """
        z = x.copy()
        z.update(y)
        return z


####################################################
####################################################

def RGISfunction(funct, args):
    """
        Interface to the functions in RGISfunctions.sh
        Evaluates the function defined by funct with arguments
        args.
            funct:         the name of the function in RGISfunctions.sh to be
                        called
            args:        the list of arguments passed to the function above
    """
    passon = Dir2Ghaas + "/Scripts/RGISfunctions.sh " + funct
    if type(args) is not list:
        if type(args) is not str:
            args = str(args)
        passon = passon + " " + args
    else:
        for arg in args:
            if arg == "+":
                passon = passon + "+"
            else:
                if type(arg) is not str:
                    arg = str(arg)
                passon = passon + " " + arg
    if sys.version_info[0] < 3:
        tmp = list(_runProcess(passon.split()))[0]
        value = tmp.translate(None, ''.join('\n'))
    else:
        tmp = list(_runProcess(passon.split()))[0].decode('UTF-8')
        value = tmp.translate(str.maketrans('', '', '\n'))
    return value


####################################################
####################################################

def fwFunction(funct, args):
    """
        Interface to the functions in fwFunctions22.sh
        Evaluates the function defined by funct with arguments
        args.
            funct:         the name of the function in fwFunctions22.sh to be
                        called
            args:        the list of arguments passed to the function above
    """
    passon = Dir2Ghaas + "/Scripts/fwFpython.sh " + funct
    if type(args) is not list:
        if type(args) is not str:
            args = str(args)
        passon = passon + " " + args
    else:
        for arg in args:
            if arg == "+":
                passon = passon + "+"
            else:
                if type(arg) is not str:
                    arg = str(arg)
                passon = passon + " " + arg

    value = list(_runProcess(passon.split()))[0].translate(None, ''.join('\n'))
    return value


def _runProcess(exe):
    """
        Runs the RGISpython.sh with the appropriate parameters
        and returns the results
    """
    p = sp.Popen(exe, stdout=sp.PIPE, stderr=sp.STDOUT)
    while (True):
        retcode = p.poll()  # returns None while subprocess is running
        line = p.stdout.readline()
        yield line
        if (retcode is not None):
            break


__author__ = 'fabiocorsi'
__date__ = 'September 8, 2016'
