"""
Handles the inteface between Python and the rgis framework
The goal is to expose the directory structure of the rgis
framework to Python to allow easy access to the RGISarchives
and the results of the WBM model runs
Contains:
class grid()
Loads a rgis grid file into a numpy multidimensional array
class wbm()
Reads the WBM script and loads the configuration variables,
which define the output variables produced and the directory
structure of the WBM results.
function RGISfunction()
It exposes the functions of the RGISfunctions.sh script.
!!!!!!!!!!! The following section kept for historical reference !!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!! NO LONGER NEEDED  !!!!!!!!!!!!!!!!!!!!!!!!!!
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

from ctypes import *
from copy import deepcopy
import sys
import gzip
import struct
import subprocess as sp
import os
import numpy as np
import pandas as pd
import random
import xarray as xr

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

if 'GHAASDIR' in os.environ:
    Dir2Ghaas = os.environ['GHAASDIR']
else:
    Dir2Ghaas = '/usr/local/share/ghaas'


class MFmissing(Union):
    _fields_ = [("Int", c_int),
                ("Float", c_double)]


class MFdsHeader(Structure):
    _fields_ = [("Swap", c_short),
                ("Type", c_short),
                ("ItemNum", c_int),
                ("Missing", MFmissing),
                ("Date", c_char * 24)]


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

def _LoadDBLayers(inFile, TimeSeriesFlag):
    # Loads the DBLayers table of the RiverGIS file. This table can be
    # loaded into a pandas DataFrame and the parameters are used to read the rest
    # of the file
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

def _LoadDBItems(inFile):  # ,TimeSeriesFlag):
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

def _LoadDBCells(inFile):
    # Loads a network DBCells table into a Pandas DAtaFrame
    cmd = Dir2Ghaas + '/bin/rgis2table -a DBCells '
    proc = sp.Popen(cmd + inFile, stdout=sp.PIPE, shell=True)

    data1 = StringIO(bytearray(proc.stdout.read()).decode("utf-8"))

    return pd.read_csv(data1, sep='\t')


####################################################
####################################################

class grid():
    """
        Properties defined here:
            Name:     the fully qualifies name of the file to be loaded
            Layers:   the dictionary with the grid settings read from the above
            Data:     the list of variables generated by the WBM run
    """

    def __init__(self, Name=None, TimeSeries=False):
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

    def Load(self, template=None):
        if not self.DataStream:  # If NOT a DataStream then expect GDBC and load DBLayers table
            self.Layers, self.Year = _LoadDBLayers(self.Name, self.TimeSeries)
            self.nLayers = int(len(self.Layers.index))
            self.nByte = int(self.Layers.ix[0]['ValueSize'])
            self.nRows = int(self.Layers.ix[0]['RowNum'])
            self.nCols = int(self.Layers.ix[0]['ColNum'])
            self.ResX = float(self.Layers.ix[0]['CellWidth'])
            self.ResY = float(self.Layers.ix[0]['CellHeight'])
            self.nType = int(self.Layers.ix[0]['ValueType'])
            self.npType = self._npType(self.nType)
            self.Items = _LoadDBItems(self.Name)
            self._LoadData(int(self.nRows * self.nCols * self.nByte))
            self._LoadGeo()
        elif not template == None:  # If DataStream and Template not None
            Result = self.FindFile(template)
            self.Template = Result['Name']
            self.TemplNet = Result['Network']
            if self.TemplNet:  # We have a network as template need to load the mapping
                self.Cells = _LoadDBCells(self.Template)
            self.Layers, self.Year = _LoadDBLayers(self.Template, self.TimeSeries)
            self.nLayers = int(len(self.Layers.index))
            self.nByte = int(self.Layers.ix[0]['ValueSize'])
            self.nRows = int(self.Layers.ix[0]['RowNum'])
            self.nCols = int(self.Layers.ix[0]['ColNum'])
            self.ResX = float(self.Layers.ix[0]['CellWidth'])
            self.ResY = float(self.Layers.ix[0]['CellHeight'])
            self.nType = int(self.Layers.ix[0]['ValueType'])
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

    def _LoadData(self, perLayer, template=''):

        if template == '':
            cmd = Dir2Ghaas + '/bin/rgis2ds '
        else:
            cmd = Dir2Ghaas + '/bin/rgis2ds -m ' + template
            # data=bytearray()
        dataparts = []
        proc = sp.Popen(cmd + self.Name, stdout=sp.PIPE, shell=True)
        data = bytearray(proc.stdout.read())
        self.Data = np.zeros((self.nLayers, self.nRows, self.nCols), dtype=self.npType)
        """
        for i in range(0,self.nLayers):
            bstart=(40 + perLayer)*i+39
            bend=bstart + perLayer
            self.Data[i,:,:]=np.frombuffer(data[bstart:bend], dtype=self.npType).reshape(self.nRows,self.nCols)
        dump40 = MFdsHeader()
        dump40 = data[0:39]
        """
        for i in range(0, self.nLayers):
            dump40 = MFdsHeader()
            proc.stdout.readinto(dump40)
            self.Type = dump40.Type
            # data=data+bytearray(proc.stdout.read(perLayer))
            # data.extend(bytearray(proc.stdout.read(perLayer)))
            dataparts.append(bytearray(proc.stdout.read(perLayer)))
        data = b"".join(dataparts)

        if dump40.Type > 6:
            self.NoData = dump40.Missing.Float
        else:
            self.NoData = dump40.Missing.Int
        """
        self.Data = np.frombuffer(data, dtype=self.npType)
        self.Data.setflags(write=1)
        """
        if dump40.Type > 6:
            self.Data[self.Data == self.NoData] = np.nan
        else:
            self.Data[self.Data == self.NoData] = 0
        self.Data.shape = (self.nLayers, self.nRows, self.nCols)
        for i in range(0, self.nLayers):
            self.Data[i, :, :] = np.flipud(self.Data[i, :, :])

    def _LoadGeo(self):
        if self.Compressed:
            with gzip.open(self.Name, "rb") as ifile:
                ifile.seek(40)
                self.LLx = struct.unpack('d', ifile.read(8))[0]
                self.LLy = struct.unpack('d', ifile.read(8))[0]
        else:
            with open(self.Name, "rb") as ifile:
                ifile.seek(40)
                self.LLx = struct.unpack('d', ifile.read(8))[0]
                self.LLy = struct.unpack('d', ifile.read(8))[0]
        self.URx = self.LLx + (self.nCols * self.ResY)
        self.URy = self.LLy + (self.nRows * self.ResY)
        if ((self.LLx >= -180.0) and (self.URx <= 180.0) and (self.LLy >= -90.) and (self.URy <= 90.)):
            self.Projection = "Spherical"
        else:
            self.Projection = "Cartesian"

    def Save(self, OutFile, Template):
        cmd = Dir2Ghaas + '/bin/ds2rgis ' + '-m ' + Template + ' -t pippo '
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
                    h.Date = self.Layers.Name[i]
                MakeGDBC = p.communicate(input=bytearray(h) + bytearray(np.flipud(self.Data[i, :, :])))[0]

    def SaveAs(self, OutFile, Template, Title, Date=False, step='', Y=0, M=0, D=0, R=0, I=0):
        if OutFile[len(OutFile) - 3:len(OutFile)] == '.gz':
            Compress = True
        else:
            Compress = False
        cmd = [Dir2Ghaas + '/bin/ds2rgis', '-m', Template, '-t', Title]  # - ' + OutFile

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
                    # to rgis step...
            else:
                h.Date = self.Layers.Name[i].encode()
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
        print(AsType, self.nByte, self.Type)
        tmp = np.empty(self.Data.shape)
        tmp = self.Data.astype(AsType)
        print(self.Data.dtype)
        print(tmp.dtype)
        self.Data = tmp
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
        addLayers['ValueSize'] = int(self.Layers.ix[0]['ValueSize'])
        addLayers['RowNum'] = int(self.Layers.ix[0]['RowNum'])
        addLayers['ColNum'] = int(self.Layers.ix[0]['ColNum'])
        addLayers['CellWidth'] = float(self.Layers.ix[0]['CellWidth'])
        addLayers['CellHeight'] = float(self.Layers.ix[0]['CellHeight'])
        addLayers['ValueType'] = int(self.Layers.ix[0]['ValueType'])
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

        tmp = np.full(int(self.Layers.ix[0]['RowNum']) * int(self.Layers.ix[0]['ColNum']) * Num, np.nan)
        self.Data = np.concatenate((self.Data.flatten(), tmp))
        self.Data.shape = (self.nLayers + Num, self.nRows, self.nCols)
        self.nLayers += Num
        return

    def Copy(self):
        return deepcopy(self)

    def Xarray(self):
        xcoords = [self.LLx + self.ResX * n for n in range(0, self.nCols)]
        ycoords = [self.LLy + self.ResY * n for n in range(0, self.nRows)]
        ycoords.reverse()
        if self.TimeSeries:
            out_xr = xr.DataArray(self.Data,
                                  coords=[('time', self.Layers.index), ('latitude', ycoords), ('longitude', xcoords)])
        else:
            out_xr = xr.DataArray(self.Data, coords=[
                ('time', pd.DatetimeIndex(self.Layers['Name'], freq=pd.infer_freq(self.Layers['Name']))),
                ('latitude', ycoords), ('longitude', xcoords)])
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
