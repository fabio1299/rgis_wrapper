# RGIS_wrapper
Python wrapper for the RGIS (River GIS) framework and data structure

The rgis folder contains the code that implements the wrapper classes

The two other files are sample files 

## Installation
python >= 3.9 required

[RGIS](https://github.com/bmfekete/RGIS) required

```sh
git clone https://github.com/bmfekete/RGIS /tmp/RGIS \
 && /tmp/RGIS/install.sh /usr/local/share \
 && rm -rf /tmp/RGIS
```

The easiest way to get going is with conda to circumvent any gdal dependency issues. 

```sh

conda create -n rgis python=3.10 gdal

pip install git+https://github.com/fabio/rgis_wrapper.git
```
