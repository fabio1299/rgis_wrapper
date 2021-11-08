from setuptools import setup

setup(name='rgis',
      version='0.1',
      description='RGIS python wrapper',
      url='https://github.com/fabio1299/RGISpy',
      author='Fabio Corsi',
      author_email='fcorsi@ccny.cuny.edu',
      license='MIT',
      packages=['rgis'],
      install_requires=[
          #'ray',
          'numpy',
          'pandas',
          'xarray',
          'psutil'
      ],
      zip_safe=False)
