## Welcome to Data Wrangle section.

The first step which is considered very crucial is to import the currency pairs in foreign currency market to your relational database management system MySQL. 

#### Data specification:

The dataset of more than 66 currency pairs (1 Minute price) from 2000 to 2022 in zipped files.

Data source from [here](https://github.com/philipperemy/FX-1-Minute-Data/blob/master/README.md).

#### How to install:

Download data from the aforementioned source.

Save 3 files [import_FX_data.ipynb](https://github.com/huongmdang/QuantFinance/blob/main/FXData/src/import_FX_data.ipynb), [data_FX.py](https://github.com/huongmdang/QuantFinance/blob/main/FXData/src/data_FX.py), [config.py](https://github.com/huongmdang/QuantFinance/blob/main/FXData/src/config.py) to your local computer.

**Notes**: all files must locate in **the same parent directory** of the data folder.

*Example: My data path is '...Datasets/FX-1-Minute-Data/', where:*
- *'Datasets' is the parent folder, of which 3 files above should be placed in.*
- *'FX-1-Minute-Data' is an unzipped folder containing the FX data.*

Create '.env' file including the information 'hostname, dbname, uname and pwd' which is reused in various places.

Run file [import_FX_data.ipynb](https://github.com/huongmdang/QuantFinance/blob/main/FXData/src/import_FX_data.ipynb) **ONCE** to import all data. It may take several hours to complete. Please be patient! 

#### Contact:
Any questions please feel free to reach out to me at huong.dang.m@gmail.com or via [Linkedin](https://www.linkedin.com/in/huong-dang-bb589521/). 

Thanks for your time!

