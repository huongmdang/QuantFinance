#### Welcome to my Quant Finance repo!

The repository includes three parts:

## Data wrangle

# Data specification:

- The dataset of more than 66 FX pairs (1 Minute price) from 2000 to 2022 in zipped files.

- Data source [online] (https://github.com/philipperemy/FX-1-Minute-Data/blob/master/README.md)

# How to install:

- Download data from the source.

- Save 3 files [FXdata/src/import_FX_data.ipynb], [FXdata/src/data_FX.py], [FXdata/src/config.py] to your local computer.

Notes: these files must locate in the same directory with the data folder.

Example: My current data path is 'Datasets/FX-1-Minute-Data/', where:
		'Datasets' is in the same directory with files above.
		'FX-1-Minute-Data' is a folder containing the FX pairs.
	
- Create '.env' file includes the information 'hostname, dbname, uname and pwd' which is reused in various places.
	
- Run file [FXdata/src/import_FX_data.ipynb] ONCE to import all data. It may take several hours to complete. 

## Project 1: G10 currencies - Eigenvalues approach applied Excess out-of-sample covariance

# Description:

Project 1 is going to examine the correlation between eigenvalues and the volatility of FX pairs through: 
- Analyze the correlation between G10 currencies using correlation heatmap.
- Derive the eigenvalues applying the concept of excess out-of-sample-covariance, which is thought to be a useful metric for explaining volatility.
- Compare empirical and theoretical eigenvalues density to find the optimal set of parameters.

# How to install:

- Save and run this file [FXdata/src/eigenvalues_FX.ipynb] in the same directory with the data related files.

# Reference link:

- Reference article [G10 currencies - Eigenvalues approach applied Excess out-of-sample covariance] (https://medium.com/@nmdang/g10-currencies-eigenvalue-approach-applied-excess-out-of-sample-covariance-70bfafc43a2e)

## Project 2: PCA (Principal Component Analysis) in FX Replication

# Description: 

Project 2 is to answer the common question in the predictive model. That is how to combine variables in the model such that it still keeps significant anticipation power but also minimizes the existing noise. 
Using the same idea in FX, the project is going to replicate AUD/USD return from the basket of 11 other pairs.

- Compute PCA to extract the number of optimal components.
- Estimate the weight of each component selected in the previous step.
- Replicate the target return and compare with the true return.

# How to install:

- Save and run this file [FXdata/src/PCA_FX.ipynb] in the same directory with the data related files.

# Reference link:

- Reference article [FX Replication using PCA](https://medium.com/@nmdang/fx-replication-using-pca-15d3c79e2a46)

# Contact:
- Contact information: huong.dang.m@gmail.com
