import pandas as pd
import numpy as np
import zipfile
import os
import pymysql
import mysql.connector
import optuna
import math

from sqlalchemy import create_engine
from getpass import getpass
# from mysql.connector import connect, Error
# from pydantic_settings import BaseSettings
from config import settings
from datetime import timedelta
from numpy import linalg as LA
from scipy.linalg import sqrtm
from functools import partial
from sklearn.metrics import mean_squared_error
    
class SQLconnect:
    def __init__(self, conn_settings=settings):
        try:
            host = settings.hostname
        except:
            host = input("Enter host: ")
                
        try:
            username = settings.uname
        except:
            username = input("Enter username: ")
            
        try:
            pwd = settings.pwd
        except:
            pwd = getpass("Enter password: ")
        
        self.__host = host
        self.__username = username
        self.__pwd = pwd
      
    def get_sql_connection(self, dbname):
        
        # params: database name

        # connect
        connection = mysql.connector.connect(
            host=self.__host,
            user=self.__username,
            password=self.__pwd)
        
        try:
            connection.connect(database=dbname)
            # cursor = connection.cursor()

            if connection.is_connected() == True:
                print('connection is ready') 
            else:
                print('connection is NOT ready')
        except:
            print(f'error connection to {dbname}')
            # cursor = None

        return connection

class SQLRepository:
    
    def __init__(self, connection, host=settings.hostname, dbname = settings.dbname, dbname_stock = settings.dbname_stock, username=settings.uname, pwd=settings.pwd):
        
        # params: connection in previous step
        
        self.connection = connection
        self.__host = host
        self.__dbname = dbname
        self.__dbname_stock = dbname_stock
        self.__username = username
        self.__pwd = pwd
            
    def get_engine(self, dbname):
        
        # params: database name
        
        engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=self.__host, db=dbname, user=self.__username, pw=self.__pwd))
        
        return engine
    
    def import_data(self, folder_name: str = 'Datasets\\FX-1-Minute-Data') -> str:
        # params "folder_name" is a folder name containing data
        
        abs_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(abs_path)
        data_path = os.path.join(dir_path, folder_name) 
            
        pair_lists = os.listdir(data_path)
        n_inserted = []
        engine = self.get_engine(dbname=self.__dbname)

        for pair in pair_lists:

            # create  pair path 
            pair_path = os.path.join(data_path, pair)

            # show all zipped files
            zip_file_lists = os.listdir(pair_path)

            df_list = list()

            for zip_file in zip_file_lists:
                zf = zipfile.ZipFile(os.path.join(pair_path, zip_file))
                filenames = [filelist.filename for filelist in zf.filelist]
                csv_files = [filename for filename in filenames if filename[-3:]=='csv'] 

                for csv_file in csv_files:
                    df = pd.read_csv(zf.open(csv_file), delimiter=';', header=None)
                    df_list.append(df)

            df_merged = pd.concat(df_list, ignore_index=True)
            df_merged.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            df_merged['time'] = pd.to_datetime(df_merged['time'], format='%Y%m%d %H%M%S')
            df_merged = df_merged.sort_values(by='time')
            df_merged = df_merged.drop_duplicates(subset='time', keep='last')

            # df_merged.set_index('time', inplace=True, verify_integrity=True)

            # Write DF into SQL
            result = df_merged.to_sql(f'{pair}', engine, if_exists='fail', index=False)
            n_inserted.append(result)

        return n_inserted
    
    def import_test(self, data_path):
        
        # params: path of data saved in computer
        
        engine = self.get_engine(dbname=self.__dbname)
        
        pair_lists = os.listdir(data_path)
        n_inserted = []

        for pair in pair_lists[0:1]:

            # create  pair path 
            pair_path = os.path.join(data_path, pair)

            # show all zipped files
            zip_file_lists = os.listdir(pair_path)
            zip_file = zip_file_lists[0]
            
                      
            zf = zipfile.ZipFile(os.path.join(pair_path, zip_file))
            filenames = [filelist.filename for filelist in zf.filelist]
            csv_file = [filename for filename in filenames if filename[-3:]=='csv'] 
            
            
            df_merged = pd.read_csv(zf.open(csv_file[0]), delimiter=';', header=None)
            df_merged.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            df_merged['time'] = pd.to_datetime(df_merged['time'], format='%Y%m%d %H%M%S')
            df_merged = df_merged.sort_values(by='time')
            df_merged = df_merged.drop_duplicates(subset='time', keep='last')

            # Write DF into SQL
            result = df_merged.to_sql('test', engine, if_exists='append', index=False)
            n_inserted.append(result)
            
        return n_inserted
    
    def read_table(self, from_date, to_date, ticker, dbname, data_column, axis):
       
        df_list = []
        
        if dbname == 'db':
            engine = self.get_engine(dbname)
            
            for t in ticker:
                sql_template = f"SELECT * FROM {dbname}.{t} \
                WHERE time BETWEEN '{from_date}' and '{to_date}'"
                df = pd.read_sql_query(sql_template, engine, index_col='time')
                df['pair'] = t
                
                 # Get close price only
                if data_column == 'close':
                    df = df[['close', 'pair']]
            
                df_list.append(df)
                
            df = pd.concat(df_list, axis = axis, ignore_index=False)

            if data_column == 'close':
                df = df.reset_index().pivot(index='time', columns='pair', values='close')
            else:
                df = df.reset_index()\
                        .pivot(index='time', columns='pair')\
                        .reorder_levels(axis=1,order=[1,0]).sort_index(axis=1)
           
        if dbname == 'db_stock':
            engine = self.get_engine(dbname)

            for t in ticker:
                sql_template = f"SELECT * FROM {dbname}.{t} WHERE date BETWEEN '{from_date}' and '{to_date}'"
                df = pd.read_sql_query(sql_template, engine, index_col='date')
                df['symbol'] = t
                
                 # Get close price only
                if data_column == 'adjusted_close':
                    df = df[['adjusted_close', 'symbol']]
                df_list.append(df)
                
            df = pd.concat(df_list, axis = axis, ignore_index=False)
            
            if (data_column == 'adjusted_close') and len(ticker) == 1:
                df = df
            elif (data_column == 'adjusted_close') and len(ticker) != 1:
                df = df.reset_index().pivot(index='date', columns='symbol', values='adjusted_close')
            else:
                df = df.reset_index()\
                        .pivot(index='date', columns='symbol')\
                        .reorder_levels(axis=1,order=[1,0]).sort_index(axis=1)

        return df
    
    def insert_table_stock(self, ticker, data, if_exists='append'):
        
        # create table with primary key is column 'date'
        conn = self.connection
        cursor = conn.cursor()
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {ticker} (date DATETIME, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, adjusted_close DOUBLE, volume DOUBLE, dividend_amount DOUBLE, split_coefficient DOUBLE, symbol TEXT, PRIMARY KEY (date));")
        conn.commit()
        cursor.close()
        conn.close()
        
        # Insert data into tables
        engine = self.get_engine(dbname=self.__dbname_stock)
        n_inserted = data.to_sql(name=ticker, con=engine, if_exists=if_exists, index=True)
        engine.dispose()
        
        return n_inserted
    
class DataAnalysis:
    
    # def __init__(self):
    
    def week_start_end(self, d):
        w = d.weekday()
        if w<=5:
            week_start = d - timedelta(w+1)
        else:
            week_start = d
        return week_start.date()
    
    def data_wrangle(self, data):
        df1 = data.copy()
        
        # Update pairs names
        col1 = [t[:3] for t in df1.columns if t.index('usd') == 3]
        col1.sort()
        col2 = [t[3:] for t in df1.columns if t.index('usd') == 0]
        col2.sort()
        col1.extend(col2)
        df1.columns = col1

        # Inverse rate for pairs having USD as base currency (for ex: from USDJPY to JPYUSD - from USD base to JPY base)
        for i in col2:
            df1[f'{i}'] = (df1[f'{i}'])**(-1)
            
        # Add 'week' column showing the date of each Sunday (starting time of trading week) 
        week_start_list = []
        for i in df1.index:
            week_start = self.week_start_end(i)
            week_start_list.append(week_start)

        df1['week'] = week_start_list
        return df1
        
    def data_reindex(self, df1):

        # Reindex database to cover full trading hour (5 p.m. EST on Sunday until 5 p.m. EST on Friday)
        df_reindex = []
        for week, df_week in df1.groupby('week'):
            start_time = pd.to_datetime(f'{week} 17:00:00')
            date_index = pd.to_datetime(f'{week} 17:00:00') + pd.timedelta_range(start='0min', end='7199min', freq='1min')
            df_r = df_week.reindex(date_index, method='ffill')

            df_reindex.append(df_r)
        df_reindex = pd.concat(df_reindex)
        
        return df_reindex
    
    def compute_return(self, df_reindex, drop_from_time, drop_end_time):

        df_g10 = df_reindex.drop('week', axis=1).pct_change()*100

        df_g10.fillna(0, inplace=True)

        # Drop data during missing period identified in the previous step
        
        drop_from_hour = drop_from_time.split(sep=':')[0]
        drop_from_min = drop_from_time.split(sep=':')[1]

        drop_end_hour = drop_end_time.split(sep=':')[0]
        drop_end_min = drop_end_time.split(sep=':')[1]

        df_g10 = df_g10[df_g10.index.strftime('%H:%M').isin([f'{drop_from_hour}:{str(i).zfill(2)}' \
                            for i in range(int(drop_from_min), int(drop_end_min)+1)]) == False]
               
        return df_g10
    
    def compute_eig(self, df_g10):
        
        ret = df_g10.values
        idx_time = df_g10.index.values
        
        # define the in-sample and out-of-sample period
        in_sample = 60 # 60 mins in-sample
        out_sample = 10 # 10 mins following

        # compute covariance and eigenvalues
        dict_eig = {}

        for i in range(in_sample, len(ret)-out_sample): 
            try:
                in_cov = np.cov(ret[i-in_sample:i-1,:].T)
                out_cov = np.cov(ret[i:i+out_sample-1,:].T)
                inver_sqr = LA.inv(sqrtm(in_cov))
                risk_mat = np.matmul(np.matmul(inver_sqr, out_cov), inver_sqr)
                eigvalues, _ = LA.eig(risk_mat)
                dict_eig[idx_time[i]] = eigvalues
            # except Exception as e:
            #     raise(e)
            except:
                pass

        # concatenate eigenvalues 
        df_eig = pd.DataFrame(index=list(dict_eig.keys()), columns=range(1,len(df_g10.columns)+1), data=dict_eig.values())

        df_eig = df_eig[np.sum(np.isreal(df_eig.values),1)==9]

        df_eig = pd.DataFrame(index=df_eig.index, columns=df_eig.columns, data=np.real(df_eig.values))

        # only keep points in same week
        not_same_week_cond = ((df_eig.index.weekday == 6) & (df_eig.index.hour < 17)) \
                        | ((df_eig.index.weekday == 4) & (df_eig.index.hour > 15) & (df_eig.index.minute > 50))

        df_eig_filtered = df_eig[~not_same_week_cond]
        
        return df_eig_filtered

class OptimizeDensity:
    
    def emp_density(self, df_eig_final, percent):
        
        df_eig_melt = df_eig_final.melt(var_name='no', value_name='eigenvalue')
        cond = df_eig_melt < np.percentile(df_eig_melt['eigenvalue'], q=percent)
        eig_hist = df_eig_melt[cond]['eigenvalue'].dropna()
        
        hist, bin_edges = np.histogram(eig_hist, bins = 100, density=True)
        # Notes: bin_edges has length N+1 while hist has length N
        
        return eig_hist, hist, bin_edges
   
    def rho(self, q_in, q_out, bin_edges=None, lambda_array=None):
        temp = q_in + q_out * (1-q_in)
        lambda_max = ((1+ temp + 2*(temp**0.5)) / ((1 - q_in)**2)).real
        lambda_min = ((1+ temp - 2*(temp**0.5)) / ((1 - q_in)**2)).real
        
        if lambda_array is None:
            lambda_array = np.array(bin_edges[:-1])

        pos = np.maximum(0, (lambda_max - lambda_array) * (lambda_array - lambda_min))    

        p_lambda = (1-q_in)/(2*math.pi) * (pos**0.5) / (lambda_array * (q_in * lambda_array + q_out)) + max(0,(1 - 1/q_out))*(lambda_array==0)
        p_lambda = p_lambda.real

        return p_lambda, lambda_min, lambda_max, lambda_array

    def objective(self, trial, bin_edges=None, hist=None):

        q_in = trial.suggest_float("q_in", 0.001, 1, log=True)
        q_out = trial.suggest_float("q_out", 1, 4, log=True)

        # q_in = trial.suggest_float("q_in", q_in_min, q_in_max, log=True)
        # q_out = trial.suggest_float("q_out", q_out_min, q_out_max, log=True)

        p_lambda, lambda_min, lambda_max, lambda_array = self.rho(q_in, q_out, bin_edges=bin_edges)

        return mean_squared_error(hist, p_lambda)

    def optimize_params(self, n_trials, bin_edges, hist):
                
        study = optuna.create_study(direction='minimize')
        objective = partial(self.objective, bin_edges=bin_edges, hist=hist)
        study.optimize(objective, n_trials=n_trials)

        return study.best_trial.number, study.best_trial.value, study.best_params

    
        
    
    
        

    

    
    
    




