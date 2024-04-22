import os
import pandas as pd
import json
import numpy as np



class DataPreprocessing():
    def __init__(self, path, independent_signals, dependent_signals, sequence_length):
        
      self.path = path
      self.input_signals = independent_signals
      self.output_signals = dependent_signals
      self.sequence_length = sequence_length
      #self.sampling_time_in_seconds = sampling_time_in_seconds
      self.U = None #input, independent_signal_data
      self.X = None #output, dependent_signal_data


      
      

    @staticmethod
    def extract_signals_and_time_from_raw_data(path_dataset, independent_signals,dependent_signals):
        dict_signals = {}
        dict_time = {}
        with open(path_dataset, "rb") as infile:
          dict_data = json.load(infile)

        for signame in independent_signals + dependent_signals:
          t_i, sig_i = zip(*dict_data[signame]['values'])
          t_i, sig_i = np.array(t_i), np.array(sig_i)
          # if signame in ['latitude_degree', 'longitude_degree']:
             # sig_i = sig_i - sig_i[0]
          dict_signals[signame] = sig_i
          dict_time[signame] = t_i

        return dict_signals, dict_time   
    
    def sampling_with_interpolation(self, sampling_time_in_seconds = 0.1):

      self.dict_signals , self.dict_time = DataPreprocessing.extract_signals_and_time_from_raw_data(self.path, self.input_signals, self.output_signals)
       
      t_min_in_s = np.min(self.dict_time['acceleration_x']) / 1e6
      t_max_in_s = np.max(self.dict_time['acceleration_x']) / 1e6
      self.t_interp = np.arange(0, t_max_in_s - t_min_in_s, step = sampling_time_in_seconds)

      list_input_signals = []
      for signame in self.input_signals:
        t_sig_norm = (self.dict_time[signame] - np.min(self.dict_time[signame])) / 1e6
        sig_i = np.interp(self.t_interp, t_sig_norm, self.dict_signals[signame])
        list_input_signals.append(sig_i)
    
      list_output_signals = []
      for signame in self.output_signals:
        t_sig_norm = (self.dict_time[signame] - np.min(self.dict_time[signame])) / 1e6
        sig_i = np.interp(self.t_interp, t_sig_norm, self.dict_signals[signame])
        list_output_signals.append(sig_i)
      
      self.U = np.stack(list_input_signals, axis=-1).astype('float32') 
      self.X = np.stack(list_output_signals, axis=-1).astype('float32')
      #return self.X.shape
       
    def normalize_train(self):

      self.mean_U, self.std_U = DataPreprocessing.compute_mean_and_stddev(self.U)
      self.mean_X, self.std_X = DataPreprocessing.compute_mean_and_stddev(self.X)

      self.U = DataPreprocessing.normalize(self.U, self.mean_U, self.std_U)
      self.X = DataPreprocessing.normalize(self.X, self.mean_X, self.std_X)

    def get_normalize_test(self):
      self.U = DataPreprocessing.normalize(self.U, self.mean_U, self.std_U)
      self.X = DataPreprocessing.normalize(self.X, self.mean_X, self.std_X)
      return self.t_interp, self.U, self.X
         
    def train_test_split(self, train_size=0.5):
      T_train = int(train_size * self.X.shape[0])
      idxs_train = np.arange(self.X.shape[0]) < T_train
      self.t_interp_train, self.t_interp_val = self.t_interp[idxs_train], self.t_interp[~idxs_train]
      self.U_train, self.U_val = self.U[idxs_train], self.U[~idxs_train]
      self.X_train, self.X_val = self.X[idxs_train], self.X[~idxs_train]

    @staticmethod
    def compute_mean_and_stddev(x):
      return np.mean(x, axis=0), np.std(x, axis=0)
    
    @staticmethod
    def normalize(x, mean, std):
      return (x - mean) / std    

    def get_train_test_data(self):
      return self.t_interp_train, self.U_train, self.X_train, self.t_interp_val, self.U_val,  self.X_val                      