import pandas as pd
import ray
import vectorbtpro as vbt
from logger_tt import logger


class Data_Handler:
    """
    Handles Data, everything from loading to saving

    Args:
         ():
         ():

    Attributes:
         ():
         ():

    Note:
        This is crazy
    """

    def __init__(self, genie_object):
        """Constructor for Data_Handler"""
        self.genie_object = genie_object

        #

    def print_dict(self, optional_object: object = None) -> object:
        import pprint
        pprint.pprint(self.__dict__ if not optional_object else optional_object.__dict__)

    @staticmethod
    def fetch_csv_data(data: object, data_files_dir: object) -> object:
        """

        Args:
            data:
            data_files_dir:

        Returns:
            object:

        """
        print(f'Loading {data} from CSV file')
        bar_data = pd.read_csv(f'{data_files_dir}/{data}.csv',
                               index_col=0, parse_dates=True)
        print(f'Finished Loading {data} from CSV file')
        return bar_data

    def fetch_data(self) -> object:
        """

        Returns:
            object: 
        """
        logger.info(f'Fetching Data')
        #
        if self.genie_object.runtime_settings['Data_Settings.load_CSV_from_pickle']:
            logger.warning("Loading data from pickle not reading from CSV")
            symbols_data = vbt.Data.load(
                f'{self.genie_object.data_dir_path}/{self.genie_object.runtime_settings["Data_Settings.saved_data_file"]}')

        else:
            import multiprocessing as mp
            number_of_pools = min(len(self.genie_object.runtime_settings["Data_Settings.data_files_names"]), 8)
            with mp.Pool(number_of_pools) as pool:
                data_array = [pool.apply_async(self.fetch_csv_data,
                                               args=(data, self.genie_object.runtime_settings[
                                                   "Data_Settings.data_files_dir"]))
                              for data in self.genie_object.runtime_settings["Data_Settings.data_files_names"]]
                data_array = [res.get() for res in data_array]

            datas_dict = {}
            for data_name, data_bars in zip(self.genie_object.runtime_settings["Data_Settings.data_files_names"],
                                            data_array):
                datas_dict[data_name] = data_bars

            symbols_data = vbt.Data.from_data(datas_dict)

            symbols_data.save(
                f'{self.genie_object.data_dir_path}/{self.genie_object.runtime_settings["Data_Settings.saved_data_file"]}')
            logger.info(
                f'Saved Symbol\'s Data to {self.genie_object.runtime_settings["Data_Settings.saved_data_file"]}')
            symbols_data.save(
                f'{self.genie_object.study_path}/{self.genie_object.runtime_settings["Data_Settings.saved_data_file"]}')

        setattr(self.genie_object, "symbols_data_id", ray.put(symbols_data))

        return self

    @staticmethod
    def fetch_dates_from_df(df, start_date=None, end_date=None):
        """

        Args:
            df:
            start_date:
            end_date:

        Returns:
            Cut DF
        """
        df_index = df.index
        mask = (df_index > start_date) & (df_index <= end_date)
        return df.loc[mask]

    def break_up_olhc_data_from_symbols_data(self) -> object:
        if not self.genie_object.symbols_data_id:
            self.fetch_data()

        '''Get OLHC'''
        symbols_data = ray.get(self.genie_object.symbols_data_id)
        open_data = symbols_data.get('Open')

        idx = pd.date_range(open_data.index[0], open_data.index[len(open_data) - 1], freq='1 min')

        open_data = open_data.tz_localize(None) if self.genie_object.runtime_settings[
            "Data_Settings.delocalize_data"] else open_data
        open_data = open_data.dropna() if self.genie_object.runtime_settings["Data_Settings.drop_nan"] else open_data
        open_data = open_data.ffill() if self.genie_object.runtime_settings["Data_Settings.ffill"] else open_data
        open_data = open_data.reindex(idx) if self.genie_object.runtime_settings[
            "Data_Settings.fill_dates"] else open_data

        low_data = symbols_data.get('Low')
        low_data = low_data.tz_localize(None) if self.genie_object.runtime_settings[
            "Data_Settings.delocalize_data"] else low_data
        low_data = low_data.dropna() if self.genie_object.runtime_settings["Data_Settings.drop_nan"] else low_data
        low_data = low_data.ffill() if self.genie_object.runtime_settings["Data_Settings.ffill"] else low_data
        low_data = low_data.reindex(idx) if self.genie_object.runtime_settings["Data_Settings.fill_dates"] else low_data

        high_data = symbols_data.get('High')
        high_data = high_data.tz_localize(None) if self.genie_object.runtime_settings[
            "Data_Settings.delocalize_data"] else high_data
        high_data = high_data.dropna() if self.genie_object.runtime_settings["Data_Settings.drop_nan"] else high_data
        high_data = high_data.ffill() if self.genie_object.runtime_settings["Data_Settings.ffill"] else high_data
        high_data = high_data.reindex(idx) if self.genie_object.runtime_settings[
            "Data_Settings.fill_dates"] else high_data

        close_data = symbols_data.get('Close')
        close_data = close_data.tz_localize(None) if self.genie_object.runtime_settings[
            "Data_Settings.delocalize_data"] else close_data
        close_data = close_data.dropna() if self.genie_object.runtime_settings["Data_Settings.drop_nan"] else close_data
        close_data = close_data.ffill() if self.genie_object.runtime_settings["Data_Settings.ffill"] else close_data
        close_data = close_data.reindex(idx) if self.genie_object.runtime_settings[
            "Data_Settings.fill_dates"] else close_data

        '''Separates the OLHC input data into two (or three) chunks:
                1.  Optimization Date-Range
                2.  Date-Range prior to "(1)" to be used for \bar{ATR}
                3.  The rest of the data that will not be utilized'''
        #  Split
        optimization_open_data = self.fetch_dates_from_df(open_data,
                                                          self.genie_object.optimization_start_date,
                                                          self.genie_object.optimization_end_date)
        optimization_low_data = self.fetch_dates_from_df(low_data,
                                                         self.genie_object.optimization_start_date,
                                                         self.genie_object.optimization_end_date)
        optimization_high_data = self.fetch_dates_from_df(high_data,
                                                          self.genie_object.optimization_start_date,
                                                          self.genie_object.optimization_end_date)
        optimization_close_data = self.fetch_dates_from_df(close_data,
                                                           self.genie_object.optimization_start_date,
                                                           self.genie_object.optimization_end_date)
        #
        bar_atr_days = self.genie_object.tp_sl_selection_space["bar_atr_days"]
        from datetime import timedelta
        one_day = timedelta(days=1, hours=0, minutes=0, seconds=0)
        #
        bar_atr_open_data = self.fetch_dates_from_df(open_data,
                                                     self.genie_object.optimization_start_date - bar_atr_days,
                                                     self.genie_object.optimization_start_date - one_day)
        bar_atr_low_data = self.fetch_dates_from_df(low_data,
                                                    self.genie_object.optimization_start_date - bar_atr_days,
                                                    self.genie_object.optimization_start_date - one_day)
        bar_atr_high_data = self.fetch_dates_from_df(high_data,
                                                     self.genie_object.optimization_start_date - bar_atr_days,
                                                     self.genie_object.optimization_start_date - one_day)
        bar_atr_close_data = self.fetch_dates_from_df(close_data,
                                                      self.genie_object.optimization_start_date - bar_atr_days,
                                                      self.genie_object.optimization_start_date - one_day)

        # # Set Optimization Data Attr's (ray.put)
        # setattr(self.genie_object, "optimization_open_data", ray.put(optimization_open_data))
        # setattr(self.genie_object, "optimization_low_data", ray.put(optimization_low_data))
        # setattr(self.genie_object, "optimization_high_data", ray.put(optimization_high_data))
        # setattr(self.genie_object, "optimization_close_data", ray.put(optimization_close_data))
        # #
        # # Set \bar{ATR} Data Attr's  (ray.put)
        # setattr(self.genie_object, "bar_atr_open_data", ray.put(bar_atr_open_data))
        # setattr(self.genie_object, "bar_atr_low_data", ray.put(bar_atr_low_data))
        # setattr(self.genie_object, "bar_atr_high_data", ray.put(bar_atr_high_data))
        # setattr(self.genie_object, "bar_atr_close_data", ray.put(bar_atr_close_data))
        #
        # Set Optimization Data Attr's
        setattr(self.genie_object, "optimization_open_data", optimization_open_data)
        setattr(self.genie_object, "optimization_low_data", optimization_low_data)
        setattr(self.genie_object, "optimization_high_data", optimization_high_data)
        setattr(self.genie_object, "optimization_close_data", optimization_close_data)
        #
        # Set \bar{ATR} Data Attr's
        setattr(self.genie_object, "bar_atr_open_data", bar_atr_open_data)
        setattr(self.genie_object, "bar_atr_low_data", bar_atr_low_data)
        setattr(self.genie_object, "bar_atr_high_data", bar_atr_high_data)
        setattr(self.genie_object, "bar_atr_close_data", bar_atr_close_data)
        return self

        ...
