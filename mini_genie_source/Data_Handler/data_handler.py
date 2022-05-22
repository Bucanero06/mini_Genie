import pandas as pd
import ray
import vectorbtpro as vbt
from logger_tt import logger


class Data_Handler(object):
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

    def print_dict(self, optional_object=None):
        import pprint
        pprint.pprint(self.__dict__ if not optional_object else optional_object.__dict__)

    def fetch_csv_data(self, data, data_files_dir):
        print(f'Loading {data} from CSV file')
        bar_data = pd.read_csv(f'{data_files_dir}/{data}.csv',
                               index_col=0, parse_dates=True)
        print(f'Finished Loading {data} from CSV file')
        return bar_data

    def fetch_data(self):
        logger.info(f'Fetching Data')
        #
        # logger.info(self.print_dict(self.genie_object.runtime_settings))
        #
        # exit()
        if self.genie_object.runtime_settings['Data_Settings.load_CSV_from_pickle']:
            symbols_data = vbt.Data.load(
                f'{self.genie_object.data_path}/{self.genie_object.runtime_settings["Data_Settings.saved_data_file"]}')

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
                f'{self.genie_object.data_path}/{self.genie_object.runtime_settings["Data_Settings.saved_data_file"]}')
            logger.info(
                f'Saved Symbol\'s Data to {self.genie_object.runtime_settings["Data_Settings.saved_data_file"]}')
            symbols_data.save(
                f'{self.genie_object.study_path}/{self.genie_object.runtime_settings["Data_Settings.saved_data_file"]}')

        setattr(self.genie_object, "symbols_data_id", ray.put(symbols_data))

        return self

    def break_up_olhc_data_from_symbols_data(self):
        if not self.genie_object.symbols_data_id:
            self.fetch_data()

        symbols_data = ray.get(self.genie_object.symbols_data_id)
        open_data = symbols_data.get('Open')

        idx = pd.date_range(open_data.index[0], open_data.index[len(open_data) - 1], freq='1 min')

        open_data = open_data.tz_localize(None) if self.genie_object.runtime_settings["Data_Settings.delocalize_data"] else open_data
        open_data = open_data.dropna() if self.genie_object.runtime_settings["Data_Settings.drop_nan"] else open_data
        open_data = open_data.ffill() if self.genie_object.runtime_settings["Data_Settings.ffill"] else open_data
        open_data = open_data.reindex(idx) if self.genie_object.runtime_settings["Data_Settings.fill_dates"] else open_data

        low_data = symbols_data.get('Low')
        low_data = low_data.tz_localize(None) if self.genie_object.runtime_settings["Data_Settings.delocalize_data"] else low_data
        low_data = low_data.dropna() if self.genie_object.runtime_settings["Data_Settings.drop_nan"] else low_data
        low_data = low_data.ffill() if self.genie_object.runtime_settings["Data_Settings.ffill"] else low_data
        low_data = low_data.reindex(idx) if self.genie_object.runtime_settings["Data_Settings.fill_dates"] else low_data

        high_data = symbols_data.get('High')
        high_data = high_data.tz_localize(None) if self.genie_object.runtime_settings["Data_Settings.delocalize_data"] else high_data
        high_data = high_data.dropna() if self.genie_object.runtime_settings["Data_Settings.drop_nan"] else high_data
        high_data = high_data.ffill() if self.genie_object.runtime_settings["Data_Settings.ffill"] else high_data
        high_data = high_data.reindex(idx) if self.genie_object.runtime_settings["Data_Settings.fill_dates"] else high_data

        close_data = symbols_data.get('Close')
        close_data = close_data.tz_localize(None) if self.genie_object.runtime_settings["Data_Settings.delocalize_data"] else close_data
        close_data = close_data.dropna() if self.genie_object.runtime_settings["Data_Settings.drop_nan"] else close_data
        close_data = close_data.ffill() if self.genie_object.runtime_settings["Data_Settings.ffill"] else close_data
        close_data = close_data.reindex(idx) if self.genie_object.runtime_settings["Data_Settings.fill_dates"] else close_data

        setattr(self.genie_object, "open_id", ray.put(open_data))
        setattr(self.genie_object, "low_id", ray.put(low_data))
        setattr(self.genie_object, "high_id", ray.put(high_data))
        setattr(self.genie_object, "close_id", ray.put(close_data))

        return self
