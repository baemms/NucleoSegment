"""
Wrapper class to handle organised data
Storage is based on pandas dataframes
"""

import pandas as pd
import numpy as np
import collections

import storage.config as cfg

class LookupFrame:

    def __init__(self, nuclei, cols=None):
        """
        Init datastructure

        :param cols:
        """
        # store nuclei for queries
        self.nuclei = nuclei

        # store cols
        self.data_cols = cols

        # internal storage
        self.data_frame = pd.DataFrame()

        # init columns
        if cols is not None and len(cols) > 0:
            # go through cols and add to data structure
            for col in cols:
                self.data_frame[col] = None

    def create_data_row(self, data_row):
        """
        Create a new data row and return index

        :param data_row:
        :return:
        """
        nIDs = self.get_nIDs()

        # get number of nIDs and add one
        if len(nIDs) > 0:
            next_nID = len(nIDs) + 1
        else:
            next_nID = 0

        # create new row
        new_row = pd.DataFrame(data_row, index=(next_nID, ), columns=self.data_cols)

        # append data to structure
        self.data_frame = self.data_frame.append(new_row)

        # return index
        return self.data_frame.index.tolist()[-1]

    def add_data_row(self, nID, data):
        """
        Add data row to structure

        :param nID:
        :param data:
        :return:
        """
        # add zeros to undefined values
        if len(self.data_cols) > data.shape[1]:
            data = np.append(data, np.zeros([len(data), len(self.data_cols) - data.shape[1]]), 1)

        # prepare indices as many as there are datapoints
        indices = [nID] * len(data)

        # create new rows
        new_data = pd.DataFrame(data, index=indices, columns=self.data_cols)

        # append data to structure
        self.data_frame = self.data_frame.append(new_data)

    def del_data_row(self, nID, col=None, arg=None):
        """
        Add data row to structure

        :param nID:
        :param data:
        :return:
        """
        # get all data except the one matching arg
        # FIX: the problem is that the nIDs are the indicies
        #      dropping the rows by indicies will drop all the data
        #      for the specific nucleus
        # COMMENT: You should have used multi-indexing!
        # get all values except the ones matching arg for nID
        if self.is_nID_in_data_frame(nID):
            filtered_arg = None

            if col is not None and arg is not None:
                filtered_arg = self.data_frame[(self.data_frame[col] != arg)].loc[nID]

            # drop all values from nID
            self.data_frame.drop(nID, inplace=True)

            if filtered_arg is not None:
                # append data
                self.data_frame = self.data_frame.append(filtered_arg)

    def remove_nan_rows(self, col):
        """
        Remove rows that have nan values

        :return:
        """
        self.data_frame = self.data_frame[np.isfinite(self.data_frame[col])]

    def update_data(self, nID, data, col):
        """
        Change data for nID at specific column

        :param nID:
        :param data:
        :param col:
        :return:
        """
        # is nID in the data frame?
        if self.is_nID_in_data_frame(nID):
            # get location of data and update column
            self.data_frame.loc[nID][col] = data

    def is_nID_in_data_frame(self, nID):
        """
        Check if the nID is in the data frame

        :param nID:
        :return:
        """
        is_in_data_frame = False

        if isinstance(nID, (int, float, np.int64, np.float64)) is False:
            if set(nID).issubset(set(self.data_frame.index.tolist())):
                is_in_data_frame = True
        elif nID in self.data_frame.index:
            is_in_data_frame = True

        return is_in_data_frame

    def is_col_in_data_frame(self, col):
        """
        Check if the col is in the data frame

        :param col:
        :return:
        """
        is_in_data_frame = False

        # single column?
        if type(col) is str and col in self.data_frame.columns.tolist():
            is_in_data_frame = True
        # more columns?
        elif type(col) is list and set(col) <= set(self.data_frame.columns.tolist()):
            is_in_data_frame = True

        return is_in_data_frame

    def get_data_from_nID(self, nID):
        """
        Return data from specific nucleus

        :param nID:
        :return:
        """
        df_nID = None

        if self.is_nID_in_data_frame(nID):
            df_nID = self.data_frame.loc[nID]

        return df_nID

    def calc_sum_from_col_for_nID(self, col, nID):
        """
        Calculate the sum from a column for a nucleus

        :param col:
        :param nID:
        :return:
        """

        sum_col_nID = -1

        if self.is_col_in_data_frame(col)\
                and self.is_nID_in_data_frame(nID):
            sum_col_nID = self.data_frame[col].loc[nID].sum()

        return sum_col_nID

    def get_min_from_col_for_nID(self, col, nID):
        """
        Get minimum from a column for a nucleus

        :param col:
        :param nID:
        :return:
        """

        min_col_nID = -1

        if self.is_col_in_data_frame(col)\
                and self.is_nID_in_data_frame(nID):
            min_col_nID = self.data_frame[col].loc[nID].min()

        return min_col_nID

    def get_max_from_col_for_nID(self, col, nID):
        """
        Get maximum from a column for a nucleus

        :param col:
        :param nID:
        :return:
        """

        max_col_nID = -1

        if self.is_col_in_data_frame(col)\
                and self.is_nID_in_data_frame(nID):
            max_col_nID = self.data_frame[col].loc[nID].max()

        return max_col_nID

    def get_vals(self):
        """
        Get all value rows
        :return:
        """
        vals = self.data_frame.values

        return vals

    def extract_accepted_nuclei(self, data_frame):
        """
        Extract accepted nuclei from data frame

        :param data_frame:
        :return:
        """
        vals = data_frame.values.copy()
        indicies = data_frame.index

        # go through indicies and check if nuclei were rejected
        for i, index in enumerate(indicies):
            # check if nucleus is rejected
            if self.nuclei.is_nucleus_rejected(index):
                vals[i] = None

        # delete all empty values
        vals = vals[~np.isnan(vals)]

        return vals

    def get_vals_for_nID(self, nID):
        """
        Get all value rows for nucleus

        :param nID:
        :return:
        """
        vals = None

        if self.is_nID_in_data_frame(nID):
            data = self.data_frame.loc[nID]

            # are there multiple values?
            if type(data) is pd.Series or type(data) is pd.DataFrame:
                vals = data.values
            else:
                vals = data

            if len(vals.shape) == 1:
                vals = np.array([vals])

        return vals


    def get_vals_from_col_for_nID(self, col, nID, elmnt=None, join_params=None):
        """
        Get value from column for nucleus

        :param col:
        :param nID:
        :param elmnt:
        :return:
        """
        val = None

        if self.is_col_in_data_frame(col):
            data = self.data_frame[col].loc[nID]

            # are there multiple values?
            if type(data) is pd.Series or type(data) is pd.DataFrame:
                # join params from main table?

                if join_params is not None:
                    data = pd.concat([data, self.nuclei.data_frames['data_params'].data_frame[join_params]],
                                     axis=1, join_axes=[data.index])

                elmnts = data.values

                val = elmnts

                if elmnt is not None:
                    if elmnt < len(elmnts):
                        val = elmnts[elmnt]

                        # is there only one column?
                        if type(col) is str and type(val) is np.array:
                            val = val[0]
            else:
                val = data

            # check if data is only a single row
            # FIX: why do I get a string? - ignore this case
            if type(val) is not str and val.ndim == 1:
                val = np.array([val])

        return val

    def get_val_from_col_for_nID(self, col, nID, elmnt=0, join_params=None):
        """
        Get value from column for nucleus

        :param col:
        :param nID:
        :param elmnt:
        :return:
        """

        return self.get_vals_from_col_for_nID(col, nID, elmnt, join_params=join_params)

    def get_vals_from_col(self, col, only_accepted=False, data_frame=None):
        """
        Get values from column for all nuclei

        :param col:
        :return:
        """
        vals = None

        if data_frame is None:
            data_frame = self.data_frame

        if self.is_col_in_data_frame(col):
            # go through indicies and check if nuclei were rejected
            if only_accepted is True:
                vals = self.extract_accepted_nuclei(data_frame[col])
            else:
                vals = data_frame[col].values

        return vals

    def change_vals_for_nID(self, nID, value, force_add=False):
        """
        Change values for nucleus

        :param nID:
        :param value:
        :param force_del:
        :return:
        """
        vals_changed = False

        if force_add is True or self.is_nID_in_data_frame(nID):
            # more than one value?
            if force_add is True or type(value) is np.ndarray:
                # drop data for nucleus
                if self.is_nID_in_data_frame(nID):
                    self.data_frame.drop(nID, inplace=True)

                # build indices
                if type(value) is np.ndarray:
                    indices = [nID] * value.shape[0]
                else:
                    indices = [nID]

                # create new rows
                new_row = pd.DataFrame(value, index=indices, columns=self.data_frame.columns)

                # append data
                self.data_frame = self.data_frame.append(new_row)
            else:
                self.data_frame.loc[nID] = value

            vals_changed = True

        return vals_changed

    def change_col_for_nID(self, col, nID, value, force_add=False, is_array=False):
        """
        Change value in column for nucleus

        :param col:
        :param nID:
        :param value:
        :param force_del:
        :param is_array:
        :return:
        """
        col_changed = False

        # add column first?
        if self.is_col_in_data_frame(col) is False:
            self.data_frame[col] = np.nan

        if force_add is True\
            or (self.is_col_in_data_frame(col)\
                and self.is_nID_in_data_frame(nID)):
            # more than one value?
            if force_add is True or type(value) is np.ndarray and is_array is False:
                # drop data for nucleus
                if self.is_nID_in_data_frame(nID):
                    self.data_frame.drop(nID, inplace=True)

                # build indices
                if type(value) is np.ndarray:
                    indices = [nID] * value.shape[0]
                else:
                    indices = [nID]

                # adjust columns
                if type(col) is str:
                    col = [col]

                # create new rows
                new_row = pd.DataFrame(value, index=indices, columns=col)

                # append data
                self.data_frame = self.data_frame.append(new_row)
            else:
                # convert to object if arrays are stored
                if is_array is True and self.data_frame[col].dtype.type != np.object_:
                    self.data_frame[col] = self.data_frame[col].astype(object)

                self.data_frame[col].loc[nID] = value

            col_changed = True

        return col_changed

    def get_nIDs(self, only_accepted=False):
        """
        Get nIDs from data frame

        :return:
        """
        # get a list of the indices
        if only_accepted is False:
            indices = self.data_frame.index.tolist()
        else:
            # filter nuclei out that were rejected
            indices = self.data_frame[self.data_frame['rejected'] < 1].index.tolist()

        # ensure unique indices
        nIDs = list(collections.OrderedDict.fromkeys(indices))

        return nIDs

    def save_as_csv(self, path):
        """
        Save dataframe as csv

        :param path:
        :return:
        """
        self.data_frame.to_csv(path, sep=cfg.CSV_DEL, float_format='%.2f')

    def load_from_csv(self, path):
        """
        Load dataframe from csv

        :param path:
        :return:
        """
        self.data_frame = pd.read_csv(path, sep=cfg.CSV_DEL, index_col=0)

    def sort_by_col(self, col, asc=[True], sort_by_nID=False, inplace=True):
        """
        Sort values by column

        :param col:
        :param asc:
        :return:
        """
        # adjust column if necessary
        if type(col) is str:
            col = [col]

        # add nID as sorting criteria at the beginning
        if sort_by_nID is True:
            col.insert(0, 'nID')
            asc.insert(0, True)

            self.add_col_from_index()
            new_df = self.data_frame.sort_values(col, ascending=asc)
            self.del_col_from_index()
        else:
            new_df = self.data_frame.sort_values(col, ascending=asc)

        if inplace is True:
            self.data_frame = new_df

        return new_df

    def add_col_from_index(self):
        """
        Add index as column to dataframe

        :return:
        """
        self.data_frame['nID'] = self.data_frame.index

    def del_col_from_index(self):
        """
        Add index as column to dataframe

        :return:
        """
        self.data_frame.drop('nID', axis=1, inplace=True)
