"""
Lookup tables for nuclei
"""

import numpy as np

class Lookup:

    # table identifiers
    _tID_params_nID_ = 0
    _tID_params_volume_ = 1
    _tID_params_surface_ = 2
    _tID_params_depth_ = 3

    _tID_int_nID_ = 0
    _tID_int_lamin_ = 1
    _tID_int_dapi_ = 2
    _tID_int_membrane_ = 3

    _tID_centre_nID_ = 0
    _tID_centre_z_ = 1
    _tID_centre_y_ = 2
    _tID_centre_x_ = 3

    _tID_centroid_nID_ = 0
    _tID_centroid_z_ = 1
    _tID_centroid_y_ = 2
    _tID_centroid_x_ = 3

    _tID_coords_nID_ = 0
    _tID_coords_z_ = 1
    _tID_coords_y_ = 2
    _tID_coords_x_ = 3

    _tID_area_nID_ = 0
    _tID_area_z_ = 1
    _tID_area_val_ = 2

    # infos to get for dictionary
    _infos_to_get_ = ['box', 'lamin_box', 'labels_box', 'membrane_box', 'dapi_box',
                      'cropped_lamin_box', 'cropped_membrane_box', 'cropped_dapi_box', 'cropped_rgb_box',
                      'projection_z', 'projection_y', 'projection_x',
                      'img']

    def __init__(self):
        """
        Init lookup

        :param nuclei:
        """
        # tables
        self.tbl_params = None
        self.tbl_intensities = None
        self.tbl_centre = None
        self.tbl_centroid = None
        self.tbl_coords = None
        self.tbl_area = None

        # info dict
        self.info_dict = None

        # rows for tables
        self.tbl_rows_params = list()
        self.tbl_rows_intensities = list()
        self.tbl_rows_centre = list()
        self.tbl_rows_centroid = list()
        self.tbl_rows_coords = list()
        self.tbl_rows_area = list()

    def rebuild_tables(self, nuclei):
        """
        Rebuild tables with the given nuclei

        :param nuclei:
        :return:
        """
        # init tables
        self.init_tables()

        # create rows for tables
        for nucleus in nuclei:
            self.create_rows_for_nucleus(nucleus)

        self.tbl_params = self.create_table_from_rows(self.tbl_rows_params)
        self.tbl_intensities = self.create_table_from_rows(self.tbl_rows_intensities)
        self.tbl_centre = self.create_table_from_rows(self.tbl_rows_centre)
        self.tbl_centroid = self.create_table_from_rows(self.tbl_rows_centroid)
        self.tbl_coords = self.create_table_from_rows(self.tbl_rows_coords)
        self.tbl_area = self.create_table_from_rows(self.tbl_rows_area)

    def rebuild_info_dict(self, nuclei):
        """
        Go through nuclei and rebuild the information dictionary

        :param nuclei:
        :return:
        """
        # init dict
        if self.info_dict is None:
            self.info_dict = dict()

        # go through nuclei and add information
        for nucleus in nuclei:
            self.create_entry_in_info_dict(nucleus)

    def create_entry_in_info_dict(self, nucleus):
        """
        Create entry for nucleus in information dictionary

        :param nucleus:
        :return:
        """
        # create new entry
        if nucleus['nID'] in self.info_dict is False:
            self.info_dict[nucleus['nID']] = dict()

            # go through infos to get and map to infos dictionary
            for info in self._infos_to_get_:
                self.info_dict[nucleus['nID']][info] = nucleus[info]

    def create_table_from_rows(self, rows):
        """
        Create table from rows

        :param rows:
        :return:
        """
        new_table = np.array(rows)

        return new_table

    def create_rows_for_nucleus(self, nucleus):
        """
        Create rows in tables for nucleus

        :param nucleus:
        :return:
        """

        # create rows
        row_params = list()
        row_intensities = list()
        row_centre = list()
        rows_centroid = list()
        nrows_coords = list()
        rows_area = list()

        row_params.append(nucleus['nID'])
        row_params.append(nucleus['volume'])
        row_params.append(nucleus['surface'])
        row_params.append(nucleus['depth'])

        row_intensities.append(nucleus['nID'])
        row_intensities.append(nucleus['lamin_int'])
        row_intensities.append(nucleus['dapi_int'])
        row_intensities.append(nucleus['membrane_int'])

        row_centre.append(nucleus['nID'])
        row_centre.append(nucleus['centre'][0])
        row_centre.append(nucleus['centre'][1])
        row_centre.append(nucleus['centre'][2])

        for centroid in nucleus['centroid']:
            rows_centroid.append(list())
            rows_centroid[-1].append(nucleus['nID'])
            rows_centroid[-1].append(centroid[0])
            rows_centroid[-1].append(centroid[1][0])
            rows_centroid[-1].append(centroid[1][1])

        for coords in nucleus['coords']:
            nrows_coords.append(list())

            for coord in coords[1]:
                nrows_coords[-1].append(list())
                nrows_coords[-1][-1].append(nucleus['nID'])
                nrows_coords[-1][-1].append(coords[0])
                nrows_coords[-1][-1].append(coord[0])
                nrows_coords[-1][-1].append(coord[1])

        for area in nucleus['area']:
            rows_area.append(list())
            rows_area[-1].append(nucleus['nID'])
            rows_area[-1].append(area[0])
            rows_area[-1].append(area[1])

        # add rows to tables
        self.add_row_to_table_rows(row_params, self.tbl_rows_params)
        self.add_row_to_table_rows(row_intensities, self.tbl_rows_intensities)
        self.add_row_to_table_rows(row_centre, self.tbl_rows_centre)
        self.add_rows_to_table_rows(rows_centroid, self.tbl_rows_centroid)
        self.add_nrows_to_table_rows(nrows_coords, self.tbl_rows_coords)
        self.add_rows_to_table_rows(rows_area, self.tbl_rows_area)

    def add_nrows_to_table_rows(self, nrows, table_rows):
        """
        Add rows to table

        :param row:
        :param table_rows:
        :return:
        """
        # unpack rows
        for rows in nrows:
            self.add_rows_to_table_rows(rows, table_rows)

    def add_rows_to_table_rows(self, rows, table_rows):
        """
        Add rows to table

        :param row:
        :param table_rows:
        :return:
        """
        # unpack rows
        for row in rows:
            self.add_row_to_table_rows(row, table_rows)

    def add_row_to_table_rows(self, row, table_rows):
        """
        Add row to table

        :param row:
        :param table_rows:
        :return:
        """
        # add row to table
        table_rows.append(row)

    def init_tables(self):
        """
        Init tables for nuclei

        :return:
        """
        self.tbl_params = None
        self.tbl_intensities = None
        self.tbl_centre = None
        self.tbl_centroid = None
        self.tbl_coords = None
        self.tbl_area = None

        self.tbl_rows_params = list()
        self.tbl_rows_intensities = list()
        self.tbl_rows_centre = list()
        self.tbl_rows_centroid = list()
        self.tbl_rows_coords = list()
        self.tbl_rows_area = list()

    def get_nID_by_coords_pos(self, pos):
        """
        Return nID by position

        :param pos:
        :return:
        """
        row_z = None
        row_y = None
        row_x = None

        # lookup nID from coords for position
        indices_z = self.tbl_coords[:, self._tID_coords_z_] == pos[0]

        if indices_z is not None:
            row_z = self.tbl_coords[indices_z]

        if row_z is not None:
            indices_y = row_z[:, self._tID_coords_y_] == pos[1]

            if indices_y is not None:
                row_y = row_z[indices_y]

        if row_y is not None:
            indices_x = row_y[:, self._tID_coords_x_] == pos[2]

            if indices_x is not None:
                row_x = row_y[indices_x]

        nID = -1
        if row_x is not None and len(row_x) > 0:
            nID = row_x[0][0]

        return nID

    def get_nIDs_by_coords_pos_range(self, pos_range):
        """
        Return nIDs by position range

        :param pos:
        :return:
        """
        # lookup nID from coords for position
        row_z_a = self.tbl_coords[self.tbl_coords[:, self._tID_coords_z_] >= pos_range[0]]
        row_z_b = row_z_a[row_z_a[:, self._tID_coords_z_] <= pos_range[3]]

        row_y_a = row_z_b[row_z_b[:, self._tID_coords_y_] >= pos_range[1]]
        row_y_b = row_y_a[row_y_a[:, self._tID_coords_y_] <= pos_range[4]]

        row_x_a = row_y_b[row_y_b[:, self._tID_coords_x_] >= pos_range[2]]
        row_x_b = row_x_a[row_x_a[:, self._tID_coords_x_] <= pos_range[5]]

        unique_nIDs = np.unique(row_x_b[:, self._tID_coords_nID_])
        nIDs = list(unique_nIDs)

        return nIDs
