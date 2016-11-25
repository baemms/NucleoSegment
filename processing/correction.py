"""
Storage and functionality to correct segmentation of nuclei
"""

import pickle
import os
import numpy as np
import random

from processing.segmentation import Segmentation
import storage.config as cfg

from storage.stacks import Stack


class Correction:

    def __init__(self, segmentation):
        """
        Initialise
        """
        self.segmentation = segmentation

        self.corr_nonuc = None
        self.corr_fila = None
        self.corr_filtered = None
        self.corr_added = None
        self.corr_overlap = None

        self.nuclei_nonuc = None
        self.nuclei_fila = None
        self.nuclei_filtered = None
        self.nuclei_added = None
        self.nuclei_overlap = None

        self.stacks = Stack()
        self.stacks.nonuc = None
        self.stacks.fila = None
        self.stacks.filtered = None
        self.stacks.added = None
        self.stacks.overlap = None

        # load corrections from experiment
        self.load_corrections()

    def add_correction_fila(self, nucleus, start, stop):
        """
        Add correction to start/stop of nucleus

        :param nucleus:
        :param start:
        :param stop:
        :return:
        """
        self.corr_fila = self.add_to_correction_list(self.corr_fila, nucleus, start, stop)

    def del_correction_fila(self, nucleus):
        """
        delete correction to start/stop of nucleus

        :param nucleus:
        :return:
        """
        self.corr_fila = self.del_from_correction_list(self.corr_fila, nucleus)

    def add_correction_nonuc(self, nucleus):
        """
        Add nucleus to non-nucleus list

        :param nucleus:
        :return:
        """
        self.corr_nonuc = self.add_to_correction_list(self.corr_nonuc, nucleus)

    def del_correction_nonuc(self, nucleus):
        """
        Delete nucleus from non-nucleus list

        :param nucleus:
        :return:
        """
        self.corr_nonuc = self.del_from_correction_list(self.corr_nonuc, nucleus)

    def add_nuclei_to_correction_filtered(self, nuclei):
        """
        Go through nuclei and add to correction filtered

        :param nuclei:
        :return:
        """

        for nucleus in nuclei:
            self.add_correction_filtered(nucleus)

    def add_correction_filtered(self, nucleus):
        """
        Add nucleus to filtered list

        :param nucleus:
        :return:
        """
        self.corr_filtered = self.add_to_correction_list(self.corr_filtered, nucleus)

    def del_correction_filtered(self, nucleus):
        """
        Delete nucleus from filtered list

        :param nucleus:
        :return:
        """
        self.corr_filtered = self.del_from_correction_list(self.corr_filtered, nucleus)

    def add_nuclei_to_correction_added(self, nuclei):
        """
        Go through nuclei and add to correction added

        :param nuclei:
        :return:
        """

        for nucleus in nuclei:
            self.add_correction_added(nucleus)

    def add_correction_added(self, nucleus):
        """
        Add nucleus to added list

        :param nucleus:
        :return:
        """
        self.corr_added = self.add_to_correction_list(self.corr_added, nucleus)

    def del_correction_added(self, nucleus):
        """
        Delete nucleus from added list

        :param nucleus:
        :return:
        """
        self.corr_added = self.del_from_correction_list(self.corr_added, nucleus)

    def add_nuclei_to_correction_overlap(self, nuclei):
        """
        Go through nuclei and add to correction overlap

        :param nuclei:
        :return:
        """

        for nucleus in nuclei:
            self.add_correction_overlap(nucleus)

    def add_correction_overlap(self, nucleus):
        """
        Add nucleus to overlap list

        :param nucleus:
        :return:
        """
        self.corr_overlap = self.add_to_correction_list(self.corr_overlap, nucleus)

    def del_correction_overlap(self, nucleus):
        """
        Delete nucleus from overlap list

        :param nucleus:
        :return:
        """
        self.corr_overlap = self.del_from_correction_list(self.corr_overlap, nucleus)

    def add_to_correction_list(self, corr_list, nucleus, start=-1, stop=-1):
        """
        Add nucleus to a correction list

        :param corr_list:
        :param nucleus:
        :return:
        """
        # save correction to list
        if corr_list is None:
            corr_list = list()

        if nucleus is not None:
            # is the nucleus already in the list?
            mod_lID = -1
            for lID, cur_nucleus in enumerate(corr_list):
                if type(cur_nucleus) is int:
                    cur_ID = cur_nucleus
                else:
                    cur_ID = cur_nucleus[0]

                if cur_ID == nucleus['nID']:
                    mod_lID = lID
                    break

            # create element to add
            if start >= 0 and stop >= 0:
                el_to_add = (nucleus['nID'], (start, stop))
            else:
                el_to_add = nucleus['nID']

            if mod_lID < 0:
                corr_list.append(el_to_add)
            else:
                corr_list[mod_lID] = el_to_add

            # update stacks
            self.update_correction_stacks()

        return corr_list

    def del_from_correction_list(self, corr_list, nucleus):
        """
        Delete nucleus to a correction list

        :param corr_list:
        :param nucleus:
        :return:
        """
        if nucleus is not None:
            # is the nucleus in the list?
            mod_lID = -1
            if corr_list is not None:
                for lID, cur_nucleus in enumerate(corr_list):
                    if type(cur_nucleus) is int:
                        cur_ID = cur_nucleus
                    else:
                        cur_ID = cur_nucleus[0]

                    if cur_ID == nucleus['nID']:
                        mod_lID = lID
                        break

            if mod_lID >= 0:
                # delete from correction
                corr_list.pop(mod_lID)

                # add to segmentation
                self.segmentation.add_nucleus_to_list(nucleus)

            # update stacks
            self.update_correction_stacks()

        return corr_list

    def is_correction_nonuc(self, nucleus):
        """
        Is the nucleus in the non-nucleus list?

        :param nucleus:
        :return:
        """
        # is the nucleus already in the list?
        nonuc = False

        if self.corr_nonuc is not None:
            nonuc = Correction.is_correction_nonuc_with_list(nucleus, self.corr_nonuc)

        return nonuc

    @staticmethod
    def is_correction_nonuc_with_list(nucleus, corr_nonuc_list):
        """
        Is the nucleus in the non-nucleus list?

        :param nucleus:
        :param corr_nonuc_list:
        :return:
        """
        # is the nucleus already in the list?
        nonuc = False

        for id, cur_nucleus in enumerate(corr_nonuc_list):
            if cur_nucleus == nucleus['nID']:
                nonuc = True

        return nonuc

    def save_corrections(self):
        """
        Save corrections to experiment folder

        :return:
        """
        # get path
        dirs = self.segmentation.get_results_dir()

        # store corrections
        if self.corr_fila is not None:
            with open(dirs.corr + cfg.file_corr_fila, "wb") as fin:
                pickle.dump(self.corr_fila, fin)

        if self.corr_nonuc is not None:
            with open(dirs.corr + cfg.file_corr_nonuc, "wb") as fin:
                pickle.dump(self.corr_nonuc, fin)

        if self.corr_filtered is not None:
            with open(dirs.corr + cfg.file_corr_filtered, "wb") as fin:
                pickle.dump(self.corr_filtered, fin)

        if self.corr_added is not None:
            with open(dirs.corr + cfg.file_corr_added, "wb") as fin:
                pickle.dump(self.corr_added, fin)

        if self.corr_overlap is not None:
            with open(dirs.corr + cfg.file_corr_overlap, "wb") as fin:
                pickle.dump(self.corr_overlap, fin)

    def load_corrections(self):
        """
        Load corrections from experiment folder

        :return:
        """
        # get path
        dirs = self.segmentation.get_results_dir()

        # load corrections
        if os.path.isfile(dirs.corr + cfg.file_corr_fila):
            with open(dirs.corr + cfg.file_corr_fila, "rb") as fin:
                self.corr_fila = pickle.load(fin)

        if os.path.isfile(dirs.corr + cfg.file_corr_nonuc):
            with open(dirs.corr + cfg.file_corr_nonuc, "rb") as fin:
                self.corr_nonuc = pickle.load(fin)

        if os.path.isfile(dirs.corr + cfg.file_corr_filtered):
            with open(dirs.corr + cfg.file_corr_filtered, "rb") as fin:
                self.corr_filtered = pickle.load(fin)

        if os.path.isfile(dirs.corr + cfg.file_corr_added):
            with open(dirs.corr + cfg.file_corr_added, "rb") as fin:
                self.corr_added = pickle.load(fin)

        if os.path.isfile(dirs.corr + cfg.file_corr_overlap):
            with open(dirs.corr + cfg.file_corr_overlap, "rb") as fin:
                self.corr_overlap = pickle.load(fin)

        self.update_correction_stacks()

    def update_correction_stacks(self):
        """
        Update correction stacks

        :return:
        """
        # create nuclei and stacks
        self.stacks.nonuc = self.update_correction_stack('nonuc')
        self.stacks.fila = self.update_correction_stack('fila')
        self.stacks.filtered = self.update_correction_stack('filtered')
        self.stacks.added = self.update_correction_stack('added')
        self.stacks.overlap = self.update_correction_stack('overlap')

    def update_correction_stack(self, var_name):
        """
        Update correction stack

        :param var_name:
        :return:
        """
        corr_list = getattr(self, 'corr_' + var_name)
        stack = getattr(self.stacks, var_name)

        # is the list not empty?
        if corr_list is not None:
            # create template stack
            stack = np.zeros_like(self.segmentation.stacks.nuclei)

            # create list of nuclei
            setattr(self, 'nuclei_' + var_name, list())
            nuclei_list = getattr(self, 'nuclei_' + var_name)

            # go through and add get nuclei
            for cur_nID in corr_list:
                if type(cur_nID) is int:
                    nID = cur_nID
                else:
                    nID = cur_nID[0]

                nucleus = self.segmentation.get_raw_nucleus_by_id(nID)

                # added nuclei are stored in the primary nuclei list
                if nucleus is None:
                    nucleus = self.segmentation.get_nucleus_by_id(nID)

                if nucleus is not None:
                    nuclei_list.append(nucleus)

            # add nuclei to stack
            if len(nuclei_list) > 0:
                stack = self.segmentation.add_nuclei_to_stack(nuclei_list,
                                                              stack, -1)

        return stack

    def apply_corrections(self, save=True):
        """
        Go through correction lists and apply changes

        :return:
        """
        # delete non-nuclei
        if self.corr_nonuc is not None:
            for to_delete in self.corr_nonuc:
                if self.segmentation.is_nucleus_in_nuclei(to_delete):
                    print('NONUC %i' % to_delete)
                    self.segmentation.remove_nucleus(to_delete)
                else:
                    print('attempted NONUC %i' % to_delete)

        # go through z corrections
        if self.corr_fila is not None:
            for fila in self.corr_fila:
                # remerge nucleus
                if self.segmentation.is_nucleus_in_nuclei(fila[0]):
                    print('FILA %i' % fila[0])
                    self.segmentation.remerge_nucleus(fila[0], fila[1][0], fila[1][1],
                                                      force_raw_labels_props=True)  # force to use raw labels
                else:
                    print('attemped FILA %i' % fila[0])

        # update segmentation
        self.segmentation.update(save=save, calc_nuclei_params=False)

        # update stacks
        self.update_correction_stacks()
