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
        self.corr_remerge = None

        self.nuclei_nonuc = None
        self.nuclei_fila = None
        self.nuclei_filtered = None
        self.nuclei_added = None
        self.nuclei_overlap = None
        self.nuclei_remerge = None

        self.stacks = Stack()
        self.stacks.nonuc = None
        self.stacks.fila = None
        self.stacks.filtered = None
        self.stacks.added = None
        self.stacks.overlap = None
        self.stacks.remerge = None

        # load corrections from experiment
        self.load_corrections()

    def add_correction_fila(self, nID, start, stop):
        """
        Add correction to start/stop of nID

        :param nID:
        :param start:
        :param stop:
        :return:
        """
        self.corr_fila = self.add_to_correction_list(self.corr_fila, nID, start, stop)

    def del_correction_fila(self, nID):
        """
        delete correction to start/stop of nID

        :param nID:
        :return:
        """
        self.corr_fila = self.del_from_correction_list(self.corr_fila, nID)

    def add_correction_nonuc(self, nID):
        """
        Add nID to non-nID list

        :param nID:
        :return:
        """
        self.corr_nonuc = self.add_to_correction_list(self.corr_nonuc, nID)

    def del_correction_nonuc(self, nID):
        """
        Delete nID from non-nID list

        :param nID:
        :return:
        """
        self.corr_nonuc = self.del_from_correction_list(self.corr_nonuc, nID)

    def add_nuclei_to_correction_filtered(self, nID, add_to_stack=True):
        """
        Go through nuclei and add to correction filtered

        :param nID:
        :return:
        """

        for nID in nID:
            self.add_correction_filtered(nID, add_to_stack=add_to_stack)

    def add_correction_filtered(self, nID, add_to_stack=True):
        """
        Add nID to filtered list

        :param nID:
        :return:
        """
        self.corr_filtered = self.add_to_correction_list(self.corr_filtered, nID, add_to_stack=add_to_stack)

    def del_correction_filtered(self, nID):
        """
        Delete nID from filtered list

        :param nID:
        :return:
        """
        self.corr_filtered = self.del_from_correction_list(self.corr_filtered, nID)

    def add_nuclei_to_correction_added(self, nIDs):
        """
        Go through nuclei and add to correction added

        :param nIDs:
        :return:
        """

        for nID in nIDs:
            self.add_correction_added(nID)

    def add_correction_added(self, nID):
        """
        Add nID to added list

        :param nID:
        :return:
        """
        self.corr_added = self.add_to_correction_list(self.corr_added, nID)

    def del_correction_added(self, nID):
        """
        Delete nID from added list

        :param nID:
        :return:
        """
        self.corr_added = self.del_from_correction_list(self.corr_added, nID)

    def add_nuclei_to_correction_overlap(self, nIDs):
        """
        Go through nuclei and add to correction overlap

        :param nIDs:
        :return:
        """

        for nID in nIDs:
            self.add_correction_overlap(nID)

    def add_correction_overlap(self, nID):
        """
        Add nID to overlap list

        :param nID:
        :return:
        """
        self.corr_overlap = self.add_to_correction_list(self.corr_overlap, nID)

    def del_correction_overlap(self, nID):
        """
        Delete nID from overlap list

        :param nID:
        :return:
        """
        self.corr_overlap = self.del_from_correction_list(self.corr_overlap, nID)

    def add_correction_remerge(self, nID):
        """
        Add correction

        :param nID:
        :return:
        """
        self.corr_remerge = self.add_to_correction_list(self.corr_remerge, nID)

    def del_correction_remerge(self, nID):
        """
        delete correction to start/stop of nID

        :param nID:
        :return:
        """
        self.corr_remerge = self.del_from_correction_list(self.corr_remerge, nID)

    def add_to_correction_list(self, corr_list, nID, start=-1, stop=-1, add_to_stack=False):
        """
        Add nID to a correction list

        :param corr_list:
        :param nID:
        :return:
        """
        if corr_list is None:
            corr_list = list()

        if nID is not None:
            # is the nID already in the list?
            mod_lID = -1
            for lID, corr in enumerate(corr_list):
                cur_nID = corr

                if type(corr) is tuple:
                    cur_nID = corr[0]

                if cur_nID == nID:
                    mod_lID = lID
                    break

            # create element to add
            if start >= 0 and stop >= 0:
                el_to_add = (nID, (start, stop))
            else:
                el_to_add = nID

            if mod_lID < 0:
                corr_list.append(el_to_add)
            else:
                corr_list[mod_lID] = el_to_add

            # update stacks
            if add_to_stack is True:
                self.update_correction_stacks()

        return corr_list

    def del_from_correction_list(self, corr_list, nID):
        """
        Delete nID to a correction list

        :param corr_list:
        :param nID:
        :return:
        """
        if nID is not None:
            # is the nID in the list?
            mod_lID = -1
            if corr_list is not None:
                for lID, cur_nID in enumerate(corr_list):
                    # multiple value field?
                    if type(cur_nID) is tuple:
                        cur_nID = cur_nID[0]

                    if cur_nID == nID:
                        mod_lID = lID
                        break

            if mod_lID >= 0:
                # delete from correction
                corr_list.pop(mod_lID)

            # accept nucleus
            self.segmentation.nuclei.accept_nucleus(nID)

            # update stacks
            # takes a long time if you do that every time
            # self.update_correction_stacks()

        return corr_list

    def is_correction_nonuc(self, nID):
        """
        Is the nID in the non-nID list?

        :param nID:
        :return:
        """
        # is the nID already in the list?
        nonuc = False

        if self.corr_nonuc is not None:
            if nID in self.corr_nonuc:
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

        if self.corr_remerge is not None:
            with open(dirs.corr + cfg.file_corr_remerge, "wb") as fin:
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

        if os.path.isfile(dirs.corr + cfg.file_corr_remerge):
            with open(dirs.corr + cfg.file_corr_remerge, "rb") as fin:
                self.corr_remerge = pickle.load(fin)

        self.update_correction_stacks()

    def update_correction_stacks(self):
        """
        Update correction stacks

        :return:
        """

        print('TEST UNCOMMEND')
        # create nuclei and stacks
        #self.stacks.nonuc = self.update_correction_stack('nonuc')
        #self.stacks.fila = self.update_correction_stack('fila')
        #self.stacks.filtered = self.update_correction_stack('filtered')
        #self.stacks.added = self.update_correction_stack('added')
        #self.stacks.overlap = self.update_correction_stack('overlap')
        #self.stacks.remerge = self.update_correction_stack('remerge')

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
                nID = cur_nID

                if nID is not None:
                    nuclei_list.append(nID)

            # add nuclei to stack
            if len(nuclei_list) > 0:
                print('Create correction stack %s' % var_name)
                stack = self.segmentation.nuclei.add_nuclei_to_stack(stack, nIDs=nuclei_list)

        return stack

    def apply_corrections(self, save=True):
        """
        Go through correction lists and apply changes

        :return:
        """
        # delete non-nuclei
        if self.corr_nonuc is not None:
            for to_delete in self.corr_nonuc:
                if self.segmentation.nuclei.is_nucleus_in_nuclei(to_delete):
                    print('NONUC %i' % to_delete)
                    self.segmentation.nuclei.reject_nucleus(to_delete)
                else:
                    print('attempted NONUC %i' % to_delete)

        # go through z corrections
        if self.corr_fila is not None:
            for fila in self.corr_fila:
                # remerge nID
                if self.segmentation.nuclei.is_nucleus_in_nuclei(fila[0]):
                    print('FILA %i' % fila[0])
                    self.segmentation.nuclei.remerge_nucleus(fila[0],
                                                             int(float(fila[1][0])), int(float(fila[1][1])),
                                                             force_raw_labels_props=True)  # force to use raw labels
                else:
                    print('attemped FILA %i' % fila[0])

        # do remerges
        if self.corr_remerge is not None:
            for remerge in self.corr_remerge:
                # remerge nID
                if self.segmentation.nuclei.is_nucleus_in_nuclei(remerge) \
                    and self.segmentation.nuclei.get_nucleus_volume(remerge) < 0:
                    print('REMERGE %i' % remerge)
                    self.segmentation.nuclei.remerge_nucleus(remerge,
                                                             0, (self.segmentation.stacks.lamin.shape[0] - 1),
                                                             merge_depth=True, force_raw_labels_props=True)
                else:
                    print('attemped REMERGE %i' % remerge)

        # resort z values
        self.segmentation.nuclei.sort_vals_by_z()

        # update segmentation
        self.segmentation.update(save=save, calc_nuclei_params=False)

        # update stacks
        self.update_correction_stacks()
