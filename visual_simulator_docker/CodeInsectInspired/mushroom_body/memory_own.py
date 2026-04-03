## Author : Antoine Wystrach
# modified by Dequan Ou

import numpy as np
import time

class MushroomBody(object):
    def __init__(self,
                 PN_nb=795,
                 KC_nb=5000,
                 MBON_nb=2,
                 KCtoPN_synapses=4,
                 KC_norm_param=0.01,
                 seed=None,
                 verbose=True):

        if KCtoPN_synapses > PN_nb:
            raise ValueError('KCtoPN_synapses must be <= PN_nb')

        start = time.time()

        self._pruned = False
        self._pruning_matrix = np.ones(PN_nb, dtype=bool)
        self._std_pns_prun = np.ones(PN_nb, dtype=float)

        self._seed = seed

        # Cell numbers
        self._PN_nb = PN_nb
        self._KC_nb = KC_nb
        self._MBON_nb = MBON_nb
        self._KCtoPN_synapses = KCtoPN_synapses
        self._normed_KC_nb = int(np.ceil(self._KC_nb * KC_norm_param))

        self.__init_connection_weights()
        self.__init_cell_activities()

        end = time.time()
        if verbose:
            print('\nMushroom Body initialized in {:.3f} secs.'.format(end-start))

    def __init_connection_weights(self):
        self._W_KCtoMBON = np.ones((self._KC_nb, self._MBON_nb), dtype='float32')
        self._W_PNtoKC = np.zeros((self.PN_nb, self._KC_nb), dtype='float32')
        self._W_PNtoKC[:self._KCtoPN_synapses, :] = 1

        np.random.seed(seed=self._seed)
        for col in np.arange(self._KC_nb):
            np.random.shuffle(self._W_PNtoKC[:, col])

        # Make a backup of the original random pattern
        self.__backup_W_PNtoKC = np.copy(self._W_PNtoKC)

    def __init_cell_activities(self):
        self._pn_before_feedback = np.zeros(self.PN_nb, dtype='float32')
        self._pn_activity = np.zeros(self.PN_nb, dtype='float32')
        self._pn_feedback = np.zeros(self.PN_nb, dtype='float32')
        self._kc_ePSP = np.zeros(self._KC_nb, dtype='float32')
        self._kc_spikes = np.zeros(self._KC_nb, dtype='float32')
        self._mbon_activity = np.zeros(self._MBON_nb, dtype='float32')

    @property
    def pn_activity_before_feedback(self):
        return self._pn_before_feedback

    @property
    def pn_activity(self):
        return self._pn_activity

    @property
    def pn_feedback(self):
        return self._pn_feedback

    @property
    def kc_ePSP(self):
        return self._kc_ePSP

    @property
    def kc_spikes(self):
        return self._kc_spikes.astype('int')

    @property
    def mbon_activity(self):
        return self._mbon_activity

    @property
    def PN_nb(self):
        return np.sum(self._pruning_matrix)

    @property
    def KC_nb(self):
        return self._KC_nb

    @property
    def KCtoPN_synapses(self):
        return self._KCtoPN_synapses

    @property
    def W_KCtoMBON(self):
        return self._W_KCtoMBON
    
    @property
    def normed_KC_nb(self):
        return self._normed_KC_nb

    @property
    def pruned(self):
        return self._pruned
    
    @property
    def MBON_nb(self):
        return self._MBON_nb

    def _update_pn(self):
        self._pn_activity = self._pn_before_feedback - self._pn_feedback

    def _update_pn_feedback(self):
        new_feedback = self._pn_feedback 
        self._pn_feedback = new_feedback

    def _update_kc_ePSP(self):
        self._kc_ePSP = np.matmul(self._pn_activity, self._W_PNtoKC)

    def _update_kc_spikes(self):
        self._kc_spikes.fill(0)
        idx_active = np.argpartition(self._kc_ePSP, -self._normed_KC_nb)[-self._normed_KC_nb:]
        self._kc_spikes[idx_active] = 1.0

    def _update_mbon_activity(self):
        self._mbon_activity = np.matmul(self._kc_spikes, self._W_KCtoMBON) / float(self._normed_KC_nb)

    def refresh(self, view=None):
        if view is None:
            self._pn_before_feedback.fill(0.0)
        else:
            view = view.squeeze().reshape(-1)[self._pruning_matrix]
            self._pn_before_feedback = np.asanyarray(view, dtype='float32')

        self._update_pn()
        self._update_pn_feedback()
        self._update_kc_ePSP()
        self._update_kc_spikes()
        self._update_mbon_activity()

    def reset(self, shuffle=False):
        self.__init_cell_activities()

        if shuffle:
            self.__init_connection_weights()
        else:
            # TODO: if shuffle is False, this may give an error? NUM_MBONS is not defined
            self._MBON_nb = NUM_MBONS
            self._W_KCtoMBON = np.ones((self._KC_nb, self._MBON_nb), dtype='float32')
            self._W_PNtoKC = np.copy(self.__backup_W_PNtoKC)

    def get_nonzero_KCtoMBON_weights(self, mbon_index=0):
        return np.count_nonzero(self._W_KCtoMBON[:, mbon_index])

    def learn(self, mbon_index=0):
        # Get the indices of the currently firing KCs (where _kc_spikes is 1)
        active_kc_indices = np.where(self._kc_spikes == 1)[0]
        
        # Set the weights for the active KCs to 0, ONLY for the specified MBON column (mbon_index)
        self._W_KCtoMBON[active_kc_indices, mbon_index] = 0

    def load_memory(self, W_KCtoMBON):
        self._W_KCtoMBON = np.copy(W_KCtoMBON)

    def get_unused_KC_ratio(self):
        return np.mean(self._W_KCtoMBON)

    def get_unfamiliarity(self, current_view):
        self.refresh(current_view)
        return self._mbon_activity

    def get_familiarity(self, current_view):
        return 1.0 - self.get_unfamiliarity(current_view)

    def __repr__(self):
        return '\n\nMushroom Body object.\n' \
               '-- Params: --\n' \
               'PN cells: {} ({} active)\n' \
               'Kenyon Cells (KC): {}\n' \
               'Synapses (per KC): {}\n' \
               'Normalized KC number: {}\n' \
               'Seed: {}\n' \
               'Mean KC to MBON synaptic weight: {}'.format(self._PN_nb,
                                                            self.PN_nb,
                                                            self._KC_nb,
                                                            self.MBON_nb,
                                                            self._KCtoPN_synapses,
                                                            self._normed_KC_nb,
                                                            self._seed,
                                                            np.mean(self._W_KCtoMBON)
                                                            )
    def __str__(self):
        return 'Mushroom Body'