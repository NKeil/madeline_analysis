import sys
sys.path.append("../U19_CADA_analysis/")
sys.path.append("../U19_CADA_analysis/packages/RR_bmat")

from utils import *
from behaviors import *
from packages.photometry_functions import get_zdFF
from peristimulus import *
from neuro_series import *
from neurobehavior_base import *
import numbers
import logging
import itertools
logging.basicConfig(level=logging.INFO)
from os.path import join as oj
# sns.set_context("talk")
# %matplotlib inline

class ScopeSeries:
    """
    Wrapper class for manipulating population imaging data.
    """
    io_specs = {'sig': 'neuron',
                'neuropil': 'pil'}
    params = {'fr': 10,
              'time_unit': 'ms',
              'control': '410nm',
              'ignore_channels': []}
    fp_flags = {}
    quality_metric = 'snr'

    def __init__(self, data_file, animal='test', session='0'):
        self.neural_dfs = {}
        self.neural_df = None
        self.all_channels = {}
        self.sig_channels = {}
        self.animal, self.session = animal, session
        # TODO: add method to label quality for all ROIs

    def estimate_fr(self, ts):
        # First convert all the time units to seconds
        if self.params['time_unit'] == 'ms':
            ts = ts / 1000

    def calculate_dff(self, method, zscore=True, **kwargs):
        # for all of the channels, calculate df using the method specified
        # TODO: add visualization technique by plotting the approximated
        # baseline against signal channels
        # Currently use original time stamps
        # method designed for lossy_ctrl merge channels method
        dff_name_map = {'iso_jove_dZF': 'jove'}
        iso_time = self.neural_df['time']
        dff_dfs = {'time': iso_time}
        for ch in self.sig_channels:
            for roi in self.sig_channels[ch]:
                rec_time = self.neural_df['time'].values
                rec_sig = self.neural_df[roi].values
                iso_sig = self.neural_df[roi.replace(ch, self.params['control'])].values
                if method == 'dZF_jove':
                    assert zscore, 'isosbestic jove is always zscored'
                    dff = get_zdFF(iso_sig, rec_sig, smooth_win=int(self.params['fr']), remove=0)
                elif method == 'dZF_jove_raw':
                    assert zscore, 'isosbestic jove is always zscored'
                    dff = get_zdFF(iso_sig, rec_sig, smooth_win=int(self.params['fr']), remove=0, use_raw=True)
                elif method == 'dZF_jove_old':
                    assert zscore, 'isosbestic jove is always zscored'
                    dff = get_zdFF_old(iso_sig, rec_sig, smooth_win=int(self.params['fr']), remove=0)
                elif method == 'ZdF_jove_old':
                    assert zscore, 'isosbestic jove is always zscored'
                    dff = get_zdFF_old(iso_sig, rec_sig,
                                       smooth_win=int(self.params['fr']), remove=0, raw=True)
                    dff = (dff - np.mean(dff)) / np.std(dff)
                else:
                    dff = raw_fluor_to_dff(rec_time, rec_sig, iso_time, iso_sig,
                                           baseline_method=method, zscore=zscore, **kwargs)
                    prefix = 'ZdFF_' if zscore else 'dFF_'
                    method = prefix + method
                meas_name = 'ZdFF' if zscore else 'dFF'
                dff_dfs[roi + '_ ' + meas_name] = dff
                dff_dfs['method'] = [method] * len(dff)

        dff_df = pd.DataFrame(dff_dfs)
        id_labls = ['time', 'method']
        meas = np.setdiff1d(dff_df.columns, id_labls)
        melted = pd.melt(dff_df, id_vars=id_labls, value_vars=meas, var_name='roi', value_name=meas_name)
        melted['roi'] = melted['roi'].str.replace('_ ' + meas_name, '')
        return melted

    def realign_time(self, reference=None):
        if isinstance(reference, BehaviorMat):
            transform_func = lambda ts: reference.align_ts2behavior(ts)
        if self.neural_df is not None:
            self.neural_df['time'] = transform_func(self.neural_df['time'])

    def diagnose_multi_channels(self, viz=True, plot_path=None):
        # Step 1: Visualize the Discontinuity in first 5 min
        # whole session visualization
        # Step 2:
        time_axis = self.neural_df['time']
        control_ch = self.params['control']
        sig_scores = {}
        fig_tag = f'{self.animal} {self.session}'
        for ch in self.sig_channels:
            for roi in self.sig_channels[ch]:
                raw_reference = self.neural_df[roi.replace(ch, control_ch)].values
                raw_signal = self.neural_df[roi].values

                fig, sig_score, _ = FP_quality_visualization(raw_reference, raw_signal, time_axis,
                                                             fr=self.params['fr'], drop_frame=200,
                                                             time_unit=self.params['time_unit'],
                                                             sig_channel=ch, control_channel=control_ch,
                                                             roi=roi, tag=fig_tag, viz=viz)
                if fig is not None and plot_path is not None:
                    animal_folder = oj(plot_path, self.animal)
                    animal, session = self.animal, self.session
                    if not os.path.exists(animal_folder):
                        os.makedirs(animal_folder)
                    fig.savefig(oj(animal_folder,
                                   f'{animal}_{session}_{roi}_quality_{self.quality_metric}.png'))
                sig_scores[roi] = sig_score
        return sig_scores


class Suite2pSeries:

    def __init__(self, suite2p):
        suite2p = os.path.join(suite2p, 'plane0')
        Fraw = np.load(oj(suite2p, 'F.npy'))
        ops = np.load(oj(suite2p, 'ops.npy'), allow_pickle=True)
        neuropil = np.load(oj(suite2p, 'Fneu.npy'))
        cells = np.load(oj(suite2p, 'iscell.npy'))
        stat = np.load(oj(suite2p, 'stat.npy'), allow_pickle=True)
        self.Fraw = Fraw
        self.ops = ops
        self.neuropil = neuropil
        self.cells = cells
        self.stat = stat

        F = Fraw - neuropil * 0.7  # subtract neuropil
        # find number of cells
        numcells = np.sum(cells[:, 0] == 1.0)
        # create a new array (Fcells) with F data for each cell
        Fcells = np.zeros((numcells, F.shape[1]))
        counter = 0
        for cell in range(0, len(cells)):
            if cells[cell, 0] == 1.0:  # if ROI is a cell
                Fcells[counter] = F[cell]
                counter += 1
        F0 = []
        for cell in range(0, Fcells.shape[0]):
            include_frames = []
            std = np.std(Fcells[cell])
            avg = np.mean(Fcells[cell])
            for frame in range(0, Fcells.shape[1]):
                if Fcells[cell, frame] < std + avg:
                    include_frames.append(Fcells[cell, frame])
            F0.append(np.mean(include_frames))
        dFF = np.zeros(Fcells.shape)
        for cell in range(0, Fcells.shape[0]):
            for frame in range(0, Fcells.shape[1]):
                dFF[cell, frame] = (Fcells[cell, frame] - F0[cell]) / F0[cell]
        self.neural_df = pd.DataFrame(data=dFF.T, columns=[f'neuron{i}' for i in range(numcells)])
        self.neural_df['time'] = np.arange(self.neural_df.shape[0])

    def realign_time(self, reference=None):
        if isinstance(reference, BehaviorMat):
            transform_func = lambda ts: reference.align_ts2behavior(ts)
        if self.neural_df is not None:
            self.neural_df['time'] = transform_func(self.neural_df['time'])

    def calculate_dff(self):
        rois = [c for c in self.neural_df.columns if c != 'time']
        melted = pd.melt(self.neural_df, id_vars='time', value_vars=rois, var_name='roi', value_name='ZdFF')
        return melted


class GNGTBehaviorMat(BehaviorMat):
    # Behavior Mat for Probswitch
    # Figure out how to make it general
    # code_map = {1: ('center_in', 'center_in'),
    #             11: ('center_in', 'initiate'),
    #             2: ('center_out', 'center_out'),
    #             3: ('side_in', 'left'),
    #             4: ('side_out', 'left'),
    #             44: ('side_out', 'left'),
    #             5: ('side_in', 'right'),
    #             6: ('side_out', 'right'),
    #             66: ('side_out', 'right'),
    #             71.1: ('outcome', 'correct_unrewarded'),
    #             71.2: ('outcome', 'correct_rewarded'),
    #             72: ('outcome', 'incorrect_unrewarded'),
    #             73: ('outcome', 'missed'),  # saliency questionable
    #             74: ('outcome', 'abort')}  # saliency questionable

    # divide things into events, event_features, trial_features
    fields = ['onset', 'first_lick_in', 'last_lick_out', 'water_valve_on', 'outcome']

    time_unit = 's'

    # event_features = 'reward', 'action',
    # trial_features = 'quality', 'struct_complex', 'explore_complex', 'BLKNo', 'CPort'
    # Always use efficient coding
    def __init__(self, animal, session, hfile, csvfile):
        super().__init__(animal, session)
        if isinstance(hfile, str):
            with h5py.File(hfile, 'r') as hf:
                frame_time = np.array(hf['out/frame_time']).ravel()
        else:
            frame_time = np.array(hfile['out/frame_time']).ravel()
        self.time_aligner = lambda t: frame_time
        self.df = pd.read_csv(csvfile)

    def __str__(self):
        return f"BehaviorMat({self.animal}_{self.session})"

    def todf(self):
        self.df['session'] = self.df['session'].astype(str)
        return self.df


class GoNoGo_NBMat(NeuroBehaviorMat):
    # Fill out the fields for your experiment
    fields = ['onset', 'first_lick_in',
              'last_lick_out', 'water_valve_on', 'outcome']
    # Add capacity for only behavior
    behavior_events = GNGTBehaviorMat.fields

    event_features = {'reward': ['water_valve_on', 'outcome'],
                      'licks_out': ['last_lick_out', 'outcome'],
                      'num_water_valve_on': ['water_valve_on_time', 'outcome_time']}

    trial_features = ['sound_num', 'go_nogo']

    id_vars = ['animal', 'session', 'roi']

    def __init__(self, fp_series=None, behavior_df=None):
        super().__init__()  # fp_series, behavior_df)
        self.event_time_windows = {'onset': np.arange(-1, 1.001, 0.05),
                                   'first_lick_in': np.arange(-1, 1.001, 0.05),
                                   'last_lick_out': np.arange(-1, 1.001, 0.05),
                                   'water_valve_on': np.arange(-1, 1.001, 0.05),
                                   'outcome': np.arange(-1, 1.001, 0.05)}


class GoNoGo_Expr(NBExperiment):
    # TODO: for decoding, add functions to merge multiple rois
    info_name = 'gn_neural_subset.csv'
    spec_name = 'gn_animal_specs.csv'

    def __init__(self, folder, **kwargs):
        super().__init__(folder)
        self.folder = folder
        pathlist = folder.split(os.sep)[:-1] + ['plots']
        self.plot_path = oj(os.sep, *pathlist)
        print(f'Changing plot_path as {self.plot_path}')
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
        for kw in kwargs:
            if hasattr(self, kw):
                setattr(self, kw, kwargs[kw])

        # info = pd.read_csv(os.path.join(folder, self.info_name))
        # spec = pd.read_csv(os.path.join(folder, self.spec_name))
        # self.meta = info.merge(spec, left_on='animal', right_on='alias', how='left')
        # # # self.meta.loc[self.meta['session_num']]
        # self.meta['cell_type'] = self.meta['animal_ID'].str.split('-', expand=True)[0]
        # self.meta['session'] = self.meta['age'].apply(self.cvt_age_to_session)
        self.nbm = GoNoGo_NBMat()

    # def cvt_age_to_session(self, age):
    #     DIG_LIMIT = 2  # limit the digits allowed for age representation (max 99)
    #     age = float(age)
    #     if np.allclose(age % 1, 0):
    #         return f'Day{int(age)}'
    #     else:
    #         digit = np.around(age % 1, DIG_LIMIT)
    #         agenum = int(age // 1)
    #         if np.allclose(digit, 0.05):
    #             return f'Day{agenum}_session0'
    #         else:
    #             snum = str(digit).split('.')[1]
    #             return f'Day{agenum}_session{snum}'

    def load_animal_session(self, animal_arg, session, options='all'):
        filemap = self.encode_to_filename(animal_arg, session, ['suite2p', 'caiman', 'trial', 'log'])
        bmat = GNGTBehaviorMat(animal_arg, session, filemap['log'], filemap['trial'])
        gn_series = Suite2pSeries(filemap['suite2p'])
        gn_series.realign_time(bmat)
        return bmat, gn_series

    def encode_to_filename(self, animal, session, ftypes="all"):
        """
        :param folder: str
                folder for data storage
        :param animal: str
                animal name: e.g. JUV001
        :param session: str
                session name: e.g. 211205
        :param ftype: list or str:
                list (or a single str) of typed files to return
                'suite2p': neural suite2p folder
                'caiman': hdf5 from caiman processed
                'log': behavior log csv from lunghao script
                'trial': trial structure
        :return:
                returns all 5 files in a dictionary; otherwise return all file types
                in a dictionary, None if not found
        """
        folder = self.folder
        p = os.path.join(folder, animal, session)
        results = {ft: None for ft in ftypes}
        registers = 0
        if ftypes == "all":
            ftypes = ['suite2p', "caiman", 'trial']
        elif isinstance(ftypes, str):
            ftypes = [ftypes]
        name_map = {'caiman': 'hdf5',
                    'trial': 'behavior_output',
                    'log': 'behaviorLOG'}

        if os.path.exists(p):
            for f in os.listdir(p):
                for ift in ftypes:
                    file_match = False
                    if (ift == 'suite2p'):
                        if (f == 'suite2p'):
                            file_match = True
                    else:
                        ift_arg = name_map[ift]
                        file_match = (ift_arg in f) and (animal in f) and (session in f)
                    if file_match:
                        results[ift] = os.path.join(p, f)
                        registers += 1
                        if registers == len(ftypes):
                            return results if len(results) > 1 else results[ift]
        return results if len(results) > 1 else list(results.values())[0]

if __name__ == "__main__":
    # fields = ['onset', 'first_lick_in',
    #           'last_lick_out', 'water_valve_on', 'outcome']
    data_root = r'\\filenest.diskstation.me\Wilbrecht_file_server\Madeline\processed_data'
    gse = GoNoGo_Expr(data_root)
    animal, session = 'JUV011', '211215'
    bmat, gn_series = gse.load_animal_session(animal, session)
    bdf, dff_df = bmat.todf(), gn_series.calculate_dff()

    # cue 2: first go cue, cue 7: nogo
    event = 'onset'
    neurons = [f'neuron{i}' for i in range(10, 20)]
    neur_dff_df = dff_df[dff_df['roi'].isin(neurons)].reset_index(drop=True)
    nb_df = gse.nbm.align_B2N_dff_ID(bdf, neur_dff_df, [event], form='wide')
    plot_df = gse.nbm.lag_wide_df(nb_df, {f'{event}_neur': {'long':True}})
    sns.relplot(data=plot_df, x=f'{event}_neur_time', y=f'{event}_neur_ZdFF', col='roi', kind='line', palette='Spectral')
