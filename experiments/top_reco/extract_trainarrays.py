import h5py
import pickle
import numpy as np
from pepper.hdffile import HDF5File
import vector
import awkward as ak


new_data_dir = '/nfs/dust/cms/user/stafford/For_Emanuele/reconn/2007_data'


#data_dir = '/gpfs/dust/cms/user/stafford/For_Emanuele/reconn/2007_data/_TTbarDMJets_Dilepton_scalar_LO_Mchi_1_Mphi_250_TuneCP5_13TeV_madgraph_mcatnlo_pythia8.h5'
#data_dir = '/gpfs/dust/cms/user/stafford/For_Emanuele/reconn/2907_nn_inputs/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/traindata.hdf5'
data_dir = '/gpfs/dust/cms/user/stafford/For_Emanuele/reconn/2907_nn_inputs/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/validatedata.hdf5'
with HDF5File(data_dir, "r") as f:
    df = f["data"]
print_data = False
if print_data:
    print('available fields')
    print(df.fields)
    #print('for the gen tops this is:')
    #print(df['gent'].fields)
    print('for the jets this is')
    print(df['jet_pt'].fields)
    print(f'and the jets look like {df["jet_pt"].type}')

jet_t = df["jet_t"][:, :7]
jet_x = df["jet_x"][:, :7]
jet_y = df["jet_y"][:, :7]
jet_z = df["jet_z"][:, :7]

jets = ak.fill_none(ak.pad_none(ak.concatenate([jet_t[:,:, np.newaxis], jet_x[:,:, np.newaxis], jet_y[:,:, np.newaxis], jet_z[:,:, np.newaxis]], axis=2), 7, axis=1), 0)
leptons = ak.concatenate([df["lep_t"][:, np.newaxis, np.newaxis], df["lep_x"][:, np.newaxis, np.newaxis], df["lep_y"][:, np.newaxis,  np.newaxis], df["lep_z"][:, np.newaxis, np.newaxis]], axis=2)
antileptons = ak.concatenate([df["alep_t"][:, np.newaxis, np.newaxis], df["alep_x"][:, np.newaxis, np.newaxis], df["alep_y"][:, np.newaxis, np.newaxis], df["alep_z"][:, np.newaxis, np.newaxis]], axis=2)
bottoms = ak.concatenate([df["bot_t"][:, np.newaxis, np.newaxis], df["bot_x"][:, np.newaxis, np.newaxis], df["bot_y"][:, np.newaxis, np.newaxis], df["bot_z"][:, np.newaxis, np.newaxis]], axis=2)
antibottoms = ak.concatenate([df["abot_t"][:, np.newaxis, np.newaxis], df["abot_x"][:, np.newaxis, np.newaxis], df["abot_y"][:, np.newaxis, np.newaxis], df["abot_z"][:, np.newaxis, np.newaxis]], axis=2)
met = ak.concatenate([df["met_pt"][:, np.newaxis, np.newaxis], df["met_x"][:, np.newaxis, np.newaxis], df["met_y"][:, np.newaxis, np.newaxis], df["met_phi"][:, np.newaxis, np.newaxis]], axis=2)

top = ak.concatenate([df["top_t"][:, np.newaxis, np.newaxis], df["top_x"][:, np.newaxis, np.newaxis], df["top_y"][:, np.newaxis, np.newaxis], df["top_z"][:, np.newaxis, np.newaxis]], axis=2)
antitop = ak.concatenate([df["atop_t"][:, np.newaxis, np.newaxis], df["atop_x"][:, np.newaxis, np.newaxis], df["atop_y"][:, np.newaxis, np.newaxis], df["atop_z"][:, np.newaxis, np.newaxis]], axis=2)


target = ak.concatenate([top, antitop], axis=1)
print(target.type)
target = ak.to_numpy(ak.concatenate([top, antitop], axis=1), allow_missing=True)
inputarr = ak.to_numpy(ak.concatenate([jets, leptons, antileptons, bottoms, antibottoms], axis=1), allow_missing=True)

np.savez('data/train_TTTo2L2Nu_val.npz', x=inputarr, y=target)

