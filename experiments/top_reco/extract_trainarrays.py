import h5py
import pickle
import numpy as np
from pepper.hdffile import HDF5File
import vector
import awkward as ak


new_data_dir = '/data/dust/user/stafford/For_Emanuele/reconn/2007_data'


#data_dir = '/gpfs/dust/cms/user/stafford/For_Emanuele/reconn/2007_data/_TTbarDMJets_Dilepton_scalar_LO_Mchi_1_Mphi_250_TuneCP5_13TeV_madgraph_mcatnlo_pythia8.h5'
#data_dir = '/gpfs/dust/cms/user/stafford/For_Emanuele/reconn/2907_nn_inputs/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/traindata.hdf5'
#data_dir = '/data/dust/user/stafford/For_Emanuele/reconn/2907_nn_inputs/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/validatedata.hdf5'
data_dir = '/data/dust/user/stafford/For_Emanuele/reconn/2907_nn_inputs/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/traindata.hdf5'
with HDF5File(data_dir, "r") as f:
    print('available fields in the file:')
    print(f.keys())
    df = f["data"]
    offset = f['offset']
    scale = f['scale']
print_data = True
print(f'the scale is {scale}')
print(f'and the offset is {offset}')
if print_data:
    print('available fields')
    print(df.fields)
    #print('for the gen tops this is:')
    #print(df['gent'].fields)
    print('for the jets this is')
    print(df['jet_pt'].fields)
    print(f'and the jets look like {df["jet_pt"].type}')
    print(f'the mtt is {df["mtt"].fields}')
    print(f'the max mtt is {np.max(df["mtt"])}')
    print(f'the top mass is {np.mean(df["atop_mass"])}')
    print(f'the weights are in mean {np.mean(df["weight"])}')


jet_t = df["jet_t"][:, :7]*scale['jet_t'] + offset['jet_t']
jet_x = df["jet_x"][:, :7]*scale['jet_x'] + offset['jet_x']
jet_y = df["jet_y"][:, :7]*scale['jet_y'] + offset['jet_y']
jet_z = df["jet_z"][:, :7]*scale['jet_z'] + offset['jet_z']
print(f' the maximum jet component is {np.max(jet_z)}')

jets = ak.fill_none(ak.pad_none(ak.concatenate([jet_t[:,:, np.newaxis], jet_x[:,:, np.newaxis], jet_y[:,:, np.newaxis], jet_z[:,:, np.newaxis]], axis=2), 7, axis=1), 0)
leptons = ak.concatenate([df["lep_t"][:, np.newaxis, np.newaxis]*scale['lep_t'] + offset['lep_t'], df["lep_x"][:, np.newaxis, np.newaxis]*scale['lep_x'] + offset['lep_x'], df["lep_y"][:, np.newaxis,  np.newaxis]*scale['lep_y'] + offset['lep_y'], df["lep_z"][:, np.newaxis, np.newaxis]*scale['lep_z'] + offset['lep_z']], axis=2)
antileptons = ak.concatenate([df["alep_t"][:, np.newaxis, np.newaxis]*scale['alep_t'] + offset['alep_t'], df["alep_x"][:, np.newaxis, np.newaxis]*scale['alep_x'] + offset['alep_x'], df["alep_y"][:, np.newaxis, np.newaxis]*scale['alep_y'] + offset['alep_y'], df["alep_z"][:, np.newaxis, np.newaxis]*scale['alep_z'] + offset['alep_z']], axis=2)
bottoms = ak.concatenate([df["genbot_t"][:, np.newaxis, np.newaxis]*scale['genbot_t'] + offset['genbot_t'], df["genbot_x"][:, np.newaxis, np.newaxis]*scale['genbot_x'] + offset['genbot_x'], df["genbot_y"][:, np.newaxis, np.newaxis]*scale['genbot_y'] + offset['genbot_y'], df["genbot_z"][:, np.newaxis, np.newaxis]*scale['genbot_z'] + offset['genbot_z']], axis=2)
antibottoms = ak.concatenate([df["genabot_t"][:, np.newaxis, np.newaxis]*scale['genabot_t'] + offset['genabot_t'], df["genabot_x"][:, np.newaxis, np.newaxis]*scale['genabot_x'] + offset['genabot_x'], df["genabot_y"][:, np.newaxis, np.newaxis]*scale['genabot_y'] + offset['genabot_y'], df["genabot_z"][:, np.newaxis, np.newaxis]*scale['genabot_z'] + offset['genabot_z']], axis=2)
met = ak.concatenate([df["met_pt"][:, np.newaxis, np.newaxis]*scale['met_pt'] + offset['met_pt'], df["met_x"][:, np.newaxis, np.newaxis]*scale['met_x'] + offset['met_x'], df["met_y"][:, np.newaxis, np.newaxis]*scale['met_y'] + offset['met_y'], df["met_phi"][:, np.newaxis, np.newaxis]*scale['met_phi'] + offset['met_phi']], axis=2)

top = ak.concatenate([df["top_t"][:, np.newaxis, np.newaxis]*scale['top_t'] + offset['top_t'], df["top_x"][:, np.newaxis, np.newaxis]*scale['top_x'] + offset['top_x'], df["top_y"][:, np.newaxis, np.newaxis]*scale['top_y'] + offset['top_y'], df["top_z"][:, np.newaxis, np.newaxis]*scale['top_z'] + offset['top_z']], axis=2)
antitop = ak.concatenate([df["atop_t"][:, np.newaxis, np.newaxis]*scale['atop_t'] + offset['atop_t'], df["atop_x"][:, np.newaxis, np.newaxis]*scale['atop_x'] + offset['atop_x'], df["atop_y"][:, np.newaxis, np.newaxis]*scale['atop_y'] + offset['atop_y'], df["atop_z"][:, np.newaxis, np.newaxis]*scale['atop_z'] + offset['atop_z']], axis=2)

print(f'the maximum top component is {np.max(top)}')

chel = df["chel"]*scale['chel'] + offset['chel']
mtt = df["mtt"] * scale['mtt'] + offset['mtt']

target_scalar = ak.to_numpy(ak.concatenate([chel[:, np.newaxis], mtt[:, np.newaxis]], axis=1), allow_missing=True)

target = ak.concatenate([top, antitop], axis=1)
print(target.type)
target = ak.to_numpy(ak.concatenate([top, antitop], axis=1), allow_missing=True)
inputarr = ak.to_numpy(ak.concatenate([jets, leptons, antileptons, bottoms, antibottoms, met], axis=1), allow_missing=True)
print(target.shape)

np.savez('data/train_TTTo2L2Nu_train_scaled_genbot.npz', x=inputarr, y=target, scalars=target_scalar)
#np.savez('data/train_TTTo2L2Nu_val_scaled_genbot.npz', x=inputarr, y=target, scalars=target_scalar)

