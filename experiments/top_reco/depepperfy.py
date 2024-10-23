import h5py
import pickle
import numpy as np
from pepper.hdffile import HDF5File
import vector
import awkward as ak


new_data_dir = '/nfs/dust/cms/user/stafford/For_Emanuele/reconn/2007_data'


#data_dir = '/gpfs/dust/cms/user/stafford/For_Emanuele/reconn/2007_data/_TTbarDMJets_Dilepton_scalar_LO_Mchi_1_Mphi_250_TuneCP5_13TeV_madgraph_mcatnlo_pythia8.h5'
data_dir = '/gpfs/dust/cms/user/stafford/For_Emanuele/reconn/2907_nn_inputs/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/traindata.hdf5'
with HDF5File(data_dir, "r") as f:
    df = f["data"]
print_data = True
if print_data:
    print('available fields')
    print(df.fields)
    #print('for the gen tops this is:')
    #print(df['gent'].fields)
    print('for the jets this is')
    print(df['jet_pt'].fields)
    print(f'and the jets look like {df["jet_pt"].type}')
    exit()

def convert_to_cartesian(vec, i):
    #print(f'vector is {vec.pt.type}')
    #pad the vector
    paddedpt = ak.fill_none(ak.pad_none(vec.pt, i, axis=1), 0)
    paddedphi = ak.fill_none(ak.pad_none(vec.phi, i, axis=1), 0)
    paddedeta = ak.fill_none(ak.pad_none(vec.eta, i, axis=1), 0)
    paddedmass = ak.fill_none(ak.pad_none(vec.mass, i, axis=1), 0)
    #print('----')
    #print(paddedpt.type)
    ret = vector.array({
       'pt': paddedpt[:,i],
       'phi': paddedphi[:,i],
       'eta': paddedeta[:,i],
       'mass': paddedmass[:,i]
       })
    x = ret.px
    y = ret.py
    z = ret.pz
    t = ret.t
    return np.stack([x, y, z, t], axis=1)


topl = convert_to_cartesian(df['gent'], 0)
topsl = convert_to_cartesian(df['gent'], 1)

#do this for top 5 leptons 
genls = convert_to_cartesian(df['genlepton'], 0)
for i in range(1):
    i = i+1
    genls = np.concatenate([genls,convert_to_cartesian(df["genlepton"], i)], axis=1)

#do this for top 5 jets
genjs = convert_to_cartesian(df['Jet'], 0)
for i in range(1):
    i = i + 1 
    print(f' i is {i}')
    genjs = np.concatenate([genjs, convert_to_cartesian(df["Jet"], i)], axis=1)
#do this for top 2 b-jets
genbs = convert_to_cartesian(df['genb'], 0)
for i in range(0):
    i = i + 1
    genbs = np.concatenate([genbs, convert_to_cartesian(df["genb"], i)], axis=1)

out_array = np.concatenate([genbs, genjs, genls, topl, topsl], axis=1)
print(f'finished creating the out array')
print(f'it has shape {out_array.shape}')
