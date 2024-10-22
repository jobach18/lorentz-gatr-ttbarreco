import h5py
import pickle
import numpy as np
from pepper.hdffile import HDF5File
import vector
import awkward as ak


new_data_dir = '/nfs/dust/cms/user/stafford/For_Emanuele/reconn/2007_data'


data_dir = '/gpfs/dust/cms/user/stafford/For_Emanuele/reconn/2007_data/_TTbarDMJets_Dilepton_scalar_LO_Mchi_1_Mphi_250_TuneCP5_13TeV_madgraph_mcatnlo_pythia8.h5'
with HDF5File(data_dir, "r") as f:
    df = f["data"]
print_data = False
if print_data:
    print('available fields')
    print(df.fields)
    print('for the gen tops this is:')
    print(df['gent'].fields)
    print('for the jets this is')
    print(df['Jet'].fields)
    print(f'and the jets look like {df["Jet"].type}')

def convert_to_cartesian(vec, i):
    print(f'vector is {vec.pt[0]}')
    #pad the vector
    paddedpt = ak.fill_none(ak.pad_none(vec.pt[0], i, axis=1), 0)
    paddedphi = ak.fill_none(ak.pad_none(vec.phi[0], i, axis=1), 0)
    paddedeta = ak.fill_none(ak.pad_none(vec.eta, i, axis=1), 0)
    paddedmass = ak.fill_none(ak.pad_none(vec.mass, i, axis=1), 0)
    print('----')
    print(paddedpt)
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
for i in range(2):
    i = i + 1 
    print(f' i is {i}')
    genjs = np.concatenate([genjs, convert_to_cartesian(df["Jet"], i)], axis=1)
#do this for top 2 b-jets
genbs = convert_to_cartesian(df['genb'], 0)
for i in range(1):
    i = i + 1
    genbs = np.concatenate([genbs, convert_to_cartesian(df["genb"], i)], axis=1)

print(genbs)
print(genbs.shape)

