import torch
import numpy as np
from torch_geometric.data import Data
from experiments.logger import LOGGER
import os, json

EPS = 1e-5


class RecoDataset(torch.utils.data.Dataset):
    """
    We use torch_geometric to handle point cloud of jet constituents more efficiently
    The torch_geometric dataloader concatenates jets along their constituent direction,
    effectively combining the constituent index with the batch index in a single dimension.
    An extra object batch.batch for each batch specifies to which jet the constituent belongs.
    We extend the constituent list by a global token that is used to embed extra global
    information and extract the classifier score.

    Structure of the elements in self.data_list
    x : torch.tensor of shape (num_elements, 4)
        List of 4-momenta of jet constituents
    scalars : empty placeholder
    target : torch.tensor of shape (8), dtype torch.float
        target 4 vectors for reconstructed tops 
    is_global : torch.tensor of shape (num_elements), dtype torch.bool
        True for the global token (first element in constituent list), False otherwise
        We set is_global=None if no global token is used
    """

    def __init__(self):
        super().__init__()

    def load_data(self, filename, mode, data_scale):
        raise NotImplementedError

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class TopRecoDataset(RecoDataset):
    def load_data(
        self,
        filename,
        data_scale=None,
        scalar_target = False,
        dtype=torch.float32,
    ):
        """
        Parameters
        ----------
        filename : str
            Path to file in npz format where the dataset in stored
        mode : {"train", "test", "val"}
            Purpose of the dataset
            Train, test and validation datasets are already seperated in the specified file
        data_scale : float
            std() of all entries in the train dataset
            Effectively a change of units to make the network entries O(1)
        """
        data = np.load(filename)
        kinematics = data["x"]
        targets = data["y"]
        if scalar_target:
            sc_targets = data["scalars"]

        # preprocessing

        kinematics = torch.tensor(kinematics, dtype=dtype)
        targets = torch.tensor(targets, dtype=dtype)

        if data_scale=='lorentz_norm':
            #print(f'the kinematics shape before normalization is {kinematics.shape}')
            #print(f'the target shape before normalization is {targets.shape}')
            print(f' the scaling gave us: {torch.max(kinematics)} to {torch.min(kinematics)} kin. range')
            scale = 1.0 / (torch.sqrt(torch.abs(kinematics[:, :, 0]**2 - torch.sum(kinematics[:, :, 1:]**2, dim=-1)))+1e-8)
            scale = scale.unsqueeze(-1)  
            kinematics = kinematics * scale 
            scale = 1.0 / (torch.abs(torch.sqrt(targets[:, :, 0]**2 - torch.sum(targets[:, :, 1:]**2, dim=-1)))+1e-8)
            scale = scale.unsqueeze(-1)  
            #print(f'the scale shape is{scale.shape}')
            targets = targets * scale
            print(f' the scaling gave us: {torch.max(targets)} to {torch.min(targets)} range')
            print(f' the scaling gave us: {torch.max(kinematics)} to {torch.min(kinematics)} kin. range')
            #print(f'the kinematics shape after normalization is {kinematics.shape}')
            #print(f'the target shape after normalization is {targets.shape}')
        if data_scale=='std':
            scale = torch.std(kinematics)
            kinematics = kinematics / scale
            targets = targets / scale
        # scaling by 1 / (1/4 * minkNorm)
        mode = "minkNorm"
        def minkowski_norm(vectors):
            E = vectors[..., 0]
            p = vectors[..., 1:]
            norm = torch.sqrt(torch.clamp(E**2 - p.pow(2).sum(dim=-1), min=EPS))
            return norm.unsqueeze(-1)
        
        kin_norms = 0.25 * minkowski_norm(kinematics)
        kinematics = kinematics / kin_norms
        tar_norms = 0.25 * minkowski_norm(targets)
        targets = targets / tar_norms
        if scalar_target:
            sc_targets = torch.tensor(sc_targets, dtype=dtype)
            sc_tar_norms = torch.abs(sc_targets).max(dim=0, keepdim=True).values
            sc_targets = sc_targets / sc_tar_norms
        # store scaling factors
        if scalar_target:
            norms_dict = {
                "kin_norms": kin_norms.tolist(),
                "tar_norms": tar_norms.tolist(),
                "sc_tar_norms": sc_tar_norms.tolist()
            }
        else:
            norms_dict = {
                "kin_norms": kin_norms.tolist(),
                "tar_norms": tar_norms.tolist()
            }

        if "val_scaled" in filename:
            tag="val"
            #LOGGER.info(f"Storing input scaling factors computed on {tag} data via {mode}.")
            LOGGER.info(f"Storing scaling_factors_lambda4.json")
            os.makedirs("results_to_notebook/lr_1e-4", exist_ok=True)
            # json_path = os.path.join("results_to_notebook", f"scaling_factors_{tag}_{mode}.json")
            json_path = os.path.join("results_to_notebook/lr_1e-4", f"scaling_factors_lambda4.json")
            with open(json_path, "w") as json_file:
                json.dump(norms_dict, json_file, default=str)

        # create list of torch_geometric.data.Data objects
        # drop zero-padded components
        self.data_list = []
        metpt_as_scalar = False
        if metpt_as_scalar:
            metpt = kinematics[:,11,0]
            kinematics = kinematics[:,:-1,:]
        for i in range(kinematics.shape[0]):
            # drop zero-padded components
            fourmomenta = kinematics[i, ...]
            if i ==1:
                print('-'*30)
                print('at dataset creation stage:')
                print(f'the fourmomenta are {fourmomenta.shape}')
                print('-'*30)
            targets_i = targets[i, ...].flatten()
        
            if metpt_as_scalar:
                scalars = torch.zeros(
                    fourmomenta.shape[0],
                    0,
                    dtype=dtype,
                ) + metpt[i] 
                print(f'the value of metpt is {metpt[i]}')
                print(scalars.shape)
                print('and should be')
                print(torch.zeros(
                    fourmomenta.shape[0],
                    0,
                    dtype=dtype,
                ).shape)
                assert(scalars.shape == torch.zeros(
                    fourmomenta.shape[0],
                    0,
                    dtype=dtype,
                ).shape)  # no scalar information

            else:
                scalars = torch.zeros(
                    fourmomenta.shape[0],
                    0,
                    dtype=dtype,
                )  # no scalar information
            data = Data(x=fourmomenta, scalars=scalars, targets=targets_i, targets_sc=sc_targets[i,...].flatten() if scalar_target else None)
            self.data_list.append(data)
