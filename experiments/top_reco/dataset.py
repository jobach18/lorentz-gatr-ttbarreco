import torch
import numpy as np
from torch_geometric.data import Data

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

    def __init__(self, rescale_data):
        super().__init__()
        self.rescale_data = rescale_data

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

        # preprocessing

        kinematics = torch.tensor(kinematics, dtype=dtype)
        labels = torch.tensor(labels, dtype=dtype)

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(kinematics.shape[0]):
            # drop zero-padded components
            mask = (kinematics[i, ...].abs() > EPS).all(dim=-1)
            fourmomenta = kinematics[i, ...][mask]
            targets = targets[i, ...]
            scalars = torch.zeros(
                fourmomenta.shape[0],
                pdgids,
                dtype=dtype,
            )  # no scalar information
            data = Data(x=fourmomenta, scalars=scalars, targets=targets)
            self.data_list.append(data)

