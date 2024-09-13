import torch
import torch.nn as nn
from pet.src.pet import PET, cutoff_func
from pet.src.hypers import save_hypers, set_hypers_from_files, Hypers, hypers_to_dict


class PET_QM9(PET): # A wrapper for PET to use for QM9
    def __init__(self, hypers_dict, args):
        hypers = Hypers(hypers_dict) # hypers dict is contained in pet/default_hypers/default_hypers

        # maybe need to set some things in hypers based on args


        ##

        transformer_dropout = 0.0 # as in PET code
        n_atomic_species = None # max number of atoms in Q_9 (dunno)
        super().__init__(hypers, transformer_dropout, n_atomic_species) 

    def forward(self, t, xh, node_mask, edge_mask, context): # map the input of phi from EDM into sensible inputs of the PET
        batch_dict = self.get_batch_dict()
        return super().forward(batch_dict)
    
    def get_batch_dict(self, t, xh, node_mask, edge_mask, context):
        batch_dict = {}
        ## NEED THESE; REFER TO PET.get_predictions()

        # x = batch_dict["x"]
        # central_species = batch_dict["central_species"]
        # neighbor_species = batch_dict["neighbor_species"]
        batch_dict["batch"] = None # not needed
        batch_dict["mask"] = node_mask
        # nums = batch_dict["nums"]
        # neighbors_index = batch_dict["neighbors_index"]
        # neighbors_pos = batch_dict["neighbors_pos"]

        # need to also figure out what's going on with len(all_species) in PET

        return batch_dict
    
if __name__ == '__main__':
    print('hi')