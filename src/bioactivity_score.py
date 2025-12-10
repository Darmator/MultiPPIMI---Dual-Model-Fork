# Suppress warnings and RDKit logs for cleaner output
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="outdated")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import sys
import pandas as pd
import numpy as np

from torch_geometric.loader import DataLoader as GeometricDataloader
sys.path.insert(0, './src')
from datasets.PPIMI_datasets import CustomSmilesDataset
from compound_gnn_model import GNNComplete
from MultiPPIMI import DualMultiPPIMI

# Class for scoring bioactivity of small molecules against protein-protein interactions
class PPIScore():
    def __init__(self,
                 prot1 = "P62166", #NCS-1
                 prot2 = "Q9NPQ8", #Ric8
                 model_folder_location = "./final_models/", #Folder where the ensemble of models is located
                 n_models = 10
                 ): 
        """Initialize scorer with target protein pair and load ensemble of 10 models."""
        self.prot1 = prot1
        self.prot2 = prot2
        self.model_list = []
        
        # Load ensemble of 10 pre-trained dual-task models
        for i in range(n_models):
            # Initialize compound GNN encoder
            modulator_model = GNNComplete(5, 300, JK='last', drop_ratio=0, gnn_type="gin")
            modulator_model.load_state_dict(torch.load('./src/GraphMVP_C.model'))
            
            # Initialize dual-task PPIMI model
            PPIMI_model = DualMultiPPIMI(
                modulator_model,
                modulator_emb_dim=310, 
                ppi_emb_dim=1318, 
                device="cuda:0",
                h_dim=512, n_heads=2
                ).to("cuda:0")
            PPIMI_model.load_state_dict(torch.load(model_folder_location+"final_filter_"+str(i)+".model"))
            self.model_list.append(PPIMI_model)
            
    def multippimi_predicting(self, PPIMI_model, device, dataloader, regression=True):
        """Run inference on a batch of molecules using a single model."""
        PPIMI_model.eval()
        total_preds = []
        total_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                modulator, rdkit_descriptors, ppi_esm = batch
                modulator = modulator.to(device)
                rdkit_descriptors = rdkit_descriptors.to(device)
                ppi_esm = ppi_esm.to(device)
                
                # Get both regression and classification predictions
                pred_1, pred_2 = PPIMI_model(modulator, rdkit_descriptors, ppi_esm)
                
                # Select regression or classification output
                if(regression):
                    pred = pred_1.squeeze()
                else:
                    pred = pred_2.squeeze()
                    
                if pred.ndim == 1:
                    pred = pred.unsqueeze(0)
                total_preds += pred.detach().cpu().numpy().flatten().tolist()
        return np.array(total_preds)
    
    def score_predict(self, smiles, full_array = False):
        """Generate bioactivity scores for a list of SMILES strings using model ensemble."""
        # Prepare dataframe with SMILES and protein pair
        df = pd.DataFrame()
        df["SMILES"] = smiles
        df['uniprot_id1'] = [self.prot1] * len(smiles)
        df['uniprot_id2'] = [self.prot2] * len(smiles)

        # Create dataset and dataloader
        sample_dataset = CustomSmilesDataset(df, labels=False)
        sample_dataloader = GeometricDataloader(sample_dataset, batch_size=1024*3, shuffle=False, drop_last=False)
    
        # Collect predictions from all 10 models in ensemble
        score_list = []
        for PPIMI_model in self.model_list:
            score_list.append(self.multippimi_predicting(PPIMI_model, "cuda:0", sample_dataloader))
    
        # Return either full ensemble array or aggregated scores
        if(full_array):
            return score_list
        
        # Compute robust aggregate: median minus std deviation
        scores = np.median(np.array(score_list), axis = 0) - np.std(np.array(score_list), axis=0)
        return scores
        
    def __call__(self, smiles: str):
        """Convenience method for scoring single or multiple SMILES strings."""
        if isinstance(smiles, str):
            smiles = [smiles]
        
        rewards = self.score_predict(smiles)
        return rewards