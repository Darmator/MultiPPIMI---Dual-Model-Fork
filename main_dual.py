import argparse
import copy
import sys
import time
import random
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

sys.path.insert(0, './src')
from datasets.PPIMI_datasets import CustomModulatorPPIDataset, regression_evaluation, ModulatorPPIDataset, custom_performance_evaluation
from compound_gnn_model import GNNComplete
from MultiPPIMI import DualMultiPPIMI


def train(PPIMI_model, device, dataloader, optimizer, regression):
    PPIMI_model.train()
    loss_accum = 0
    for step_idx, batch in enumerate(dataloader):
        modulator, rdkit_descriptors, ppi_esm, label = batch
        modulator = modulator.to(device)
        rdkit_descriptors = rdkit_descriptors.to(device)
        ppi_esm = ppi_esm.to(device)
        label = label.to(device)
        if(regression):
            pred = PPIMI_model(modulator, rdkit_descriptors, ppi_esm, regression=True, classification = False).squeeze()
            optimizer.zero_grad()
            loss = criterion_reg(pred, label.to(dtype=torch.float32))
        else:
            pred = PPIMI_model(modulator, rdkit_descriptors, ppi_esm, regression=False, classification = True).squeeze()
            optimizer.zero_grad()
            loss = criterion_class(pred, label.to(dtype=torch.float32))
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().item()
    print('Loss:\t{}'.format(loss_accum / len(dataloader)))
    return loss_accum / len(dataloader)



def predicting(PPIMI_model, device, dataloader, regression):
    PPIMI_model.eval()
    total_preds = []
    total_labels = []
    with torch.no_grad():
        for batch in dataloader:
            modulator, rdkit_descriptors, ppi_esm, label = batch
            modulator = modulator.to(device)
            rdkit_descriptors = rdkit_descriptors.to(device)
            ppi_esm = ppi_esm.to(device)
            label = label.to(device)
            pred_1, pred_2 = PPIMI_model(modulator, rdkit_descriptors, ppi_esm)
            if(regression):
                pred = pred_1.squeeze()
            else:
                pred = pred_2.squeeze()
            if pred.ndim == 1:
                pred = pred.unsqueeze(0)
            total_preds += pred.detach().cpu().numpy().flatten().tolist()
            total_labels += label.detach().cpu().numpy().flatten().tolist()
            #total_preds.append(pred.detach().cpu().numpy().flatten())
            #total_labels.append(label.detach().cpu().numpy().flatten())

    #total_preds = torch.cat(total_preds, dim=0)
    #total_labels = torch.cat(total_labels, dim=0)
    return np.array(total_labels), np.array(total_preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of MultiPPIMI')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eval_setting', type=str, default='S1', choices=['S1', 'S2', 'S3', 'S4'])
    parser.add_argument('--fold', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runseed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--pretrained_model_file', type=str, default='./src/GraphMVP_C.model')
    parser.add_argument('--output_model_file', type=str, default='')
    parser.add_argument('--out_path', type=str, default='.')
    ########## For compound embedding ##########
    parser.add_argument('--num_layer', type=int, default=5)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--dropout_ratio', type=float, default=0.)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument('--gnn_type', type=str, default='gin')
    ########## For protein embedding ##########
    parser.add_argument('--ppi_hidden_dim', type=int, default=1318)
    ########## For attention module ##########
    parser.add_argument('--h_dim', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=2)
    args = parser.parse_args()

    ### set random seeds
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device(f'cuda:{args.device}')
    print(device)

    ########## Saving data ##########
    saving_dict = {"epoch": [], "class_loss": [], "reg_loss": [], "auc": [], "acc": [], "f1" : [], "mae": [], "mse" : [], "r2" : []}
    ########## Set up dataset and dataloader ##########
    train_reg_dataset = CustomModulatorPPIDataset(datapath= "data/reg_data/ppi_train_multinorm.csv", labels=True)   
    test_reg_dataset = CustomModulatorPPIDataset(datapath= "data/reg_data/ppi_test_multinorm.csv", labels=True)
    #train_reg_dataset = CustomModulatorPPIDataset(datapath= "data/demo_data/ppi_chembl_curated_train.csv", labels=True)   
    #test_reg_dataset = CustomModulatorPPIDataset(datapath= "data/demo_data/ppi_chembl_curated_train.csv", labels=True)   
    print('size of Regression train: {}\ttest: {}'.format(len(train_reg_dataset), len(test_reg_dataset)))


    train_reg_dataloader = DataLoader(train_reg_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    #valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_reg_dataloader = DataLoader(test_reg_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    ##################Classification########################
    train_class_dataset = ModulatorPPIDataset(mode='train', setting=args.eval_setting, fold=args.fold)    
    valid_class_dataset = ModulatorPPIDataset(mode='valid', setting=args.eval_setting, fold=args.fold)
    test_class_dataset = ModulatorPPIDataset(mode='test', setting=args.eval_setting, fold=args.fold)
    print('size of Classification train: {}\tval: {}\ttest: {}'.format(len(train_class_dataset), len(valid_class_dataset), len(test_class_dataset)))

    train_class_dataloader = DataLoader(train_class_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_class_dataloader = DataLoader(valid_class_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_class_dataloader = DataLoader(test_class_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    ########## Set up model ##########
    modulator_model = GNNComplete(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    if not args.pretrained_model_file == '':
        print('========= Loading from {}'.format(args.pretrained_model_file))
        modulator_model.load_state_dict(torch.load(args.pretrained_model_file))
    PPIMI_model = DualMultiPPIMI(
        modulator_model,
        modulator_emb_dim=310, 
        ppi_emb_dim=args.ppi_hidden_dim, 
        device=device,
        h_dim=args.h_dim, n_heads=args.n_heads
        ).to(device)

    print('MultiPPIMI model\n', PPIMI_model)
    #PPIMI_model.load_state_dict(torch.load("dual_regression_ic50.model"))
    criterion_reg = nn.MSELoss()
    criterion_class = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(PPIMI_model.parameters(), lr=args.learning_rate)

    best_model = None
    best_epoch = 0
    best_mse = 1000

    train_start_time = time.time()
    for epoch in range(1, 1+args.epochs):
        
        start_time = time.time()
        print('Start training at epoch: {}'.format(epoch))
        class_loss = train(PPIMI_model, device, train_class_dataloader, optimizer, regression=False)
        reg_loss = train(PPIMI_model, device, train_reg_dataloader, optimizer, regression=True)

        G, P = predicting(PPIMI_model, device, valid_class_dataloader, regression=False)

        current_roc_auc, current_aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels = custom_performance_evaluation(P, G)
        print('Val AUC:\t{}'.format(current_roc_auc))
        print('Val AUPR:\t{}'.format(current_aupr))
        print('Val Acc:\t{}'.format(accuracy))

        

        G, P = predicting(PPIMI_model, device, test_reg_dataloader, regression=True)

        current_mae, current_mse, current_r2 = regression_evaluation(P, G)
        print('Val MAE:\t{}'.format(current_mae))
        print('Val MSE:\t{}'.format(current_mse))
        print('Val R2:\t{}'.format(current_r2))
        if current_mse < best_mse:
            best_model = copy.deepcopy(PPIMI_model)
            best_mse = current_mse
            best_epoch = epoch
            print('MSE improved at epoch {}\tbest MSE: {}'.format(best_epoch, best_mse))
        else:
            print('No improvement since epoch {}\tbest MSE: {}'.format(best_epoch, best_mse))
        print('Took {:.5f}s.'.format(time.time() - start_time))
        print()

        #saving score
        saving_dict["epoch"].append(epoch)
        saving_dict["class_loss"].append(class_loss)
        saving_dict['auc'].append(current_roc_auc)
        saving_dict["acc"].append(accuracy)
        saving_dict["f1"].append(f1)
        saving_dict["reg_loss"].append(reg_loss)
        saving_dict["mae"].append(current_mae)
        saving_dict["mse"].append(current_mse)
        saving_dict["r2"].append(current_r2)



    print('Finish training!')
    print('Total training time: {:.5f} hours'.format((time.time()-train_start_time)/3600))
    start_time = time.time()
    print('Last epoch test results: {}'.format(args.epochs))

    G, P = predicting(PPIMI_model, device, test_class_dataloader, regression=False)
    roc_auc, aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels = custom_performance_evaluation(P, G)
    print('AUC:\t{}'.format(roc_auc))
    print('AUPR:\t{}'.format(aupr))
    print('precision:\t{}'.format(precision))
    print('accuracy:\t{}'.format(accuracy))
    print('recall:\t{}'.format(recall))
    print('f1:\t{}'.format(f1))
    print('specificity:\t{}'.format(specificity))
    print('mcc:\t{}'.format(mcc))


    G, P = predicting(PPIMI_model, device, test_reg_dataloader, regression=True)
    current_mae, current_mse, current_r2 = regression_evaluation(P, G)
    print('Test MAE:\t{}'.format(current_mae))
    print('Test MSE:\t{}'.format(current_mse))
    print('Test R2:\t{}'.format(current_r2))
    print('')
    print('Took {:.5f}s.'.format(time.time() - start_time))

    start_time = time.time()
    print('Best epoch test results: {}'.format(best_epoch))
    G, P = predicting(best_model, device, test_class_dataloader, regression=False)
    roc_auc, aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels = custom_performance_evaluation(P, G)
    print('AUC:\t{}'.format(roc_auc))
    print('AUPR:\t{}'.format(aupr))
    print('precision:\t{}'.format(precision))
    print('accuracy:\t{}'.format(accuracy))
    print('recall:\t{}'.format(recall))
    print('f1:\t{}'.format(f1))
    print('specificity:\t{}'.format(specificity))
    print('mcc:\t{}'.format(mcc))


    G, P = predicting(best_model, device, test_reg_dataloader, regression=True)
    current_mae, current_mse, current_r2 = regression_evaluation(P, G)
    print('Test MAE:\t{}'.format(current_mae))
    print('Test MSE:\t{}'.format(current_mse))
    print('Test R2:\t{}'.format(current_r2))
    print('')
    print('Took {:.5f}s.'.format(time.time() - start_time))

    # save best model
    model_name = "dual_regression_ppi_multinorm"
    model_path = model_name + '.model'
    torch.save(best_model.state_dict(), model_path)

    df = pd.DataFrame(saving_dict)
    df.to_csv(model_name + ".csv", index = False)
