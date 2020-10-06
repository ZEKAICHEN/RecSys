import torch
import tqdm
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
from torchfm.dataset.frappe import FrappeDataset
# from torchfm.dataset.rapid import RapidAdvanceDataset
from torchfm.model.afi import AutomaticFeatureInteractionModel
from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.dfm import DeepFactorizationMachineModel
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
from torchfm.model.hofm import HighOrderFactorizationMachineModel
from torchfm.model.lr import LogisticRegressionModel
from torchfm.model.ncf import NeuralCollaborativeFiltering
from torchfm.model.nfm import NeuralFactorizationMachineModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.wd import WideAndDeepModel
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
from torchfm.model.afn import AdaptiveFactorizationNetwork
from torchfm.model.mhafm import MultiheadAttentionalFactorizationMachineModel
from torchfm.model.dcan import DeepCrossAttentionalNetworkModel
from torchfm.model.dcap import DeepCrossAttentionalProductNetwork

import numpy as np


def get_dataset(name, path):
    if name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'movielens20M':
        return MovieLens20MDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    elif name == 'frappe':
        return FrappeDataset(path)
    # elif name == 'rapid':
    #     return RapidAdvanceDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=16)
    elif name == 'hofm':
        return HighOrderFactorizationMachineModel(field_dims, order=3, embed_dim=16)
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=16)
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(100, 100), dropout=0.5)
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(100, 100), dropout=0.5)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(100, 100), method='inner', dropout=0.5)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(100, 100), method='outer', dropout=0.5)
    elif name == 'dcap':
        return DeepCrossAttentionalProductNetwork(field_dims, embed_dim=16, num_heads=4, num_layers=2, mlp_dims=(100, 100), dropout=0.5)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=2, mlp_dims=(100, 100), dropout=0.5)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=16, mlp_dims=(100, 100), dropouts=(0.5, 0.5))
    elif name == 'ncf':
        # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
        assert isinstance(dataset, MovieLens20MDataset) or isinstance(dataset, MovieLens1MDataset)
        return NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2,
                                            user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx)
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim=4, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims, embed_dim=16, mlp_dims=(100, 100), dropout=0.5)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(100, 100), dropout=0.5)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=16, attn_size=64, dropouts=(0.5, 0.5))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
             field_dims, embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=2, mlp_dims=(100, 100), dropouts=(0.5, 0.5, 0.5))
    elif name == 'afn':
        print("Model:AFN")
        return AdaptiveFactorizationNetwork(
            field_dims, embed_dim=16, LNN_dim=1500, mlp_dims=(100, 100), dropouts=(0.5, 0.5, 0.5))
    elif name == 'mhafm':
        print("Model:Multihead Attention Factorization Machine Model")
        return MultiheadAttentionalFactorizationMachineModel(
            field_dims, embed_dim=16, attn_embed_dim=64, num_heads=2, ffn_embed_dim=16, num_layers=2, mlp_dims=(100, 100), dropout=0.2
        )
    elif name == 'dcan':
        print("Model:Deep Cross Attentional Network Model")
        return DeepCrossAttentionalNetworkModel(
            field_dims, embed_dim=16, attn_embed_dim=64, num_heads=2, ffn_embed_dim=64, num_layers=3, mlp_dims=(100, 100), dropout=0.2
        )
    else:
        raise ValueError('unknown model name: ' + name)


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy: # METRIC IS LOG_LOSS
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts), log_loss(targets, predicts)


def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir):
    np.random.seed(2020)
    torch.manual_seed(2020)
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    
    for seed in [0, 1994, 2020, 1019, 1125]:
        print(f'set the seed to: {seed}')
        np.random.seed(seed)
        torch.manual_seed(seed)
        for model_name in ['afm', 'nfm', 'ipnn', 'opnn', 'wd', 'dcn', 'dfm', 'xdfm', 'afi', 'afn']:
        # for model_name in ['lr', 'fm', 'afm', 'hofm', 'nfm', 'ipnn', 'opnn', 'wd', 'dcn', 'dfm', 'xdfm', 'afi', 'afn']:
        # for model_name in ['dfm']:
            print(f'model name: {model_name}')
            model = get_model(model_name, dataset).to(device)
            print(model)
            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            early_stopper = EarlyStopper(num_trials=3, save_path=f'{save_dir}/{model_name}_seed{seed}.pt')
            for epoch_i in range(epoch):
                train(model, optimizer, train_data_loader, criterion, device)
                auc, log_loss = test(model, valid_data_loader, device)
                print('epoch:', epoch_i, 'validation: auc:', auc, 'validation: log_loss:', log_loss)
                if not early_stopper.is_continuable(model, auc):
                    print(f'validation: best auc: {early_stopper.best_accuracy}')
                    break
            model = torch.load(f'{save_dir}/{model_name}_seed{seed}.pt').to(device)
            auc, log_loss = test(model, test_data_loader, device)
            print(f'test auc: {auc}')
            print(f'test log_loss: {log_loss}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='avazu')
    parser.add_argument('--dataset_path', default='data/avazu/train.csv',
                        help='criteo/train.txt, avazu/train.csv, or ml-1m/ratings.dat')
    parser.add_argument('--model_name', default='mhafm')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='checkpoints/avazu')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)
