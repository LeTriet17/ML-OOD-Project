import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from backbone import *
import numpy as np

def init_center(args, eps=0.001):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    if args.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.device)
    n_samples = 0
    c = torch.zeros(args.n_hidden, device=device)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    radius=np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    if radius<0.1:
        radius=0.1
    return radius

def loss_function(nu,scores, radius):
    
    loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    return loss

def anomaly_score(data_center,outputs,radius=0):
    
    dist = torch.sum((outputs - data_center) ** 2, dim=1)
    scores = dist - radius ** 2
    return dist,scores

class GNNSafe(nn.Module):
    '''
    The model class of energy-based models for out-of-distribution detection
    The parameter args.use_reg and args.use_prop control the model versions:
        Energy: args.use_reg = False, args.use_prop = False
        Energy FT: args.use_reg = True, args.use_prop = False
        GNNSafe: args.use_reg = False, args.use_prop = True
        GNNSafe++ args.use_reg = True, args.use_prop = True
    '''
    def __init__(self, d, c, args):
        super(GNNSafe, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.use_bn)
        elif args.backbone == 'gen':
            self.encoder = GEN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                        out_channels=c, num_layers=args.num_layers,
                        dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout, use_bn=args.use_bn)
        elif args.backbone == 'mixhop':
            self.encoder = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gcnjk':
            self.encoder = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gatjk':
            self.encoder = GATJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        else:
            raise NotImplementedError
        if args.use_oc:
            self.radius=torch.tensor(0, device=f'cuda:{args.device}')# radius R initialized with 0 by default.

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        '''return predicted logits'''
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def propagation(self, e, edge_index, prop_layers=1, alpha=0.5):
        '''energy belief propagation, return the energy after propagation'''
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        return e.squeeze(1)

    def detect(self, dataset, node_idx, device, args):
        '''return negative energy, a vector for all input nodes'''
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits = self.encoder(x, edge_index)
        if args.dataset in ('proteins', 'ppi'): # for multi-label binary classification
            logits = torch.stack([logits, torch.zeros_like(logits)], dim=2)
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1).sum(dim=1)
        else: # for single-label multi-class classification
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1)
        if args.use_prop: # use energy belief propagation
            neg_energy = self.propagation(neg_energy, edge_index, args.K, args.alpha)
        return neg_energy[node_idx]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        '''return loss for training'''
        x_in, edge_index_in = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        x_out, edge_index_out = dataset_ood.x.to(device), dataset_ood.edge_index.to(device)

        # get predicted logits from gnn classifier
        logits_in = self.encoder(x_in, edge_index_in)
        logits_out = self.encoder(x_out, edge_index_out)

        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx

        # compute supervised training loss
        if args.dataset in ('proteins', 'ppi'):
            sup_loss = criterion(logits_in[train_in_idx], dataset_ind.y[train_in_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in[train_in_idx], dim=1)
            #print(pred_in, dataset_ind.y[train_in_idx])
            
            sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))
        if args.use_oc:
            
            center = logits_in[train_in_idx].mean(dim=0)
            # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
            center[(abs(center) < args.eps) & (center < 0)] = -args.eps
            center[(abs(center) < args.eps) & (center > 0)] = args.eps
            dist,scores = anomaly_score(center,logits_in[train_in_idx],self.radius)      
            oc_loss = loss_function(args.nu, scores,self.radius)
            self.radius = get_radius(dist, args.nu)

        if args.use_reg: # if use energy regularization
            if args.dataset in ('proteins', 'ppi'): # for multi-label binary classification
                logits_in = torch.stack([logits_in, torch.zeros_like(logits_in)], dim=2)
                logits_out = torch.stack([logits_out, torch.zeros_like(logits_out)], dim=2)
                energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1).sum(dim=1)
                energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1).sum(dim=1)
            else: # for single-label multi-class classification
                energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1)
                energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1)

            if args.use_prop: # use energy belief propagation
                energy_in = self.propagation(energy_in, edge_index_in, args.K, args.alpha)[train_in_idx]
                energy_out = self.propagation(energy_out, edge_index_out, args.K, args.alpha)[train_ood_idx]
            else:
                energy_in = energy_in[train_in_idx]
                energy_out = energy_out[train_in_idx]

            # truncate to have the same length
            if energy_in.shape[0] != energy_out.shape[0]:
                min_n = min(energy_in.shape[0], energy_out.shape[0])
                energy_in = energy_in[:min_n]
                energy_out = energy_out[:min_n]
            # compute regularization loss
            reg_loss = torch.mean(F.relu(energy_in - args.m_in) ** 2 + F.relu(args.m_out - energy_out) ** 2)
            if args.use_oc:
                loss = sup_loss + args.lamda1 * reg_loss + args.lamda2*oc_loss
                print(f'loss: {sup_loss}, reg_loss: {reg_loss}, oc_loss: {oc_loss}')
            loss = sup_loss + args.lamda1 * reg_loss
            print(f'loss: {sup_loss}, reg_loss: {reg_loss}')
        else:
            if args.use_oc:
                loss = sup_loss + args.lamda1*oc_loss
                print(f'loss: {sup_loss}, oc_loss: {oc_loss}')
            loss = sup_loss
            print(f'loss: {sup_loss}')
            

        return loss
