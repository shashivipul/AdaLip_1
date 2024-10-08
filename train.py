# Code Adapted from Pro-GNN(https://github.com/ChandlerBang/Pro-GNN)
import time
import argparse
import numpy as np
import torch

from deeprobust.graph.defense import GCN
from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.utils import preprocess, encode_onehot, get_train_val_test
from scipy.sparse import load_npz
import argparse
import os
import pickle
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import pickle
from scipy.sparse import load_npz

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--bound', type=float, default=0, help='weight of bound importance')
parser.add_argument('--two_stage',type = str,help = "Use Two Stage",default="y")
parser.add_argument('--optim',type = str,help = "Optimizer",default="sgd")
parser.add_argument('--lr_optim',type = float, help = "learning rate for the graph weight update" ,default=1e-3)
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--decay',type=str,default="n",help="whether to use decay or not")
parser.add_argument('--plots',type=str,default="n",help="whether to plot the acc or not")
parser.add_argument('--test',type=str,default="y",help="Test only")
parser.add_argument('--only_gcn', action='store_true',
        default=False, help='test the performance of gcn without other components')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate for GNN model.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
        choices=['cora', 'citeseer', 'cora_ml', 'polblogs','pubmed', 'acm', 'blogcatalog', 'uai', 'flickr','chameleon', 'squirrel','german'], help='dataset')
parser.add_argument('--attack', type=str, default='meta',
        choices=['no', 'meta', 'nettack','random'])
parser.add_argument('--ptb_rate', type=float, default=0.05, help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=400, help='Number of epochs to train.')
parser.add_argument('--alpha', type=float, default=1, help='weight of Forbeius norm')
parser.add_argument('--epochs_pre', type=int,  default=500, help='Number of epochs to train in Two-Stage.')
parser.add_argument('--gamma', type=float, default=1, help='weight of GCN')
parser.add_argument('--beta', type=float, default=0, help='weight of feature smoothing')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--glp', type=int, default=100, help='percentage of graph learning')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--symmetric', action='store_true', default=False,
            help='whether use symmetric matrix')


args = parser.parse_args()


args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.ptb_rate == 0:
    args.attack = "no"

print(args)


# Here the random seed is to split the train/val/test data,
# we need to set the random seed to be the same as that when you generate the perturbed graph
# but now change the setting from nettack to prognn which directly loads the prognn splits
# data = Dataset(root='/tmp/', name=args.dataset, setting='nettack', seed=15)


dataset=args.dataset
if dataset in ['chameleon', 'squirrel','german']:
    adj=load_npz('adj.npz')
    features=load_npz('features.npz')
    labels=np.load('labels.npy')
    idx_train=np.load('idx_train.npy')
    idx_test=np.load('idx_test.npy')
    idx_val=np.load('idx_val.npy')
    perturbed_adj=adj
    
    
    #with open(f'data/{dataset}_data.pickle', 'rb') as handle:
    #    data = pickle.load(handle)
    #features = data["features"]
    #labels = data["labels"]
    #idx_train = data["idx_train"]
    #idx_val = data["idx_val"]
    #idx_test = data["idx_test"]
    #if  args.attack in ['meta','nettack']:
    #    adj = load_npz(f'data/{dataset}_perturbed_{args.ptb_rate}.npz')
    #else:
    #    adj = load_npz(f'data/{dataset}.npz')
    #if args.attack == 'nettack':
    #    idx_test = np.load(f"data/{dataset}_idx_test.npy")
    #perturbed_adj = adj

else : 

    data = Dataset(root='/tmp/', name=args.dataset,setting='prognn')
    adj, features, labels = data.adj, data.features, data.labels
    torch.save(adj,'clean_adj.pt')
    print('\n\n Saved')
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    

    if args.dataset == 'pubmed':
        # just for matching the results in the paper, see details in https://github.com/ChandlerBang/Pro-GNN/issues/2
        #print("just for matching the results in the paper," + \
        #      "see details in https://github.com/ChandlerBang/Pro-GNN/issues/2")
        idx_train, idx_val, idx_test = get_train_val_test(adj.shape[0],
                val_size=0.1, test_size=0.8, stratify=encode_onehot(labels), seed=15)

    if args.attack == 'random':
        from deeprobust.graph.global_attack import Random
        # to fix the seed of generated random attack, you need to fix both np.random and random
        # you can uncomment the following code
        import random; random.seed(args.seed)
        np.random.seed(args.seed)
        attacker = Random()
        n_perturbations = int(args.ptb_rate * (adj.sum()//2))
        attacker.attack(adj, n_perturbations, type='add')
        perturbed_adj = attacker.modified_adj

    if args.attack == 'no':
        perturbed_adj = adj

    if args.attack == 'meta' or args.attack == 'nettack':
        perturbed_data = PrePtbDataset(root='/tmp/',
                name=args.dataset,
                attack_method=args.attack,
                ptb_rate=args.ptb_rate)
        perturbed_adj = perturbed_data.adj
        torch.save(perturbed_adj,'pert_adj_25.pt')
        if args.attack == 'nettack':
            idx_test = perturbed_data.target_nodes

np.random.seed(args.seed)
torch.manual_seed(args.seed)
print(type(features))
with open('csr_matrix.pkl', 'wb') as f:
    pickle.dump(features, f)
    
    







if args.only_gcn:
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
              dropout=args.dropout, device=device)

else:
    from bounded_gcn import BoundedGCN
    model = BoundedGCN(nfeat=features.shape[1],
                       nhid=args.hidden,
                       nclass=labels.max().item() + 1,
                       dropout=args.dropout, device=device, bound=args.bound)

    if args.two_stage == "y":
        from Bounded_two_stage import RSGNN
    else:
        from BoundedJointLearning import RSGNN


if args.only_gcn:

    perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, sparse=True, device=device)
    

    model.fit(features, perturbed_adj, labels, idx_train, idx_val, verbose=True, train_iters=args.epochs)
    model.test(idx_test)
    


else:
    perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, device=device)
    #print(perturbed_adj)
    rsgnn = RSGNN(model, args, device)
    if args.two_stage=="y":
        adj_new = rsgnn.fit(features, perturbed_adj)
        t_total=time.time()
        model.fit(features, adj_new, labels, idx_train, idx_val, verbose=False, train_iters=args.epochs,bound=args.bound) #######
        print('\n Total Time = ',time.time()-t_total)
        model.test(idx_test)
    else:
        rsgnn.fit(features, perturbed_adj, labels, idx_train, idx_val)
        rsgnn.test(features, labels, idx_test)

