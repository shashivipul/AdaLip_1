import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.utils import accuracy
import matplotlib.pyplot as plt
import warnings
from utils import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
torch.set_printoptions(profile="full")
from sklearn.metrics import f1_score, roc_auc_score
from vipul_fair import fair_metric
class RSGNN:
    """ RWL-GNN (Robust Graph Neural Networks using Weighted Graph Laplacian)
    Parameters
    ----------
    model:
        model: The backbone GNN model in RWLGNN
    args:
        model configs
    device: str
        'cpu' or 'cuda'.
    Examples
    --------
    See details in https://github.com/Bharat-Runwal/RWL-GNN.
    """

    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.estimator = None
        self.model = model.to(device)

        # self.train_cost = []
        self.valid_cost = []
        self.train_acc = []
        self.valid_acc = []



    def fit(self, features, adj, labels, idx_train, idx_val):
        """Train RWL-GNN.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices
        """

       # print(" Bounded Joint Learning .........")
        args = self.args
        self.symmetric = args.symmetric
        self.optimizer = optim.Adam(self.model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        
        optim_sgl = args.optim
        lr_sgl = args.lr_optim

        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L_noise = D - adj

        # INIT
        self.weight = self.Linv(L_noise)
      
        self.weight.requires_grad = True
        self.w_old = torch.zeros_like(self.weight)  ####################  To store previous w value ( w^{t-1} )
        self.weight = self.weight.to(self.device)
        self.w_old = self.w_old.to(self.device)

        self.bound = args.bound

        self.d =  features.shape[1]
    
        c = self.Lstar(2*L_noise*args.alpha - args.beta*(torch.matmul(features,features.t())) )



        if optim_sgl == "Adam":
            self.sgl_opt =AdamOptimizer(self.weight,lr=lr_sgl)
        elif optim_sgl == "RMSProp":
            self.sgl_opt = RMSProp(self.weight,lr = lr_sgl)
        elif optim_sgl == "sgd_momentum":
            self.sgl_opt = sgd_moment(self.weight,lr=lr_sgl)
        else:
            self.sgl_opt = sgd(self.weight,lr=lr_sgl) 

        t_total = time.time()
        
        for epoch in range(args.epochs):
            if args.only_gcn:
                estimate_adj = self.A()
                self.train_gcn(epoch, features, estimate_adj,
                        labels, idx_train, idx_val)
            else:
                
                for i in range(int(args.outer_steps)):
                    self.train_specific(epoch, features, L_noise, labels,
                            idx_train, idx_val,c,epoch)

                for i in range(int(args.inner_steps)):
                    estimate_adj = self.A()
                    self.train_gcn(epoch, features, estimate_adj,
                            labels, idx_train, idx_val)


        if args.plots=="y":
            self.plot_acc()
            self.plot_cost()

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(args)

        # Testing
        #print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)



    def w_grad(self,alpha,c,new_term):
      with torch.no_grad():
        grad_f = self.Lstar(alpha*self.L()) - c + new_term
        return grad_f 



    def train_specific(self,epoch, features, L_noise, labels, idx_train, idx_val,c,iter):
        args = self.args
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()
        
        y = self.weight.clone().detach()
        y = y.to(self.device)
        y.requires_grad = True


        normalized_adj = self.normalize(y)
        output,_ = self.model(features, normalized_adj)
        loss_gcn =args.gamma * F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        # loss_diffiential = loss_fro + gamma*loss_gcn+args.lambda_ * loss_smooth_feat

        gcn_grad = torch.autograd.grad(
        inputs= y,
        outputs=loss_gcn,
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(loss_gcn),
        only_inputs= True,
          )[0]

        sq_norm_Aw = torch.norm(self.A(), p="fro")**2

        new_term = self.bound**2  * (2 * self.Astar(self.A()) - self.w_old) / \
                   (sq_norm_Aw - self.w_old.t() * self.weight)
        sgl_grad = self.w_grad(args.alpha ,c,new_term)


        total_grad  = sgl_grad + gcn_grad 

        self.w_old = self.weight
        self.weight = self.sgl_opt.backward_pass(total_grad)
        self.weight = torch.clamp(self.weight,min=0)

        self.model.eval()
        normalized_adj = self.normalize()
        output,_ = self.model(features, normalized_adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        if args.plots == "y":
            self.train_acc.append(acc_train.detach().cpu().numpy())
            self.valid_cost.append(loss_val.detach().cpu().numpy())
            self.valid_acc.append(acc_val.detach().cpu().numpy())
        
        if args.test=="n":
            print('Epoch: {:04d}'.format(epoch+1),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_val: {:.4f}'.format(acc_val.item()),
                'time: {:.4f}s'.format(time.time() - t))

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_fro: {:.4f}'.format(loss_fro.item()),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()))
                


                

    def train_gcn(self, epoch, features, adj, labels, idx_train, idx_val):
        args = self.args
        # estimator = self.estimator
        adj = self.normalize()

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        output,_ = self.model(features, adj)

        #self.l2_reg =  2 * self.bound**2  * ( torch.log(torch.norm(self.model.gc1.weight)) + torch.log(torch.norm(self.model.gc2.weight)) ) #RSGNN
        self.l2_reg =  2 * self.bound**2  * ( torch.log(torch.linalg.matrix_norm(self.model.gc1.weight,ord=2)) + torch.log(torch.linalg.matrix_norm(self.model.gc2.weight,ord=2)) ) #Adalip

        loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + self.l2_reg


        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward(retain_graph = True)
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        output,_ = self.model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = adj.detach()
            Lip_bound=torch.linalg.matrix_norm(self.model.gc1.weight,ord=2)*torch.linalg.matrix_norm(self.model.gc2.weight,ord=2)
            #print('Lips',Lip_bound)
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))

    def test(self, features, labels, idx_test):
        """Evaluate the performance of RWL-GNN on test set
        """
        #print("\t=== testing ===")
        self.model.eval()
        adj = self.best_graph
        #with open('adjacency_matrix.pkl', 'wb') as f:
    	 #   pickle.dump(adj, f)
        #print(adj)
        #diagonal_elements = torch.diag(adj)
        #non_1=torch.sum(diagonal_elements>0.5)
        #print(f"Number of non-zero diagonal entries: {non_1.item()}")
        #torch.save(adj,'adj.pt')
        #adj= torch.where(adj >= 0.1, adj, torch.tensor(0.0))
        labels_cpu= labels[idx_test].cpu().detach().numpy()
        output,emb = self.model(features, adj)
        print(output)
        #print(emb)
        #print(output.shape)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        counter_features = features.clone()
        sens_idx=0
        counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
        counter_output,emb_2 = self.model(counter_features,adj)
        output_preds = output.max(1)[1].type_as(labels)
        print(labels)
        print(output_preds)
        counter_output_preds = counter_output.max(1)[1].type_as(labels)
        counterfactual_fairness = 1 - (output_preds.eq(counter_output_preds)[idx_test].sum().item()/idx_test.shape[0])
        print('Counterfactual',counterfactual_fairness)
        f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
        print('The f1',f1_s)
        sens=torch.load('sens.pt') 
        parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
        print(f'Parity: {parity} | Equality: {equality}')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        #print('\nshape of output',output[idx_test].shape)
        #print('\nOutput is',output[idx_test])
        out_probs=torch.exp(output[idx_test])
        #print('\nThe probs are',out_probs)
        correct_class_probs=out_probs[torch.arange(out_probs.size(0)), labels[idx_test]]
        masked_probs = out_probs.clone()
        masked_probs[torch.arange(out_probs.size(0)), labels[idx_test]] = -1
        max_other_probs = masked_probs.max(dim=1).values
        differences = correct_class_probs - max_other_probs
        #print('\nThe margins are',differences)
        difference_numpy=differences.detach().cpu().numpy()
        df = pd.DataFrame({'Differences': difference_numpy})
        df.to_csv('differences_25.csv', index=False)









        



        
        output_cpu=output[idx_test].cpu().detach().numpy()
        transform= TSNE
        transformed= transform(n_components=2)
        transformed_output=transformed.fit_transform(output_cpu)
        print(transformed_output.shape)
        plt.figure(figsize=(4,4))
        scatter=plt.scatter(transformed_output[:,0],transformed_output[:,1],c=labels_cpu,cmap='spring',s=3)
        ax=plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_aspect('equal',adjustable='box')
        plt.tight_layout()
        plt.savefig("tsne_1.png",bbox_inches='tight',dpi=1200)
        
        #print(output.max(1)[1].type_as(labels))
        #preds = output.max(1)[1].type_as(labels)
        #print("\n The preds are ",preds.eq(labels).double())
        #correct = preds.eq(labels).double()
        #correct_indices = correct.nonzero(as_tuple=True)[0]
        #print("\nCorrect Indices",correct_indices)
        #max_values, max_indices = emb.max(1)
        #second_max_values = emb.scatter(1, max_indices.unsqueeze(1), -float('inf')).max(1)[0]
        #correct_max_values = max_values[correct_indices]
        #print(correct_max_values)
        #correct_second_max_values = second_max_values[correct_indices]
        #print(correct_second_max_values)
        #differences = correct_max_values - correct_second_max_values

        # Print the differences for the correct predictions
        #print("\nDifferences between max and second max embeddings for correct predictions:", differences)

        

        print("\tTest set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    def A(self,weight=None):
        # with torch.no_grad():
        if weight == None:
            k = self.weight.shape[0]
            a = self.weight
        else:
            k = weight.shape[0]
            a = weight
        n = int(0.5 * (1 + np.sqrt(1 + 8 * k)))
        Aw = torch.zeros((n,n),device=self.device)
        b=torch.triu_indices(n,n,1)
        Aw[b[0],b[1]] =a
        Aw = Aw + Aw.t()
        return Aw

    def Astar(self,adjacency):
        n = adjacency.shape[0]
        k = n * (n - 1) // 2
        weight = torch.zeros(k,device= self.device)
        b = torch.triu_indices(n, n, 1)
        weight = adjacency[b[0], b[1]]
        return weight



    def L(self,weight=None):
        if weight==None:
            k= len(self.weight)
            a = self.weight 
        else:
            k = len(weight)
            a = weight
        n = int(0.5*(1+ np.sqrt(1+8*k)))
        Lw = torch.zeros((n,n),device=self.device)
        b=torch.triu_indices(n,n,1)
        Lw[b[0],b[1]] = -a  
        Lw = Lw + Lw.t()
        row,col = np.diag_indices_from(Lw)
        Lw[row,col] = -Lw.sum(axis=1)
        return Lw     



    def Linv(self,M):
      with torch.no_grad():
        N=M.shape[0]
        k=int(0.5*N*(N-1))
        # l=0
        w=torch.zeros(k,device=self.device)
        indices=torch.triu_indices(N,N,1)
        M_t=torch.tensor(M)
        w=-M_t[indices[0],indices[1]]
        return w


    def Lstar(self,M):
        N = M.shape[1]
        k =int( 0.5*N*(N-1))
        w = torch.zeros(k,device=self.device)
        tu_enteries=torch.zeros(k,device=self.device)
        tu=torch.triu_indices(N,N,1)
    
        tu_enteries=M[tu[0],tu[1]]
        diagonal_enteries=torch.diagonal(M)

        b_diagonal=diagonal_enteries[0:N-1]
        x=torch.linspace(N-1,1,steps=N-1,dtype=torch.long,device=self.device)
        x_r = x[:N]
        diagonal_enteries_a=torch.repeat_interleave(b_diagonal,x_r)
     
        new_arr=torch.tile(diagonal_enteries,(N,1))
        tu_new=torch.triu_indices(N,N,1)
        diagonal_enteries_b=new_arr[tu_new[0],tu_new[1]]
        w=diagonal_enteries_a+diagonal_enteries_b-2*tu_enteries
   
        return w
        
        
    def fair_metric(pred, labels, sens):
    	idx_s0 = sens==0
    	idx_s1 = sens==1
    	idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    	idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    	parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    	equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    	return parity.item(), equality.item()



    def normalize(self,w=None):

        if self.symmetric:
            if w == None:
                adj = (self.A() + self.A().t())
            else:
                adj = self.A(w)
            
            adj = adj + adj.t()
        else:
            if w == None:
                adj = self.A()
            else:
                adj = self.A(w)

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx
