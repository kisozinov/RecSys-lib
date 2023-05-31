import torch
from torch import nn, optim, Tensor
import matplotlib.pyplot as plt

from data import dataloader
from utils import *
from models.lightgcn_cfg import *
class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError

# defines LightGCN model
class LightGCN(BasicModel):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """

    def __init__(self, dataset):
        """Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super(LightGCN, self).__init__()
        self.dataset : dataloader.BasicDataset = dataset
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.embedding_dim, self.K = EMB_DIM, N_LAYERS
        #self.add_self_loops = add_self_loops
        self.users_emb = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_dim) # e_u^0
        self.items_emb = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.embedding_dim) # e_i^0

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.users_emb.weight
        items_emb = self.items_emb.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        # if self.config['dropout']:
        #     if self.training:
        #         print("droping")
        #         g_droped = self.__dropout(self.keep_prob)
        #     else:
        #         g_droped = self.Graph        
        # else:
        #     g_droped = self.Graph    
        graph = self.Graph
        for layer in range(self.K):
            all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items, users_emb, items_emb
    
    def getUsersRating(self, users):
        all_users, all_items, _, _ = self.computer()
        users_emb = all_users[users]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items, _, _ = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.users_emb(users)
        pos_emb_ego = self.items_emb(pos_items)
        neg_emb_ego = self.items_emb(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        #print(users_emb.shape)
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        #print(float(len(users)))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
    
    def evaluation(self, topk, lambda_val):
        """Evaluates model loss and metrics including recall, precision, ndcg @ k

        Args:
            model (LighGCN): lightgcn model
            edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
            sparse_edge_index (sparseTensor): sparse adjacency matrix for split to evaluate
            exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
            k (int): determines the top k items to compute metrics on
            lambda_val (float): determines lambda for bpr loss

        Returns:
            tuple: bpr loss, recall @ k, precision @ k, ndcg @ k
        """
        with torch.no_grad():
            dataset = self.dataset
            S = UniformSample_original_python(dataset)
            users = torch.Tensor(S[:, 0]).long()
            posItems = torch.Tensor(S[:, 1]).long()
            negItems = torch.Tensor(S[:, 2]).long()
            users = users.to(DEVICE)
            posItems = posItems.to(DEVICE)
            negItems = negItems.to(DEVICE)

            loss, reg_loss = self.bpr_loss(users, posItems, negItems)
            reg_loss *= lambda_val
            loss += reg_loss

            # users_emb_final, pos_items_emb_final, neg_items_emb_final, \
            #     users_emb_0, pos_items_emb_0, neg_items_emb_0 = model.getEmbedding(users, posItems, negItems)
            # loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0,
            #                 neg_items_emb_final, neg_items_emb_0, lambda_val).item()

            #map_dict, ndcg_dict, recall_dict, precision_dict = {},{},{},{},
            #recall_old, precision_old, ndcg_old = get_metrics(model, topk)
            testDict = self.dataset.testDict
            users = list(testDict.keys())
            #print('test users shape ', len(users), users, sep='\n')
            rating = self.getUsersRating(users)
            allPos = self.dataset.getUserPosItems(users)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            top_K_items, top_K_indices = torch.topk(rating, k=max(topk))
            ground_truth = [testDict[u] for u in users]
            r = torch.tensor(getLabel(ground_truth, top_K_indices))
            precision = {f'Precision@{k}':0. for k in topk}
            recall = {f'Recall@{k}':0. for k in topk}
            ndcg = {f'NDCG@{k}':0. for k in topk}
            for k in topk:
                rec, pre = RecallPrecision_ATk(ground_truth, r, k)
                precision[f'Precision@{k}'] = pre
                recall[f'Recall@{k}'] = rec
                ndcg[f'NDCG@{max(topk)}'] = NDCGatK_r(ground_truth, r, max(topk))
            return loss.item(), recall, precision, ndcg

    def fit(self, optimizer, lr_scheduler, epochs=EPOCHS, loss_f=bpr_loss):
        self.train()
        train_losses = []
        val_losses = []
        val_recall = []
        val_ndcg = []
        val_precision = []
        for epoch in range(1, epochs+1):
            edges = torch.tensor(UniformSample_original_python(self.dataset), dtype=torch.long)
            #shuffle and then batch sample
            edges = edges[torch.randperm(edges.size()[0])]
            total_batch = edges.shape[0] // BATCH_SIZE + 1
            # mini batching
            aver_loss = 0.
            for i in range(0, edges.shape[0], BATCH_SIZE):
                user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(
                    BATCH_SIZE, edges[i:i+BATCH_SIZE, :])
                user_indices, pos_item_indices, neg_item_indices = user_indices.to(
                    DEVICE), pos_item_indices.to(DEVICE), neg_item_indices.to(DEVICE)

                train_loss, reg_loss = loss_f(self, user_indices, pos_item_indices, neg_item_indices)
                reg_loss = reg_loss * LAMBDA
                train_loss = train_loss + reg_loss
                #print(f'epoch {epoch} | train loss: ', train_loss.cpu().item())
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                aver_loss += train_loss.cpu().item()
            aver_loss /= total_batch

            if epoch % EPOCHS_PER_EVAL == 0 or epoch == 1:
                self.eval()
                val_loss, recall, precision, ndcg = self.evaluation(TOPK, LAMBDA)
                print(f"[Epoch {epoch}/{epochs}] train_loss: {round(aver_loss, 5)}, \
                    val_loss: {round(val_loss, 5)}\n {recall} \n {precision} \n {ndcg}")
                train_losses.append(aver_loss)
                val_losses.append(val_loss)
                val_recall.append(recall[f'Recall@{max(TOPK)}'])
                val_ndcg.append(ndcg[f'NDCG@{max(TOPK)}'])
                val_precision.append(precision[f'Precision@{max(TOPK)}'])
                self.train()
                #save_ckpt(model, optimizer=optimizer, scheduler=scheduler, epoch=epoch)

            if epoch % EPOCHS_PER_LR_DECAY == 0 and epoch != 0:
                lr_scheduler.step()

        iters = [iter * EPOCHS_PER_EVAL for iter in range(len(train_losses))]
        #figure, axis = plt.subplots(2,1)
        plot1 = plt.subplot2grid((1, 5), (0, 0), colspan=2)
        plot2 = plt.subplot2grid((1, 5), (0, 3), colspan=2)
        plot1.plot(iters, train_losses, label='train')
        plot1.plot(iters, val_losses, label='validation')
        plot1.set_xlabel('iteration')
        plot1.set_ylabel('loss')
        plot1.set_title('LightGCN train & val loss curves')
        plot1.legend()
        #plt.savefig(f'../images/lightgcn_{self.dataset.name}_losses.png')
        #plt
        plot2.plot(iters, val_recall, label='Recall@20')
        plot2.plot(iters, val_ndcg, label='NDCG@20')
        plot2.plot(iters, val_precision, label='Precision@20')
        plot2.set_xlabel('iteration')
        plot2.set_ylabel('Metric value')
        plot2.set_title('LightGCN metrics curves')
        plot2.legend()

        plt.legend()
        plt.savefig(f'../reports/figures/lightgcn_{self.dataset.name}.png')


    def save_ckpt(self, model_name, epochs):
        dataset = self.dataset
        ckpt_path = f'../models/{model_name}/{model_name}_{dataset.name}.pt'
        print('Saving checkpoint to path ', ckpt_path)
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.state_dict()
        }, ckpt_path)

