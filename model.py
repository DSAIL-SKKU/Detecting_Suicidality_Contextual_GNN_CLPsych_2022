import os
import pandas as pd
import numpy as np
from pprint import pprint
import argparse
import time
from datetime import datetime

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# torch:
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset,random_split
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
from transformers import BertTokenizer, AdamW, BertModel 
from transformers import AdamW 

from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback

#gcn
from scipy.sparse import coo_matrix
import dgl
import dgl.nn.pytorch as dglnn
from dgl import function as fn
from dgl.utils import expand_as_pair, check_eq_shape

from Graph_Conv import *
from utils import *


class Arg:
    random_seed: int = 2021  # Random Seed
    
    cache_dir = './models/cache' 
    log_dir = './models/checkpoints'
    log_name = 'bert-base-post'
    version = 1
    
    #data
    data_path = 'post_level_know_dataset.pkl'
    adj_path = 'post_dic_array_two_0906_normal.npy'
    dic_feature_path = 'two_suicide_dictionary_0906.pkl'
    dic_dic = 'dic_dic_pmi_two_0906.npy'
    
    
    #setting
    pretrained_model =  "bert-base-uncased" # Transformers PLM name
    pretrained_tokenizer = "bert-base-uncased"  #Transformers Tokenizer Name. Overrides `pretrained_model`
    
    epochs: int = 5  # Max Epochs, BERT paper setting [3,4,5]
    max_length: int = 200  # Max Length input size
    report_cycle: int = 30  # Report (Train Metrics) Cycle
    cpu_workers: int = int(os.cpu_count() -4)   # Multi cpu workers
    test_mode: bool = False#True  # Test Mode enables `fast_dev_run`
    optimizer: str = 'AdamW'  # AdamW vs AdamP
    lr_scheduler: str = 'exp'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    hidden_size = 768 # BERT-base: 768, BERT-large: 1024, BERT paper setting
    batch_size: int = 32



class Model(LightningModule):
    def __init__(self, config, options,seed):
        super().__init__()
        # config:
        self.args = options
        self.config = config
        self.batch_size = self.args.batch_size
        # meta data:
        self.epochs_index = 0
        self.label_cols = 'y' 
        self.num_labels = self.config['num_labels']
        self.seed = seed
        
        #tuning
        self.loss_type = self.config['loss']
        self.agg_type = self.config['agg']
        self.drop_out = self.config['dropout']
        self.lr = self.config['lr']
        self.hidden = self.config['hidden']
        self.s_drop = self.config['s_drop']
        self.kernel_out = self.config['kernel_out']
        self.dic_hidden = self.config['dic_hidden']
        self.gpu = self.config['gpu']
                
        # modules:
        self.tokenizer = BertTokenizer.from_pretrained(self.args.pretrained_tokenizer)
        self.bert_data = BertModel.from_pretrained(self.args.pretrained_model)
        
        # graphsage        
        self.sampler = dgl.dataloading.MultiLayerNeighborSampler([
                        {('dic', 'co-occur', 'dic'): 150,
                         ('dic', 'in', 'post'): 50},
                        {('dic', 'co-occur', 'dic'): 150,
                         ('dic', 'in', 'post'): 50}
                    ])
        
        self.conv1 = dglnn.HeteroGraphConv({
                'in' : SAGEConv((201,self.args.hidden_size), self.hidden, aggregator_type= self.agg_type, feat_drop=self.s_drop,kernel_out=self.kernel_out),
                'co-occur' : SAGEConv((201,201), self.dic_hidden, aggregator_type=self.agg_type,feat_drop=self.s_drop,kernel_out=self.kernel_out)},
                aggregate='sum').double()
        
        #
        
        self.conv2 = dglnn.HeteroGraphConv({
                'in' : SAGEConv((self.dic_hidden, self.hidden), int(self.hidden/2), aggregator_type= self.agg_type, feat_drop=self.s_drop, kernel_out = self.kernel_out),
                'co-occur' : SAGEConv((self.dic_hidden, self.dic_hidden), self.dic_hidden, aggregator_type=self.agg_type,feat_drop=self.s_drop,kernel_out=self.kernel_out)},
                aggregate='sum').double()
        
        self.dropout = nn.Dropout(self.drop_out)
        self.lin = torch.nn.Linear(int(self.hidden/2), self.num_labels)


    def forward(self,text_data, edges, **kwargs):
        
        # post embedding
        outputs_data = self.bert_data(input_ids =text_data, **kwargs) # return: last_hidden_state, pooler_output, hidden_states, attentions
        p_feat = outputs_data[1] 
        
        #graph
        edge_index = torch.nonzero(edges, as_tuple=False).T
        dic_index = torch.nonzero(self.dic_dic, as_tuple=False).T.to(f"cuda:{self.gpu}")
        
        g = dgl.heterograph(data_dict = {('dic', 'in', 'post')  : (edge_index[1], edge_index[0]),
                                        ('dic', 'co-occur', 'dic')  : (dic_index[0], dic_index[1])},
                           num_nodes_dict = {'dic':len(self.dic_feature), 'post':len(p_feat)}).to(f"cuda:{self.gpu}")
        
        
        d_feat = torch.tensor(self.dic_feature.numpy().astype(np.float32),dtype = torch.double).to(f"cuda:{self.gpu}")
        
        g.ndata['features'] = {'post' : p_feat.double(), 'dic' : d_feat}
        g.edata['features'] = {'co-occur' : torch.tensor(coo_matrix(self.dic_dic).data).to(f"cuda:{self.gpu}"), 'in':torch.tensor(coo_matrix(edges.cpu()).data).to(f"cuda:{self.gpu}")}
        
        train_nid = {'post': torch.tensor(range(len(p_feat))).to(f"cuda:{self.gpu}"),
                    'dic': torch.tensor(range(len(self.dic_feature))).to(f"cuda:{self.gpu}")} #
        #, 
        
        collator = dgl.dataloading.NodeCollator(g, train_nid, self.sampler)
        dataloader = DataLoader(
            collator.dataset, collate_fn=collator.collate,
            batch_size=int(g.number_of_nodes()/5), shuffle=False, drop_last=False)
        
        post_output = None
        for i,(input_nodes, output_nodes, blocks) in enumerate(dataloader):
            input_features = blocks[0].srcdata['features']     # returns a dict
            output = self.conv1(blocks[0], input_features,blocks[0].edata['features'])      
            output = self.conv2(blocks[1], output,blocks[1].edata['features'])

            if 'post' in output:
                if post_output is None:
                    post_output = output['post']

                else:
                    post_output = torch.cat([post_output,output['post']],dim=0)
        
        del collator
        del dataloader

        y = self.lin(self.dropout(post_output.float()))
        
        return y
        

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.5)

        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
    
    def preprocess_dataframe(self):
        
        col_name = 'token'
        df = pd.read_pickle(self.args.data_path)
        
        if int(self.num_labels) == 4:
            df['y'] = df['y'].apply(make_31)

        # adj
        dic = pd.read_pickle(self.args.dic_feature_path)
        adj_dict = np.load(self.args.adj_path)
        self.dic_feature = torch.tensor(dic['feature'].tolist())
        self.dic_dic = torch.tensor(np.load(self.args.dic_dic) ,dtype=torch.double)
        
        #add token
        words = dic['lexicon'].tolist()
        print("vocab size (before) : ", len(self.tokenizer))
        for w in words:
            self.tokenizer.add_tokens(w, special_tokens=True)
        print("vocab size (after) : ", len(self.tokenizer))
        self.bert_data.resize_token_embeddings(len(self.tokenizer))
        

        
        X_train, X_test, y_train, y_test = train_test_split(
                range(len(df)), df['y'].tolist(),
                test_size=0.2, 
                random_state = self.seed,
                stratify=df['y'].tolist())
        
        self.train_data = TensorDataset(
            torch.tensor(df[col_name].iloc[X_train].tolist(), dtype=torch.long),
            torch.tensor(adj_dict[X_train], dtype=torch.double),
            torch.tensor(y_train, dtype=torch.long),
        )
        
        self.test_data = TensorDataset(
            torch.tensor(df[col_name].iloc[X_test].tolist(), dtype=torch.long),
            torch.tensor(adj_dict[X_test], dtype=torch.double),
            torch.tensor(y_test, dtype=torch.long),
        )
    def train_dataloader(self):
        
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.args.cpu_workers,
        )

    def test_dataloader(self):

        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )
    
    def training_step(self, batch, batch_idx):
        token, adj, labels = batch  
        logits = self(token, adj)    
        loss = None        
        loss = loss_function(logits, labels, self.loss_type, self.num_labels, 1.8)
            
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        token, adj, labels = batch  
        logits = self(token, adj)    
        loss = None        
        loss = loss_function(logits, labels, self.loss_type, self.num_labels, 1.8)
        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def test_epoch_end(self, outputs):
        loss = torch.tensor(0, dtype=torch.float)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        _loss = loss / len(outputs)
        loss = float(_loss)
        y_true = []
        y_pred = []

        for i in outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']

        # save:   
        predict_dict['y_pred'] = y_pred
        predict_dict['y_true'] = y_true
            
        y_pred = np.asanyarray(y_pred)
        y_true = np.asanyarray(y_true)

        m = gr_metrics(y_pred, y_true)
        classwise_FScores = class_FScore(y_pred, y_true, self.num_labels)
        
        metrics_dict['Precision'] = [m[0]]
        metrics_dict['Recall'] = [m[1]]
        metrics_dict['FScore'] = [m[2]]
        metrics_dict['OE']= [m[3]]
        
        tensorboard_logs = {
            'val_loss': loss,
            'val_precision': m[0],
            'val_recall': m[1],
            'val_f1': m[2],
            'val_OE': m[3]
        }
        
        pprint(tensorboard_logs)
        return {'loss': _loss, 'log': tensorboard_logs}

    
def main(config,setting,seed):
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", setting.random_seed)
    seed_everything(setting.random_seed)
        
    model = Model(config,setting,seed) 
    model.preprocess_dataframe()
    logger = TensorBoardLogger(
    save_dir=setting.log_dir,
    version=setting.version,
    name=setting.log_name
    )

    print(":: Start Training ::")
        
    trainer = Trainer(
        logger = logger,
        max_epochs=setting.epochs,
        fast_dev_run=setting.test_mode,
        num_sanity_val_steps=None if setting.test_mode else 0,
        deterministic=True, # ensure full reproducibility from run to run you need to set seeds for pseudo-random generators,
        # For GPU Setup
        gpus=[config['gpu']] if torch.cuda.is_available() else None,
        precision=16 if setting.fp16 else 32
    )
    trainer.fit(model)
    trainer.test(model,test_dataloaders=model.test_dataloader())
    

if __name__ == '__main__': 

    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--num_labels", type=int, default=5,help="expt type")
    parser.add_argument("--split_seed", type=int, default=2021,help="split_seed")
    parser.add_argument("--loss", type=str, default='OE', help="loss")
    parser.add_argument("--dropout", type=float, default=0.1,help="dropout probablity")
    parser.add_argument("--lr", type=float, default=3e-5,help="learning rate")
    parser.add_argument("--agg", type=str, default='cnn', help="loss")
    parser.add_argument("--fanout", type=str, default='115,144,6,39', help="loss")
    parser.add_argument("--hidden", type=int, default=384, help="loss")
    parser.add_argument("--s_drop", type=float, default=0.0, help="loss")
    parser.add_argument("--dic_hidden", type=int, default=85, help="loss")
    parser.add_argument("--kernel_out", type=int, default=50, help="loss")
    parser.add_argument("--gpu", type=int, default=1, help="loss")

    args = parser.parse_args()
    print(args)
    setting = Arg()
    
    
    metrics_dict = {}
    predict_dict = {}
    
    start = time.time()
    main(args.__dict__,setting,args.split_seed)        
    end = time.time()
    
    print("time: ", end - start)

    