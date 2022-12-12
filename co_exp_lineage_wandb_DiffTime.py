# %%
import gc
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import networkx as nx
import random
import os
import json
import wandb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import torch.nn as nn
import torch.nn.functional as F
import time
import seaborn as sns
from torchmetrics import AUROC
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import ruptures as rpt
# %%
hyperparameter_defaults = dict(
    model = 'DiffTime',
    seed=5,
    dataset='BF_2000_1',
    epochs=60,
    lr=0.0005,#1e-4,
    batch_size=512,
    embed_dim = 128,
    output_dim = 128,
    h1_dim = 100,
    h2_dim = 100,
    prod_dim = 100,
    embed_max_norm = 1.0,
    checkpoint_frequency = 5,
    activation = 'relu',
    scaler = 'none',
    num_add_layer = 0, #1, 2
)
# %%
run = wandb.init(
    config=hyperparameter_defaults, project="co_exp_final", entity="chloewxq", reinit=True
)
# %%
config = wandb.config
# %%
wandb.define_metric("val_loss", summary="min", step_metric="epoch")

# %%
torch.manual_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)

path = './models/{}_{}_{}_{}_{}'.format(config.dataset, config.embed_dim, config.h1_dim, config.model, time.strftime('%b%d-%H-%M'))
os.makedirs(path)

# %%
if config.dataset == 'LI_2000_1':
    exp_data = pd.read_csv('/home/chloe/coexp_change/datasets/dyn-LI-2000-1/ExpressionData.csv', index_col=0).T
    time_data = pd.read_csv('/home/chloe/coexp_change/datasets/dyn-LI-2000-1/PseudoTime.csv', index_col=0)
elif config.dataset == 'LL_2000_1':
    exp_data = pd.read_csv('/home/chloe/coexp_change/datasets/dyn-LL-2000-1/ExpressionData.csv', index_col=0).T
    time_data = pd.read_csv('/home/chloe/coexp_change/datasets/dyn-LL-2000-1/PseudoTime.csv', index_col=0)
elif config.dataset == 'BF_2000_1':
    exp_data = pd.read_csv('/home/chloe/coexp_change/datasets/dyn-BF-2000-1/ExpressionData.csv', index_col=0).T
    time_data = pd.read_csv('/home/chloe/coexp_change/datasets/dyn-BF-2000-1/PseudoTime.csv', index_col=0)
elif config.dataset == 'TF_2000_1':
    exp_data = pd.read_csv('/home/chloe/coexp_change/datasets/dyn-TF-2000-1/ExpressionData.csv', index_col=0).T
    exp_data = pd.read_csv('/home/chloe/coexp_change/datasets/dyn-TF-2000-1/ExpressionData.csv', index_col=0).T
    time_data = pd.read_csv('/home/chloe/coexp_change/datasets/dyn-TF-2000-1/PseudoTime.csv', index_col=0)


if config.dataset in ['LI_2000_1', 'LL_2000_1']:
    data = exp_data.join(time_data).sort_values(by='PseudoTime')
    data['Lineage'] = 0
elif config.dataset in ['BF_2000_1', 'TF_2000_1']:
    data = exp_data.join(time_data)
    sub_data_list = []
    ptime_cols = data.filter(regex=("PseudoTime*")).columns
    for i, c in enumerate(ptime_cols):
        data_i = data[~data[c].isnull()]
        data_i['PseudoTime'] = data_i[c].copy()
        data_i = data_i.drop(columns=ptime_cols)
        data_i['Lineage'] = i
        sub_data_list.append(data_i)
    data = pd.concat(sub_data_list).sort_values(by=['Lineage', 'PseudoTime'])

# %%
if config.dataset == 'LI_2000_1':
    edge = [('g1', 'g2'), 
    ('g2', 'g3'),
    ('g3', 'g4'),
    ('g4', 'g5'),
    ('g5', 'g6'),
    ('g6', 'g7'),
    ('g7', 'g7'),
    ('g7', 'g1')]
elif config.dataset == 'LL_2000_1':
    edge = [
    ('g16', 'g17'),
    ('g1', 'g2'),
    ('g17', 'g18'),
    ('g18', 'g18'),
    ('g4', 'g5'),
    ('g3', 'g4'),
    ('g15', 'g16'),
    ('g13', 'g14'),
    ('g7', 'g8'),
    ('g6', 'g7'),
    ('g11', 'g12'),
    ('g9', 'g10'),
    ('g5', 'g6'),
    ('g14', 'g15'),
    ('g12', 'g13'),
    ('g18', 'g1'),
    ('g8', 'g9'),
    ('g2', 'g3'),
    ('g10', 'g11')
    ]
elif config.dataset == 'BF_2000_1':
    edge = [
    ('g4', 'g1'),
    ('g6', 'g1'),
    ('g3', 'g4'),
    ('g4', 'g4'),
    ('g6', 'g4'),
    ('g2', 'g3'),
    ('g4', 'g8'),
    ('g4', 'g7'),
    ('g1', 'g2'),
    ('g6', 'g6'),
    ('g3', 'g6'),
    ('g4', 'g6')
    ]
elif config.dataset == 'TF_2000_1':
    edge = [
    ('g6', 'g7'),
    ('g1', 'g2'),
    ('g3', 'g6'),
    ('g6', 'g6'),
    ('g7', 'g6'),
    ('g5', 'g6'),
    ('g4', 'g6'),
    ('g3', 'g4'),
    ('g4', 'g4'),
    ('g5', 'g4'),
    ('g6', 'g4'),
    ('g5', 'g5'),
    ('g3', 'g5'),
    ('g4', 'g5'),
    ('g6', 'g5'),
    ('g4', 'g1'),
    ('g5', 'g1'),
    ('g6', 'g1'),
    ('g2', 'g3'),
    ('g4', 'g8')
    ]

G = nx.DiGraph()
G.add_edges_from(edge)

# %%
G

# %%
plt.figure()
fig, axes = plt.subplots(nrows = int((len(data.columns)-2)/3)+1, ncols = 3)
axes = axes.flatten()         
fig.set_size_inches(15, 15)
for ax, col in zip(axes, data.drop(columns=['PseudoTime']).columns):
  sns.distplot(data[col], ax = ax)
  ax.set_title(col)
plt.savefig(os.path.join(path, "gene_exp_dist.png"))
wandb.log(
    {
        "gene_exp_dist": wandb.Image(
            str(os.path.join(path, "gene_exp_dist.png")),
            caption="Gene Expression Distribution",
        )
    }
)
# %%
data_s = data.copy()
scaler_dict = {}
if config.scaler != 'none':
    for c in data_s.columns:
        if config.scaler == 'standard':
            scaler = StandardScaler()
        elif config.scaler == 'minmax':
            scaler = MinMaxScaler()
        data_c = np.expand_dims(data[c], 1)
        scaler.fit(data_c)
        scaler_dict[c] = scaler
        data_s[c] = scaler.transform(data_c)
# %%

# %%
out = []
for i, row in tqdm(data_s.iterrows(), total=data.shape[0]):
    curr_time = row['PseudoTime']
    curr_lin = row['Lineage']
    exp = data.loc[i, :].index[:-2]
    for comb in list(itertools.combinations(exp, 2)):
        gene0_val = row[comb[0]]
        gene1_val = row[comb[1]]
        if data.loc[i, comb[0]] > 0.1 and data.loc[i, comb[1]] > 0.1:
            if (comb[0], comb[1]) in edge:
                out.append((comb[0], comb[1], gene0_val, gene1_val, 1, 1, curr_time, curr_lin))
            else:
                out.append((comb[0], comb[1], gene0_val, gene1_val, 1, 0, curr_time, curr_lin))
            #if (comb[1], comb[0]) in edge:
            #    out.append((comb[1], comb[0], gene1_val, gene0_val, 1, 1, curr_time, curr_lin))
            #else:
            #    out.append((comb[1], comb[0], gene1_val, gene0_val, 1, 0, curr_time, curr_lin))
        else:
            if (comb[0], comb[1]) in edge:
                out.append((comb[0], comb[1], gene0_val, gene1_val, 0, 1, curr_time, curr_lin))
            else:
                out.append((comb[0], comb[1], gene0_val, gene1_val, 0, 0, curr_time, curr_lin))
            #if (comb[1], comb[0]) in edge:
            #    out.append((comb[1], comb[0], gene1_val, gene0_val, 0, 1, curr_time, curr_lin))
            #else:
            #    out.append((comb[1], comb[0], gene1_val, gene0_val, 0, 0, curr_time, curr_lin))

# %%
print(len(out))

# %%
df_coexp = pd.DataFrame(out, columns=['input', 'output', 'input_val', 'output_val', 'label', 'is_edge', 'time', 'lineage'])
df_coexp.head()

# %%
print(len(df_coexp[df_coexp['label']==1])/len(df_coexp))

# %%
vocab_list = list(set(df_coexp.input).union(set(df_coexp.output)))

# %%
vocab_dict = dict(zip(vocab_list, range(len(vocab_list))))

# %%
vocab_dict

# %%
vocab_dict_r = dict(zip(range(len(vocab_list)), vocab_list))
vocab_dict_r
pd.DataFrame.from_dict(vocab_dict_r, orient='index', columns=['gene']).to_csv(os.path.join(path, 'vocab.csv'))

# %%
df_coexp['input_idx'] = df_coexp['input'].map(vocab_dict)
df_coexp['output_idx'] = df_coexp['output'].map(vocab_dict)

# %%
df_coexp

# %%
class DiffTime(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size, embed_dim, output_dim, h1_dim, h2_dim, prod_dim, embed_max_norm=1.0):
        super(DiffTime, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.prod_dim = prod_dim
        self.embed_max_norm = embed_max_norm

        # Gene embeddings
        # self.targ_embeddings = nn.Embedding(
        #     num_embeddings=vocab_size,
        #     embedding_dim=embed_dim,
        #     max_norm=embed_max_norm,
        # )

        self.cont_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            max_norm=embed_max_norm,
        )

        # Encode exp val as a function of time
        self.time_vec_h1 = nn.Linear(
            in_features=2,
            out_features=self.h1_dim,
        )

        if config.num_add_layer == 1:
            self.time_vec_h2 = nn.Linear(
                in_features=self.h1_dim,
                out_features=self.h1_dim,
            )
        elif config.num_add_layer == 2:
            self.time_vec_h2 = nn.Linear(
                in_features=self.h1_dim,
                out_features=self.h1_dim,
            )

            self.time_vec_h3 = nn.Linear(
                in_features=self.h1_dim,
                out_features=self.h1_dim,
            )

        self.time_vec_h4 = nn.Linear(
            in_features=self.h1_dim,
            out_features=self.h2_dim,
        )

        # Encode emb to matrix
        self.emb_to_mat = nn.Linear(
            in_features=self.embed_dim,
            out_features=self.h2_dim * self.prod_dim,
        )

        # Last linear layer
        self.linear = nn.Linear(
            in_features=self.prod_dim,
            out_features=self.output_dim,
        )
        
        self.m = nn.Sigmoid()
        if config.activation == 'relu':
            self.r = nn.ReLU()
        elif config.activation == 'tanh':
            self.r = nn.Tanh()

    def forward(self, inputs, targets, t, lineage):
        # Encode time and context gene identity in a vector t_i
        t_i_cont = torch.cat([t.unsqueeze(1), lineage.unsqueeze(1)], 1).unsqueeze(1)
        t_i_cont = self.r(self.time_vec_h1(t_i_cont))
        if config.num_add_layer == 1:
            t_i_cont = self.r(self.time_vec_h2(t_i_cont))
        if config.num_add_layer == 2:
            t_i_cont = self.r(self.time_vec_h2(t_i_cont))
            t_i_cont = self.r(self.time_vec_h3(t_i_cont))
        t_i_cont = self.r(self.time_vec_h4(t_i_cont))

        # Encode cont emb
        x_cont = self.cont_embeddings(inputs)
        x_cont = self.emb_to_mat(x_cont)
        x_cont = torch.reshape(x_cont, (-1, self.prod_dim, self.h2_dim))
        matmul_c = torch.matmul(x_cont, t_i_cont.squeeze().unsqueeze(-1)) #exp_cont
        matmul_c = torch.reshape(matmul_c, (-1, self.prod_dim))
        out_c = self.linear(matmul_c)

        # Encode time and target cell identity in a vector t_i
        t_i_tar = torch.cat([t.unsqueeze(1), lineage.unsqueeze(1)], 1).unsqueeze(1)
        t_i_tar = self.r(self.time_vec_h1(t_i_tar))
        if config.num_add_layer == 1:
            t_i_tar = self.time_vec_h2(t_i_tar)
        if config.num_add_layer == 2:
            t_i_tar = self.time_vec_h2(t_i_tar)
            t_i_tar = self.time_vec_h3(t_i_tar)
        t_i_tar = self.time_vec_h4(t_i_tar)

        # Encode cont emb
        x_tar = self.cont_embeddings(targets)
        x_tar = self.emb_to_mat(x_tar)
        x_tar = torch.reshape(x_tar, (-1, self.prod_dim, self.h2_dim))
        matmul_t = torch.matmul(x_tar, t_i_tar.squeeze().unsqueeze(-1)) #exp_tar
        matmul_t = torch.reshape(matmul_t, (-1, self.prod_dim))
        out_t = self.linear(matmul_t)

        logits = self.m(torch.sum(torch.multiply(out_c, out_t), 1))

        return out_c, out_t, logits

# %%
df_coexp

# %%
class Dataset(torch.utils.data.Dataset):
  def __init__(self, data_df):
        self.idx = list(range(len(data_df)))
        self.data_df = data_df

  def __len__(self):
        return len(self.idx)

  def __getitem__(self, index):
        x = torch.tensor(self.data_df.iloc[index]['input_idx'])
        y = torch.tensor(self.data_df.iloc[index]['output_idx'])
        d = torch.tensor(self.data_df.iloc[index]['time']).double()
        l = torch.tensor(self.data_df.iloc[index]['label'])
        exp_x = torch.tensor(self.data_df.iloc[index]['input_val'])
        exp_y = torch.tensor(self.data_df.iloc[index]['output_val'])
        is_edge = torch.tensor(self.data_df.iloc[index]['is_edge'])
        lineage = torch.tensor(self.data_df.iloc[index]['lineage'])
        return x, y, d, l, exp_x, exp_y, is_edge, lineage


# %%
train_df, valid_df = train_test_split(df_coexp, test_size=0.1, random_state=config.seed, shuffle=True)

# %%
train_set = Dataset(train_df)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)

# %%
valid_set = Dataset(valid_df)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=config.batch_size, shuffle=True)

# %%
class Trainer():
    """Main class for model training"""
    
    def __init__(
        self,
        model,
        epochs,
        train_dataloader,
        val_dataloader,
        checkpoint_frequency,
        criterion,
        optimizer,
        lr_scheduler,
        device,
        model_dir,
    ):  
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_frequency = checkpoint_frequency
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model = model
        self.model_dir = model_dir

        self.loss = {"train": [], 
        "val": [], 
        "train_pred": [], 
        "train_MSE": [], 
        "train_edge_MSE": [],
        "train_auc": [],
        "train_precision": [],
        "train_recall": [],
        "val_pred": [], 
        "val_MSE": [], 
        "val_edge_MSE": [],
        "val_auc": [],
        "val_precision": [],
        "val_recall": [],}
        self.model.to(self.device)

    def train(self):
        best_val_loss = 999999
        best_model = None
        for epoch in range(self.epochs):
            self._train_epoch()
            self._validate_epoch()
            print(
                "Epoch: {}/{}, Train Loss={:.5f}, Train Pred Loss={:.5f}, Train AUC={:.5f}, Train Precision={:.5f}, Train Recall={:.5f}".format(
                    epoch + 1,
                    self.epochs,
                    self.loss["train"][-1],
                    self.loss["train_pred"][-1],
                    self.loss["train_auc"][-1],
                    self.loss["train_precision"][-1],
                    self.loss["train_recall"][-1],
                )
            )

            print(
                "Epoch: {}/{}, Valid Loss={:.5f}, Valid Pred Loss={:.5f}, Valid AUC={:.5f}, Valid Precision={:.5f}, Valid Recall={:.5f}".format(
                    epoch + 1,
                    self.epochs,
                    self.loss["val"][-1],
                    self.loss["val_pred"][-1],
                    self.loss["val_auc"][-1],
                    self.loss["val_precision"][-1],
                    self.loss["val_recall"][-1],
                )
            )

            wandb.log(
                {
                "epoch": epoch+1,
                "train_loss": self.loss["train"][-1],
                "train_pred_loss": self.loss["train_pred"][-1],
                "train_auc": self.loss["train_auc"][-1],
                "train_precision": self.loss["train_precision"][-1],
                "train_recall": self.loss["train_recall"][-1],
                "val_loss": self.loss["val"][-1],
                "val_pred_loss": self.loss["val_pred"][-1],
                "val_auc": self.loss["val_auc"][-1],
                "val_precision": self.loss["val_precision"][-1],
                "val_recall": self.loss["val_recall"][-1],
                })
            self.lr_scheduler.step()

            if (epoch+1) % self.checkpoint_frequency == 0 and self.loss["val"][-1] < best_val_loss:
                print('Saved best model')
                best_val_loss = self.loss["val"][-1]
                best_model = self.model
                best_model_path = self._save_checkpoint(epoch)
                test_metrics(best_model, self.model_dir)


        return best_model_path

    def _train_epoch(self):
        self.model.train()
        running_loss = []
        running_loss_pred = []
        running_auc = []
        running_precision = []
        running_recall = []

        for i, batch_data in enumerate(tqdm(self.train_dataloader)):
            # inputs, targets, exp_inputs, exp_targets, t
            inputs = batch_data[0].to(self.device)
            targets = batch_data[1].to(self.device)
            t = batch_data[2].double().to(self.device)
            labels = batch_data[3].double().to(self.device)
            exp_inputs = batch_data[4].double().to(self.device)
            exp_targets = batch_data[5].double().to(self.device)
            is_edge = batch_data[6].double().to(self.device)
            lineage = batch_data[7].double().to(self.device)

            out_c, out_t, logits = self.model(inputs, targets, t, lineage)
            loss_pred = self.criterion[0](logits, labels) 
            loss = loss_pred

            auc = roc_auc_score(
                labels.long().detach().cpu(), 
                logits.detach().cpu())

            precision = precision_score(
                labels.long().detach().cpu(), 
                1*(logits.detach().cpu() > 0.5))

            recall = recall_score(
                labels.long().detach().cpu(), 
                1*(logits.detach().cpu() > 0.5))

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())
            running_loss_pred.append(loss_pred.item())
            running_auc.append(auc.item())
            running_precision.append(precision)
            running_recall.append(recall)

        epoch_loss = np.mean(running_loss)
        epoch_train_pred_loss = np.mean(running_loss_pred)
        epoch_auc = np.mean(running_auc)
        epoch_precision = np.mean(running_precision)
        epoch_recall = np.mean(running_recall)

        self.loss["train"].append(epoch_loss)
        self.loss["train_pred"].append(epoch_train_pred_loss)
        self.loss["train_auc"].append(epoch_auc)
        self.loss["train_precision"].append(epoch_precision)
        self.loss["train_recall"].append(epoch_recall)

    def _validate_epoch(self):
        self.model.eval()
        running_loss = []
        running_loss_pred = []
        running_auc = []
        running_precision = []
        running_recall = []

        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(self.val_dataloader)):
                inputs = batch_data[0].to(self.device)
                targets = batch_data[1].to(self.device)
                t = batch_data[2].double().to(self.device)
                labels = batch_data[3].double().to(self.device)
                exp_inputs = batch_data[4].double().to(self.device)
                exp_targets = batch_data[5].double().to(self.device)
                is_edge = batch_data[6].double().to(self.device)
                lineage = batch_data[7].double().to(self.device)

                out_c, out_t, logits = self.model(inputs, targets, t, lineage)
                #loss = self.criterion[0](logits, labels)
                #+ self.criterion[1](out_c_exp.squeeze(), exp_inputs)
                #+ self.criterion[1](out_t_exp.squeeze(), exp_targets)
                #+ self.criterion[1](torch.matmul(is_edge, out_t_exp).squeeze(), torch.matmul(is_edge, tar_est).squeeze())
                loss_pred = self.criterion[0](logits, labels)
                loss = loss_pred #+ loss_MSE #+ loss_edge

                auc = roc_auc_score(
                    labels.long().detach().cpu(), 
                    logits.detach().cpu())

                precision = precision_score(
                    labels.long().detach().cpu(), 
                    1*(logits.detach().cpu() > 0.5))

                recall = recall_score(
                    labels.long().detach().cpu(), 
                    1*(logits.detach().cpu() > 0.5))

                running_loss.append(loss.item())
                running_loss_pred.append(loss_pred.item())
                running_auc.append(auc.item())
                running_precision.append(precision)
                running_recall.append(recall)

        epoch_loss = np.mean(running_loss)
        epoch_train_pred_loss = np.mean(running_loss_pred)
        epoch_auc = np.mean(running_auc)
        epoch_precision = np.mean(running_precision)
        epoch_recall = np.mean(running_recall)

        self.loss["val"].append(epoch_loss)
        self.loss["val_pred"].append(epoch_train_pred_loss)
        self.loss["val_auc"].append(epoch_auc)
        self.loss["val_precision"].append(epoch_precision)
        self.loss["val_recall"].append(epoch_recall)

    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory"""
        epoch_num = epoch + 1
        if epoch_num % self.checkpoint_frequency == 0:
            model_path = "checkpoint_{}.pt".format(str(epoch_num).zfill(3))
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)
        return model_path

    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)

# %%
def cos_time_series(data):
    word0 = np.expand_dims(data[0, :], 0)
    cos_sim = cosine_similarity(word0, data)
    return 1-cos_sim
    
def cos_time_series_1(data):
    cos_sim = cosine_similarity(data, data)
    cos_sim_ = np.diag(cos_sim, k=1)
    ret = np.insert(1-cos_sim_, 0, 0)
    return ret

def test_metrics(model, path):
    for lineage in data['Lineage'].unique():
        for gene in vocab_dict.keys():
            data_re = data[data['Lineage']==lineage].groupby('PseudoTime').mean().reset_index()
            time = data_re['PseudoTime']
            idx = df_vocab[df_vocab['gene']==gene].index[0]
            #if config.scaler != 'none':
            #    exp0 = scaler_dict[gene0].transform(np.expand_dims(data_re[gene0], 1))
            #    exp1 = scaler_dict[gene1].transform(np.expand_dims(data_re[gene1], 1))
            #else:
            time = data_re['PseudoTime'].to_list()
            idx_ = torch.tensor([idx]).tile((len(time),1)).squeeze().to(device)
            if config.scaler != 'none':
                time_ = torch.tensor(scaler_dict['PseudoTime'].transform(np.expand_dims(data_re['PseudoTime'], 1))).double().squeeze().to(device)
                lineage_ = torch.tensor(scaler_dict['Lineage'].transform(np.expand_dims(data_re['Lineage'], 1))).double().squeeze().to(device)
            else:
                time_ = torch.tensor(time).double().squeeze().to(device)
                lineage_ = torch.tensor(data_re['Lineage']).double().squeeze().to(device)

            out_c, _, _ = model(idx_, idx_, time_, lineage_)
            out_np_co_exp_c = out_c.cpu().detach().numpy().squeeze()
            time_series_c = cos_time_series(out_np_co_exp_c)
            df_time_series = pd.DataFrame()
            df_time_series['time'] = time
            df_time_series['cos_sim_diff'] = time_series_c.tolist()[0]
            plot_change_point(df_time_series, 
            path,
            title='{} Co-exp Ebm over Time (L{})'.format(gene, lineage))
        #if config.scaler != 'none':
        #    out_np_exp_c = scaler_dict[gene0].inverse_transform(exp_c.cpu().detach().numpy())
        #    out_np_exp_t = scaler_dict[gene1].inverse_transform(exp_t.cpu().detach().numpy())
        #    out_np_tar_est = scaler_dict[gene1].inverse_transform(tar_est.cpu().detach().numpy())
        #else:
        #    out_np_exp_c = exp_c.cpu().detach().numpy()
        #    out_np_exp_t = exp_t.cpu().detach().numpy()
        #    out_np_tar_est = tar_est.cpu().detach().numpy()
        # STEP 1: Plot TSNE of co-expression embeddings
        #plot_TSNE(out_np_co_exp_c, time, exp0, 'context_{}_lineage_{}'.format(gene0, lineage), path)
        #plot_TSNE(out_np_co_exp_t, time, exp1, 'target_{}_lineage_{}'.format(gene1, lineage), path)
        #plot_time_series(out_np_co_exp_c, out_np_co_exp_t, time, "{}+{}_lineage_{}".format(gene0, gene1, lineage), path)
        #plot_exp_pred(out_np_exp_c, exp0, time, 'context_{}_lineage_{}'.format(gene0, lineage), path)
        #plot_exp_pred(out_np_exp_t, exp1, time, 'target_{}_lineage_{}'.format(gene1, lineage), path)
        #plot_coexp_exp(out_np_co_exp_c, out_np_exp_c, time, 'coexp_exp_{}_lineage_{}'.format(gene0, lineage), path, save=True, wandb_log=True)
        #plot_coexp_exp(out_np_co_exp_t, out_np_exp_t, time, 'coexp_exp_{}_lineage_{}'.format(gene1, lineage), path, save=True, wandb_log=True)
        
        #plot_exp_pred(out_np_tar_est, exp1, time, 'target_est_{}_lineage_{}'.format(gene1, lineage), path)


def plot_change_point(df_time_series, path, model_names=['l2', 'rbf'], x='time', y='cos_sim_diff', plot_exp=False, title='', save=True, wandb_log=True):
    sns.set_style("darkgrid")
    sns.set_palette("Set2")
    plt.figure()
    ax = sns.lineplot(
        x=x, 
        y=y,
        data = df_time_series,
        #color='orange'
    )

    if plot_exp:
        ax = sns.scatterplot(
        x=x, 
        y='exp_true',
        data = df_time_series,
    )


    time_series = pd.DataFrame(df_time_series[y])
    df_time_series['text'] = ['t={}'.format(round(i, 2)) for i in df_time_series[x].values] if x=='time' else ['exp={}'.format(round(i, 2)) for i in df_time_series[x].values]

    for model_name in model_names:

        if model_name == "rbf":
            algo = rpt.Pelt(model="rbf").fit(time_series)
            result = algo.predict(pen=10)
            ax = sns.scatterplot(
                x=x, 
                y=y,
                data=df_time_series[df_time_series.index.isin(result)],
                marker="o",
                color="b"
            )
            plt.title('Change Point Detection: Pelt Search Method')

        elif model_name == "l2":
            algo = rpt.Binseg(model="l2").fit(time_series)
            my_bkps = algo.predict(n_bkps=5)
            ax = sns.scatterplot(
                x=x, 
                y=y,
                data=df_time_series[df_time_series.index.isin(my_bkps)],
                marker="X",
                color="r",
                s=100,
            )
            plt.title('Change Point Detection: Binary Segmentation Search Method')
            df_temp = df_time_series[df_time_series.index.isin(my_bkps)]
            for i in df_temp.index:
                plt.text(df_temp.loc[i, x]+0.01, 
                df_temp.loc[i, y], 
                df_temp.loc[i, 'text'], 
                horizontalalignment='left', 
                size='medium', 
                color='black', 
                weight='semibold')
    if len(model_names) > 1:
        plt.title(title, fontweight="bold")
    
    if save:
        fig = ax.get_figure()
        fig.savefig(os.path.join(path, "{}.png".format(title)))
    if wandb_log:
        wandb.log({
        "{}".format(title): wandb.Image(
            str(os.path.join(path, "{}.png".format(title))))})

def plot_TSNE(out_np, time, exp, setting, path, save=True, wandb_log=True):
    tsne = TSNE(n_components=2)
    out_tsne = tsne.fit_transform(out_np)
    df_out_co_exp = pd.DataFrame(out_tsne, columns = ['tsne1', 'tsne2'])
    df_out_co_exp['time'] = time
    df_out_co_exp['exp'] = exp
    plt.figure()
    ax = sns.scatterplot(
        data=df_out_co_exp, 
        x='tsne1', 
        y='tsne2',
        hue='time')
    ax.figure.suptitle('Gene by time {}'.format(setting))
    if save:
        fig = ax.get_figure()
        fig.savefig(os.path.join(path, "gene_embedding_{}.png".format(setting)))
    if wandb_log:
        wandb.log({
        "gene_embedding_{}".format(setting): wandb.Image(
            str(os.path.join(path, "gene_embedding_{}.png".format(setting))),
            caption="Gene Embedding {}".format(setting),)})

def plot_time_series(out_np_co_exp_c, out_np_co_exp_t, time, setting, path, save=True, wandb_log=True):
    time_series_c = cos_time_series(out_np_co_exp_c)
    time_series_t = cos_time_series(out_np_co_exp_t)
    df_time_series = pd.DataFrame()
    df_time_series['time'] = time
    df_time_series['cos_sim_c'] = time_series_c.tolist()[0]
    df_time_series['cos_sim_t'] = time_series_t.tolist()[0]
    plt.figure()
    ax = sns.lineplot(
        x="time", 
        y="cos_sim_c",
        data = df_time_series,
    )
    ax = sns.lineplot(
        x="time", 
        y="cos_sim_t",
        data = df_time_series,
    )
    if save:
        fig = ax.get_figure()
        fig.savefig(os.path.join(path, "co_exp_{}.png".format(setting)))
    if wandb_log:
        wandb.log({
        "co_exp_{}".format(setting): wandb.Image(
            str(os.path.join(path, "co_exp_{}.png".format(setting))),
            caption="Co-Expression {}".format(setting),)})

def plot_coexp_exp(out_np_co_exp, exp, time, setting, path, save=True, wandb_log=True):
    time_series_c = cos_time_series(out_np_co_exp)
    df_time_series = pd.DataFrame()
    df_time_series['exp'] = exp.flatten()
    df_time_series['cos_sim'] = time_series_c.tolist()[0]
    plt.figure()
    ax = sns.scatterplot(
        x="exp", 
        y="cos_sim",
        data = df_time_series
    )
    if save:
        fig = ax.get_figure()
        fig.savefig(os.path.join(path, "co_exp_exp_{}.png".format(setting)))
    if wandb_log:
        wandb.log({
        "co_exp_exp_{}".format(setting): wandb.Image(
            str(os.path.join(path, "co_exp_exp_{}.png".format(setting))),
            caption="Co-Expression over Expression {}".format(setting),)})


def plot_exp_pred(out_np_exp, exp, time, setting, path, save=True, wandb_log=True):
    df_time_series = pd.DataFrame()
    df_time_series['time'] = time
    df_time_series['exp_pred'] = out_np_exp
    df_time_series['exp'] = exp
    plt.figure()
    ax = sns.scatterplot(
        x="time", 
        y="exp",
        data = df_time_series,
    )
    ax = sns.lineplot(
        x="time", 
        y="exp_pred",
        data = df_time_series,
    )
    if save:
        fig = ax.get_figure()
        fig.savefig(os.path.join(path, "pred_exp_{}.png".format(setting)))
    if wandb_log:
        wandb.log({
        "pred_exp_{}".format(setting): wandb.Image(
            str(os.path.join(path, "pred_exp_{}.png".format(setting))),
            caption="Predicted Expression {}".format(setting),)})

# %%
vocab_size = len(vocab_dict)
df_vocab= pd.read_csv('{}/vocab.csv'.format(path))
df_vocab.columns = ['index', 'gene']

model = DiffTime(vocab_size=vocab_size, 
embed_dim=config.embed_dim,
output_dim=config.output_dim,
h1_dim=config.h1_dim,
h2_dim=config.h2_dim,
prod_dim=config.prod_dim,
embed_max_norm=config.embed_max_norm,
).type(torch.DoubleTensor)

criterion = [
    nn.BCELoss(),
    #nn.BCEWithLogitsLoss(), 
    nn.MSELoss()]

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = Trainer(
    model=model,
    epochs=config.epochs,
    train_dataloader=train_loader,
    val_dataloader=valid_loader,
    criterion=criterion,
    optimizer=optimizer,
    checkpoint_frequency=config.checkpoint_frequency,
    lr_scheduler=scheduler,
    device=device,
    model_dir=path,
)

best_model_path = trainer.train()
print("Training finished.")

artifact = wandb.Artifact(f"best_model", type="model")
artifact.add_file(best_model_path)
run.log_artifact(artifact)

run.finish()
wandb.finish()
gc.collect()
# %%

'''
# %%
tsne = TSNE(n_components=2)
out_tsne = tsne.fit_transform(out_np_co_exp_t)
df_out_co_exp = pd.DataFrame(out_tsne, columns = ['tsne1', 'tsne2'])
#df_out_co_exp['time'] = [str(i) for i in scaler_time.inverse_transform(time)]
df_out_co_exp['time'] = [str(i) for i in time]
#df_out_co_exp['time_num'] = scaler_time.inverse_transform(time)
df_out_co_exp['time_num'] = time
df_out_co_exp['exp'] = data_re[gene0].to_list()
df_out_co_exp
ax1 = sns.scatterplot(
    data=df_out_co_exp, 
    x='tsne1', 
    y='tsne2',
    hue='time_num')
ax1.figure.suptitle('Gene by time - Target')

# %%
def plot_time_series():
    time_series_c = cos_time_series(out_np_co_exp_c)
    time_series_t = cos_time_series(out_np_co_exp_t)
    df_time_series = pd.DataFrame()
    df_time_series['time'] = df_out_co_exp['time_num']
    df_time_series['cos_sim_c'] = time_series_c.tolist()[0]
    df_time_series['cos_sim_t'] = time_series_t.tolist()[0]
    df_time_series['exp_pred'] = out_np_exp_c
#plt.figure(figsize=(16,10))
# blue, orange
    ax = sns.lineplot(
        x="time", 
        y="cos_sim_c",
        data = df_time_series,
    )
    ax = sns.lineplot(
        x="time", 
        y="cos_sim_t",
        data = df_time_series,
    )
    ax.legend()

# %%
def plot_exp_pred(out_np_exp, exp, time)
    df_time_series = pd.DataFrame()
    df_time_series['time'] = time
    df_time_series['exp_pred'] = out_np_exp
    df_time_series['exp'] = exp
    plt.figure()
    ax = sns.scatterplot(
        x="time", 
        y="exp",
        data = df_time_series,
    )
    ax = sns.lineplot(
        x="time", 
        y="exp_pred",
        data = df_time_series,
    )


# %%
time_series = cos_time_series(out_np_co_exp_t)
df_time_series = pd.DataFrame()
df_time_series['time'] = df_out_co_exp['time_num']
df_time_series['cos_sim'] = time_series.tolist()[0]
df_time_series['exp_pred'] = out_np_exp_t
df_time_series['exp_tar_est'] = out_np_tar_est
df_time_series['exp'] = data_re[gene1].to_list()

# %%
ax = sns.scatterplot(
    x="time", 
    y="exp",
    data = df_time_series,
)
ax = sns.lineplot(
    x="time", 
    y="exp_tar_est",
    data = df_time_series,
)
ax.figure.suptitle('Time Series')

# %%
#plt.figure(figsize=(16,10))
ax = sns.scatterplot(
    x="time", 
    y="exp",
    data = df_time_series,
)
ax = sns.lineplot(
    x="time", 
    y="exp_pred",
    data = df_time_series,
)
ax.figure.suptitle('Time Series')

# %%
time_series = cos_time_series(out_np_co_exp)
time_series_1 = cos_time_series_1(out_np_co_exp)
df_time_series = pd.DataFrame()
df_time_series['time'] = df_out_co_exp['time_num']
df_time_series['cos_sim'] = time_series.tolist()[0]
df_time_series['cos_sim_diff'] = time_series_1.tolist()[0]
df_time_series['exp_pred'] = out_np_exp
df_time_series['exp'] = data_re[gene0].to_list()

# %%
#plt.figure(figsize=(16,10))
ax = sns.scatterplot(
    x="exp_pred", 
    y="cos_sim",
    data = df_time_series,
)
ax = sns.scatterplot(
    x="exp_pred", 
    y="cos_sim_diff",
    data = df_time_series,
)
ax.figure.suptitle('Time Series')


# %%
#ax = sns.scatterplot(
#    x="time", 
#    y="exp",
#    data = df_time_series,
#)
ax = sns.lineplot(
    x="exp", 
    y="tar_est",
    data = df_time_series,
)
ax.figure.suptitle('Time Series')

# %%
out_np_exp.shape

# %%
tsne = TSNE(n_components=2)
df_out_exp = pd.DataFrame()
df_out_exp['time_num'] = time
df_out_exp['exp'] = exp
df_out_exp['exp_pred'] = out_np_exp
df_out_exp
ax = sns.scatterplot(
    data=df_out_exp, 
    x='time_num', 
    y='exp_pred')
ax.figure.suptitle('Gene by exp')

# %%
time_series = cos_time_series(out_np_exp)
df_time_series = pd.DataFrame()
df_time_series['exp'] = df_out_exp['exp']
df_time_series['cos_sim'] = time_series.tolist()[0]
#plt.figure(figsize=(16,10))
ax = sns.lineplot(
    x="exp", 
    y="cos_sim",
    data = df_time_series,
)
ax.figure.suptitle('Time Series')

# %%
data_day = data[data['day']=='2.0'].drop(columns=['day'])

# %%
data_day

# %%
df_corr = data_day.corr('pearson').abs().fillna(0)

# %%
df_corr.data = np.fill_diagonal(df_corr.to_numpy(), 0)

# %%
df_corr

# %%
# Day 18 days by half day, 39 categories
# For each cell, find co-expression within same-day window


# %%
import seaborn as sns
sns.distplot(data_day['Hspe1'])

# %%
sns.distplot(data['Hspe1'])
'''


