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

# %%
hyperparameter_defaults = dict(
    model = 'ExpEstimator',
    seed=5,
    dataset='BF_2000_1',
    epochs=60,
    lr=0.001,#1e-4,
    batch_size=512,
    h1_dim = 128,
    h2_dim = 128,
    checkpoint_frequency = 5,
    activation = 'relu',
    scaler = 'none',
    if_add_layer = True,
    if_out_linear = False,
)
# %%
run = wandb.init(
    config=hyperparameter_defaults, project="co_exp", entity="chloewxq", reinit=True
)
# %%
config = wandb.config
# %%
wandb.define_metric("val_auc", summary="max", step_metric="epoch")

# %%
torch.manual_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)

path = './models/{}_{}_{}_{}'.format(config.dataset, config.h1_dim, config.model, time.strftime('%b%d-%H-%M'))
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
    for gene in exp:
        out.append([gene, row[gene], curr_time, curr_lin])
# %%
print(len(out))

# %%
# comb[1], comb[0], gene1_val, gene0_val, 0, 0, curr_time)
df_exp = pd.DataFrame(out, columns=['input',  'input_val', 'time', 'lineage'])
df_exp.head()

# %%
vocab_list = list(set(df_exp.input))

# %%
vocab_dict = dict(zip(vocab_list, range(len(vocab_list))))

# %%
vocab_dict

# %%
vocab_dict_r = dict(zip(range(len(vocab_list)), vocab_list))
vocab_dict_r
pd.DataFrame.from_dict(vocab_dict_r, orient='index', columns=['gene']).to_csv(os.path.join(path, 'vocab.csv'))

# %%
df_exp['input_idx'] = df_exp['input'].map(vocab_dict)

# %%
df_exp

# %%
class ExpEstimator(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, h1_dim, h2_dim):
        super(ExpEstimator, self).__init__()
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim

        # Encode exp val as a function of time
        self.time_vec_h1 = nn.Linear(
            in_features=3,
            out_features=self.h1_dim,
        )

        if config.if_add_layer:
            self.time_vec_h2 = nn.Linear(
                in_features=self.h1_dim,
                out_features=self.h1_dim,
            )

        self.time_vec_h3 = nn.Linear(
            in_features=self.h1_dim,
            out_features=self.h2_dim,
        )

        self.exp_vec_h1 = nn.Linear(
                in_features=self.h2_dim,
                out_features=self.h1_dim,
        )

        if config.if_add_layer:
            self.exp_vec_h2 = nn.Linear(
                in_features=self.h1_dim,
                out_features=self.h1_dim,
            )

        self.exp_vec_h3 = nn.Linear(
            in_features=self.h1_dim,
            out_features=1,
        )

        if config.activation == 'relu':
            self.r = nn.ReLU()
        elif config.activation == 'tanh':
            self.r = nn.Tanh()

    def forward(self, inputs, t, lineage):
        # Encode time and context gene identity in a vector t_i
        t_i_cont = torch.cat([t.unsqueeze(1), inputs.unsqueeze(1), lineage.unsqueeze(1)], 1).unsqueeze(1)
        t_i_cont = self.r(self.time_vec_h1(t_i_cont))
        if config.if_add_layer:
            t_i_cont = self.r(self.time_vec_h2(t_i_cont))

        t_i_cont = self.r(self.time_vec_h3(t_i_cont))
        # Encode context exp value as a function of time and context gene identity
        exp_cont = self.r(self.exp_vec_h1(t_i_cont))

        if config.if_add_layer:
            exp_cont = self.r(self.exp_vec_h2(exp_cont))
        
        if config.if_out_linear:
            exp_cont = self.exp_vec_h3(exp_cont)
            exp_cont_out = exp_cont.squeeze(-1)
            exp_cont = self.r(exp_cont)
        else:
            exp_cont = self.r(self.exp_vec_h3(exp_cont))
            exp_cont_out = exp_cont.squeeze(-1)
        
        return exp_cont_out

# %%
df_exp

# %%
class Dataset(torch.utils.data.Dataset):
  def __init__(self, data_df):
        self.idx = list(range(len(data_df)))
        self.data_df = data_df

  def __len__(self):
        return len(self.idx)

  def __getitem__(self, index):
        x = torch.tensor(self.data_df.iloc[index]['input_idx'])
        d = torch.tensor(self.data_df.iloc[index]['time']).double()
        exp_x = torch.tensor(self.data_df.iloc[index]['input_val'])
        lineage = torch.tensor(self.data_df.iloc[index]['lineage'])
        return x, d, exp_x, lineage


# %%
train_df, valid_df = train_test_split(df_exp, test_size=0.1, random_state=config.seed, shuffle=True)

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
                "Epoch: {}/{}, Train Loss={:.5f}, Train MSE Loss={:.5f}".format(
                    epoch + 1,
                    self.epochs,
                    self.loss["train"][-1],
                    self.loss["train_MSE"][-1],
                )
            )

            print(
                "Epoch: {}/{}, Valid Loss={:.5f}, Valid MSE Loss={:.5f}".format(
                    epoch + 1,
                    self.epochs,
                    self.loss["val"][-1],
                    self.loss["val_MSE"][-1],
                )
            )

            wandb.log(
                {
                "epoch": epoch+1,
                "train_loss": self.loss["train"][-1],
                "train_MSE_loss": self.loss["train_MSE"][-1],
                "val_loss": self.loss["val"][-1],
                "val_MSE_loss": self.loss["val_MSE"][-1],
                })
            self.lr_scheduler.step()

            if (epoch+1) % self.checkpoint_frequency == 0 and self.loss["val"][-1] < best_val_loss:
                print('Saved best model')
                best_val_loss = self.loss["val"][-1]
                best_model = self.model
                best_model_path = self._save_checkpoint(epoch)
                for gene in vocab_dict.keys():
                    test_metrics(best_model, self.model_dir, gene=gene)
                    
        return best_model_path

    def _train_epoch(self):
        self.model.train()
        running_loss = []
        running_loss_MSE = []

        for i, batch_data in enumerate(tqdm(self.train_dataloader)):
            # x, d, exp_x, lineage
            inputs = batch_data[0].to(self.device)
            t = batch_data[1].double().to(self.device)
            exp_inputs = batch_data[2].double().to(self.device)
            lineage = batch_data[3].double().to(self.device)

            out_c_exp = self.model(inputs, t, lineage)

            loss_MSE = self.criterion[1](out_c_exp.squeeze(), exp_inputs)
            loss = loss_MSE

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())
            running_loss_MSE.append(loss_MSE.item())

        epoch_loss = np.mean(running_loss)
        epoch_train_MSE_loss = np.mean(running_loss_MSE)

        self.loss["train"].append(epoch_loss)
        self.loss["train_MSE"].append(epoch_train_MSE_loss)


    def _validate_epoch(self):
        self.model.eval()
        running_loss = []
        running_loss_MSE = []

        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(self.val_dataloader)):
                # x, d, exp_x, lineage
                inputs = batch_data[0].to(self.device)
                t = batch_data[1].double().to(self.device)
                exp_inputs = batch_data[2].double().to(self.device)
                lineage = batch_data[3].double().to(self.device)

                out_c_exp = self.model(inputs, t, lineage)
                loss_MSE = self.criterion[1](out_c_exp.squeeze(), exp_inputs)
                loss = loss_MSE

                running_loss.append(loss.item())
                running_loss_MSE.append(loss_MSE.item())


        epoch_loss = np.mean(running_loss)
        epoch_train_MSE_loss = np.mean(running_loss_MSE)

        self.loss["val"].append(epoch_loss)
        self.loss["val_MSE"].append(epoch_train_MSE_loss)

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

def test_metrics(model, path, gene):
    for lineage in data['Lineage'].unique():
        data_re = data[data['Lineage']==lineage].groupby('PseudoTime').mean().reset_index()
        time = data_re['PseudoTime']
        # Prepare data
        idx = df_vocab[df_vocab['gene']==gene].index[0]
        #if config.scaler != 'none':
        #    exp0 = scaler_dict[gene0].transform(np.expand_dims(data_re[gene0], 1))
        #    exp1 = scaler_dict[gene1].transform(np.expand_dims(data_re[gene1], 1))
        #else:
        exp = data_re[gene].to_list()
        time = data_re['PseudoTime'].to_list()
        idx_ = torch.tensor([idx]).tile((len(time),1)).squeeze().to(device)
        if config.scaler != 'none':
            time_ = torch.tensor(scaler_dict['PseudoTime'].transform(np.expand_dims(data_re['PseudoTime'], 1))).double().squeeze().to(device)
            lineage_ = torch.tensor(scaler_dict['Lineage'].transform(np.expand_dims(data_re['Lineage'], 1))).double().squeeze().to(device)
        else:
            time_ = torch.tensor(time).double().squeeze().to(device)
            lineage_ = torch.tensor(data_re['Lineage']).double().squeeze().to(device)
        exp_c = model(idx_, time_, lineage_)
       
        if config.scaler != 'none':
            out_np_exp_c = scaler_dict[gene].inverse_transform(exp_c.cpu().detach().numpy())
        else:
            out_np_exp_c = exp_c.cpu().detach().numpy()
        # STEP 1: Plot TSNE of co-expression embeddings
        #plot_TSNE(out_np_co_exp_c, time, exp0, 'context_{}_lineage_{}'.format(gene0, lineage), path)
        #plot_TSNE(out_np_co_exp_t, time, exp1, 'target_{}_lineage_{}'.format(gene1, lineage), path)
        plot_exp_pred(out_np_exp_c, exp, time, 'exp_{}_lineage_{}'.format(gene, lineage), path)
        #plot_exp_pred(out_np_tar_est, exp1, time, 'target_est_{}_lineage_{}'.format(gene1, lineage), path)


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

#h1_dim, h2_dim
model = ExpEstimator(h1_dim=config.h1_dim,
h2_dim=config.h2_dim,
).type(torch.DoubleTensor)

criterion = [
    nn.BCELoss(),
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


