import torch
from math import ceil
import numpy as np
import argparse
import os
import os.path as osp
from sklearn import metrics
import pandas as pd
from tqdm import tqdm

sample_seed = [ 42, 2, 82 ]

sample_types = ['unbalanced-lo', 'unbalanced', 'unbalanced-hi']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--method-type', type=str, default='baseline', 
                    choices=['efficient', 'baseline', 'baseline-feat'])
parser.add_argument('--perturb-type', type=str, default='discrete', 
                    choices=['discrete', 'continuous'])
parser.add_argument('--n-attack', type=int, default=500)
parser.add_argument('--eps', type=float, default=6)
args = parser.parse_args()

n_edges = np.zeros((3,3), dtype=np.int32)

dataset = args.dataset


for j, sample_type in enumerate(tqdm(sample_types)):
    for k, seed in enumerate(sample_seed):
        # data = torch.load(f'eval_{args.dataset}/attack-type-{args.method_type}_sample-type-{sample_type}_n-attack-{args.n_attack}_seed-{seed}.pt')    # pytorch-GAT
        # data = torch.load(f'eval_{args.dataset}/{args.method_type}_{sample_type}_{args.n_attack}_{seed}.pt')    # GCN
        data = torch.load(f'eval_{args.dataset}/{args.method_type}_{sample_type}_{args.perturb_type}_{args.n_attack}_{seed}_eps-{args.eps}_seed-42.pt')    # DP-GCN
        # data = torch.load(f'eval_{args.dataset}/type-{sample_type}_n-attack-{args.n_attack}_seed-42.pt')

        result = data['result']
        y = np.asarray(result['y'])

        n_edges[j][k] = y.sum()

print('pre-collecting stats done!')

n_nodes = 500
n_total = (n_nodes-1) * n_nodes // 2

def get_ratio(ne):
    return ne / n_total

baseline = np.zeros((3,3))
for j in range(3):
    for k in range(3):
        baseline[j][k] = get_ratio(n_edges[j][k])


def get_closest(value):
    cnt = 0
    while value < 1:
        value *= 10
        cnt += 1
    base = int(value)
    value -= int(value)
    if value >= 0.5:
        base += 1
    return base, cnt

approx = np.zeros((3,3), dtype=np.int32)
expn = np.zeros((3,3), dtype=np.int32)

for j in range(3):
    for k in range(3):
        a, b = get_closest(baseline[j][k])
        approx[j][k] = a
        expn[j][k] = b

def calc_f1(x, y):
    if x == 0 or y == 0: return 0
    return 2 * x * y / (x + y)

result_prec = np.zeros((3,3,5))
result_rec = np.zeros((3,3,5))
result_cnt = np.zeros((3,3,5), dtype=np.int32)
result_f1 = np.zeros((3,3,5))
result_auc = np.zeros((3,3))

for j, sample_type in enumerate(tqdm(sample_types)):
    for k, seed in enumerate(sample_seed):
        ratio = approx[j][k] / 10**expn[j][k]
        ratio_list = [ratio/4, ratio/2, ratio, ratio*2, ratio*4]

        # data = torch.load(f'eval_{args.dataset}/attack-type-{args.method_type}_sample-type-{sample_type}_n-attack-{args.n_attack}_seed-{seed}.pt')
        # data = torch.load(f'eval_{args.dataset}/{args.method_type}_{sample_type}_{args.n_attack}_{seed}.pt')
        data = torch.load(f'eval_{args.dataset}/{args.method_type}_{sample_type}_{args.perturb_type}_{args.n_attack}_{seed}_eps-{args.eps}_seed-42.pt')    # DP-GCN
        # data = torch.load(f'eval_{args.dataset}/type-{sample_type}_n-attack-{args.n_attack}_seed-{seed}.pt')
        result = data['result']

        auc = data['auc']
        fpr = auc['fpr']
        tpr = auc['tpr']
        auc_res = metrics.auc(fpr, tpr)
        result_auc[j][k] = auc_res

        pred = np.asarray(result['pred'])
        y = np.asarray(result['y'])    

        n_total = len(pred)

        for l, ratio in enumerate(ratio_list):
            n_pos = ceil(ratio * n_total)
            ind = np.argpartition(pred, -n_pos)[-n_pos:]
            n_tp = y[ind].sum()

            result_prec[j][k][l] = n_tp / n_pos
            result_rec[j][k][l] = n_tp / y.sum()
            result_cnt[j][k][l] = n_tp
            result_f1[j][k][l] = calc_f1(result_prec[j][k][l], result_rec[j][k][l])


###########################
#########       save
###########################

def get_df(mat, st):
    df = pd.DataFrame(mat.T)
    df.columns = ['low', 'normal', 'high']
    df.index = [st+':ratio/4', st+':ratio/2', st+':ratio', st+':ratio*2', st+':ratio*4']
    return df

if dataset.startswith('twitch'):
    datadir = dataset[:dataset.find('/')]
    cty = dataset[dataset.rfind('/')+1:]
else:
    datadir = dataset
    cty = ''


savedir = f'sheets/dp_attack_{datadir}'
if not osp.exists(savedir):
    os.makedirs(savedir)

# writer = pd.ExcelWriter(f'sheets/attack_{datadir}_3_layer/{cty}_{args.method_type}_{args.n_attack}.xlsx', engine='openpyxl')
# filename1 = f'sheets/attack_{datadir}/{args.method_type}_{args.n_attack}.xlsx'
if cty:
    filename1 = osp.join(savedir, f'{cty}_{args.method_type}_{args.perturb_type}_{args.eps}_{args.n_attack}.xlsx')
else:
    filename1 = osp.join(savedir, f'{args.method_type}_{args.perturb_type}_{args.eps}_{args.n_attack}.xlsx')

writer = pd.ExcelWriter(filename1, engine='openpyxl')

prec_mean = result_prec.mean(axis=1)
prec_std = result_prec.std(axis=1)

rec_mean = result_rec.mean(axis=1)
rec_std = result_rec.std(axis=1)

cnt_mean = result_cnt.mean(axis=1)
cnt_std = result_cnt.std(axis=1)

f1_mean = result_f1.mean(axis=1)
f1_std = result_f1.std(axis=1)

df = pd.concat([get_df(prec_mean, 'prec_mean'), get_df(prec_std, 'prec_std')])
df.to_excel(writer)
df = pd.concat([get_df(rec_mean, 'rec_mean'), get_df(rec_std, 'rec_std')])
df.to_excel(writer, startrow=15)
df = pd.concat([get_df(cnt_mean, 'cnt_mean'), get_df(cnt_std, 'cnt_std')])
df.to_excel(writer, startrow=30)
df = pd.concat([get_df(f1_mean, 'f1_mean'), get_df(f1_std, 'f1_std')])
df.to_excel(writer, startrow=45)

writer.save()
print(f'save prec, rec, cnt, f1 to {filename1}!')


auc_mean = result_auc.mean(axis=1)
auc_std = result_auc.std(axis=1)
auc = np.vstack((auc_mean, auc_std))

# pd.DataFrame(auc).to_csv(f'sheets/attack_{datadir}_3_layer/{cty}_{args.method_type}_{args.n_attack}_auc.csv')

# filename2 = f'sheets/attack_{datadir}/{args.method_type}_{args.n_attack}_auc.csv'
if cty:
    filename2 = osp.join(savedir, f'{cty}_{args.method_type}_{args.perturb_type}_{args.eps}_{args.n_attack}_auc.csv')
else:
    filename2 = osp.join(savedir, f'{args.method_type}_{args.perturb_type}_{args.eps}_{args.n_attack}_auc.csv')
pd.DataFrame(auc).to_csv(filename2)
print(f'save auc to {filename2}!')
