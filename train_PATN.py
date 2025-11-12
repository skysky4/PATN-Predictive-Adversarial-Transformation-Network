import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from ourmodel import PATN,Gender_model
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda:6")
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from torch.utils.data import TensorDataset

def History_Aware_Top_k_loss(x_concat, label, target_model, window_size=10, step_size=2, sample_num=5, top_k=2):

    
    batch_size, total_time, feature_dim = x_concat.shape
    all_starts = list(range(0, total_time - window_size + 1, step_size))
    
    if len(all_starts) == 0:
        return torch.tensor(0.0, device=x_concat.device)
    
    sampled_starts = list(range(0, total_time - window_size - 1, step_size))
    loss_list = []

    for start in sampled_starts:
        
        x_window = x_concat[:, start : start + window_size, :]  # (B, window_size, feature_dim)
        
        #x_window_flat = x_window.reshape(-1, feature_dim)  # (B * window_size, feature_dim)
        
        logits_adv = target_model(x_window)  # (B * window_size, num_classes)
    
        loss = F.cross_entropy(logits_adv,  1 - label)
        loss_list.append(loss)
    
    if len(loss_list) == 0:
        return torch.tensor(0.0, device=x_concat.device)
    
    loss_tensor = torch.stack(loss_list)
    topk_loss = torch.topk(loss_tensor, k=min(top_k, len(loss_list))).values
    total_loss = topk_loss.mean()

    return total_loss

# ==== 参数 ====
perturb_limits = torch.tensor([0.06451838 , 0.06075889, 0.04009763, 0.0160811,  0.02619115, 0.018439]).to(device)
criterion = nn.CrossEntropyLoss()
batch_size =64

# ==== 数据加载 ====
dataset1 = np.load('./datanpy/dataset_train_sensor6_40.npy', allow_pickle=True)
furdataset1 = np.load('./datanpy/furdataset_train_sensor6_40.npy', allow_pickle=True)
label1 = np.load('./datanpy/label_train_sensor6_40.npy', allow_pickle=True)
label1_act = np.load('./datanpy/labelact_train_sensor6_40.npy', allow_pickle=True)
his = np.load('./datanpy/his_train_sensor6_40.npy', allow_pickle=True)
furhis = np.load('./datanpy/furhis_train_sensor6_40.npy', allow_pickle=True)

dataset2 = np.load('./datanpy/dataset_test_sensor6_40.npy', allow_pickle=True)
furdataset2 = np.load('./datanpy/furdataset_test_sensor6_40.npy', allow_pickle=True)
label2 = np.load('./datanpy/label_test_sensor6_40.npy', allow_pickle=True)

val_dataset = TensorDataset(torch.tensor(dataset2, dtype=torch.float32),
                            torch.tensor(furdataset2, dtype=torch.float32),
                            torch.tensor(label2, dtype=torch.long))

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


PATN = PATN(hidden_dim=64,output_len=10).to(device)
target_model = Gender_model().to(device)
target_model.load_state_dict(torch.load("./model/Gender_model.pth"))
target_model.eval()
optimizer = optim.Adam(PATN.parameters(), lr=0.001)



def filter_correct_dataset(target_model, dataset1, furdataset1, his, furhis, label1,label1_act):
    """
    Keep the data classified correctly by target_model and return a new TensorDataset.
    """
    target_model.eval()
    
    x = torch.tensor(furdataset1, dtype=torch.float32).to(device)
    y = torch.tensor(label1, dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = target_model(x)
        preds = outputs.argmax(dim=1)


    correct_mask = (preds == y)
    correct_indices = correct_mask.nonzero(as_tuple=True)[0]
    dataset1_filtered = torch.tensor(dataset1, dtype=torch.float32)[correct_indices]
    furdataset1_filtered = torch.tensor(furdataset1, dtype=torch.float32)[correct_indices]
    his_filtered = torch.tensor(his, dtype=torch.float32)[correct_indices]
    furhis_filtered = torch.tensor(furhis, dtype=torch.float32)[correct_indices]
    label1_filtered = torch.tensor(label1, dtype=torch.long)[correct_indices]
    label_act = torch.tensor(label1_act, dtype=torch.long)[correct_indices]
    filtered_dataset = TensorDataset(dataset1_filtered, furdataset1_filtered, his_filtered, furhis_filtered, label1_filtered,label_act)

    return filtered_dataset, correct_indices

train_dataset, correct_indices = filter_correct_dataset(target_model, dataset1, furdataset1, his, furhis, label1,label1_act)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
def train_one_epoch(PATN, target_model, train_loader, optimizer, perturb_limits, device):
    PATN.train()
    target_model.eval() 
    total_loss = 0
    total_batches = 0
    for batch_xt, batch_xt1,his_xt,his_x1, label,label_act in tqdm(train_loader):
        batch_xt = batch_xt.to(device)
        batch_xt1 = batch_xt1.to(device)
        label = label.to(device)
        label_act = label_act.to(device)
        his_xt = his_xt.to(device)
        his_x1 = his_x1.to(device)
        delta = PATN(batch_xt)
        limits = perturb_limits.unsqueeze(0).unsqueeze(0).expand_as(delta)
        delta = torch.clamp(delta, -limits, limits)
        adv_xt1 = batch_xt1 + delta
        logits_adv = target_model(adv_xt1)
        his_delta = PATN(his_xt)
        limits = perturb_limits.unsqueeze(0).unsqueeze(0).expand_as(delta)
        his_delta = torch.clamp(his_delta, -limits, limits)
        adv_his = his_x1 + his_delta
        smooth_loss = F.mse_loss(delta[:, 1:, :], delta[:, :-1, :])
        adv_concat = torch.cat([adv_his, adv_xt1], dim=1)
        loss = 0.3 * History_Aware_Top_k_loss(adv_concat, label, target_model) + 0.3 * smooth_loss + F.cross_entropy(logits_adv, 1 - label) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1 
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    return avg_loss


def validate(PATN, target_model, val_loader, perturb_limits, device):
    PATN.eval()
    target_model.eval()

    total_considered = 0  
    attack_success = 0   
    with torch.no_grad():
        for batch_xt, batch_xt1, batch_y in val_loader:
            batch_xt = batch_xt.to(device)
            batch_xt1 = batch_xt1.to(device)
            batch_y = batch_y.to(device)

        
            logits_clean = target_model(batch_xt1)
            preds_clean = torch.argmax(logits_clean, dim=1)

            correct_mask = (preds_clean == batch_y)  

            if correct_mask.sum() == 0:
                continue  

            delta = PATN(batch_xt)
            limits = perturb_limits.unsqueeze(0).unsqueeze(0).expand_as(delta)
            delta = torch.clamp(delta, -limits, limits)

            adv_xt1 = batch_xt1 + delta
            logits_adv = target_model(adv_xt1)
            preds_adv = torch.argmax(logits_adv, dim=1)

            preds_adv = preds_adv[correct_mask]
            batch_y = batch_y[correct_mask]

            attack_success += (preds_adv != batch_y).sum().item()
            total_considered += correct_mask.sum().item()

    attack_success_rate = attack_success / total_considered if total_considered > 0 else 0
    return attack_success_rate


# ==== Main train ====


for epoch in range(600):
    train_loss = train_one_epoch(PATN, target_model, train_loader, optimizer, perturb_limits, device)
    attack_success_rate = validate(PATN, target_model, val_loader, perturb_limits, device)
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, ASR = {attack_success_rate*100:.2f}%")
    torch.save(PATN.state_dict(),  f"./PATN_epoch{epoch}.pth")
  
  