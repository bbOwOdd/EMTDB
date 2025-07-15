import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_data = pd.read_csv("dataset/Composite.csv")
print(train_data["label"].value_counts(normalize=True))

test_data = pd.read_csv("dataset//USTC.csv") #IoT23 USTC
print(test_data["label"].value_counts(normalize=True))

# 定義數值和文字特徵
num_features = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
text_features = ['version', 'cipher', 'certificate']

# 對數值特徵進行標準化
scaler = StandardScaler()
train_data[num_features] = scaler.fit_transform(train_data[num_features])
test_data[num_features] = scaler.fit_transform(test_data[num_features])

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# 自定義 Dataset 類別
class CustomDataset(Dataset):
    def __init__(self, data, num_features, text_features, tokenizer):
        self.data = data
        self.num_features = data[num_features].values
        self.text_features = data[text_features]
        self.labels = data['label'].values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        num_data = torch.tensor(self.num_features[idx], dtype=torch.float)

        # 將文字特徵合併
        text_data = ' '.join(self.text_features.iloc[idx].astype(str).values)
        encoding = self.tokenizer(text_data, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, num_data, label

# 建立訓練和測試資料集
train_dataset = CustomDataset(train_data, num_features, text_features, tokenizer)
test_dataset = CustomDataset(test_data, num_features, text_features, tokenizer)

# 定義模型
class BertWithNumeric(nn.Module):
    def __init__(self):
        super(BertWithNumeric, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.num_features_layer = nn.Linear(len(num_features), 128)  # 數值特徵處理層
        self.classifier = nn.Linear(768 + 128, 2)  # 結合BERT和數值特徵後進行分類

    def forward(self, input_ids, attention_mask, num_data):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output

        num_output = torch.relu(self.num_features_layer(num_data))

        combined_output = torch.cat((pooled_output, num_output), dim=1)

        logits = self.classifier(combined_output)
        return logits

model = BertWithNumeric().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

def train_model(dataloader, model, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            input_ids, attention_mask, num_data, labels = batch
            
            # 將批次資料移動到 GPU
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            num_data = num_data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, num_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# 使用交叉驗證進行訓練
def cross_validate_model(train_dataset, model, optimizer, criterion, k=5):
    kf = KFold(n_splits=k)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f'Fold {fold + 1}/{k}')
        
        # 建立子集
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        # 建立DataLoader
        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

        # 訓練模型
        train_model(train_loader, model, optimizer, criterion)

        # 在驗證集上評估
        evaluate_model(val_loader, model)

# 測試模型並獲取預測
def evaluate_model(dataloader, model):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, num_data, labels = batch
            
            # 將批次資料移動到 GPU
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            num_data = num_data.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask, num_data)
            preds = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return all_labels, all_preds

# 計算並印出評估指標
def compute_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

    #繪製混淆矩陣
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['0-malicious', '1-normal'], yticklabels=['0-malicious', '1-normal'])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

cross_validate_model(train_dataset, model, optimizer, criterion)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
labels, preds = evaluate_model(test_loader, model)
compute_metrics(labels, preds)
