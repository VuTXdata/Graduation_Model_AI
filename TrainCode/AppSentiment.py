import csv
import json
import logging
import os
import re

import emoji
import pandas as pd
import torch
from datasets import load_dataset
from pyvi import ViTokenizer
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Tải danh sách từ viết tắt
def load_abbreviations(file_path):
    abbr_dict = {}
    try:
        if file_path.endswith('.csv'):
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    abbr_dict[row['viết_tắt']] = row['từ_đầy_đủ']
        elif file_path.endswith('.json'):
            with open(file_path, mode='r', encoding='utf-8') as file:
                abbr_dict = json.load(file)
        elif file_path.endswith('.txt'):
            with open(file_path, mode='r', encoding='utf-8') as file:
                for line in file:
                    key, value = line.strip().split(':')
                    abbr_dict[key] = value
        return abbr_dict
    except Exception as e:
        logger.warning(f"Không thể tải từ viết tắt: {e}. Sử dụng danh sách mặc định.")
        return {}


# Tiền xử lý văn bản
def preprocess_text(text, restore_accent=True, abbreviations=None):
    if not isinstance(text, str):
        return ""
    text = emoji.demojize(text)
    text = re.sub(r'[^\w\s\.]', ' ', text).lower()
    if abbreviations:
        words = text.split()
        normalized_words = [abbreviations.get(word, word) for word in words]
        text = ' '.join(normalized_words)
    if restore_accent:
        try:
            text = ViTokenizer.tokenize(text)
        except Exception as e:
            logger.error(f"Lỗi khi khôi phục dấu bằng pyvi: {e}")
    return text


# Tải dữ liệu huấn luyện
def load_train_datasets(abbreviations_path=None, vsa_train_path=None, aivivn_train_path=None):
    combined_data = []
    abbreviations = load_abbreviations(abbreviations_path) if abbreviations_path else {}

    # 1. Vietnamese Students Feedback (Hugging Face)
    try:
        logger.info("Đang tải Vietnamese Students Feedback...")
        ds = load_dataset("uitnlp/vietnamese_students_feedback")
        vsf_data = pd.DataFrame(ds['train'])[['sentence', 'sentiment']].rename(
            columns={'sentence': 'text', 'sentiment': 'label'}
        )
        combined_data.append(vsf_data)
        logger.info(f"Đã tải {len(vsf_data)} mẫu từ Vietnamese Students Feedback")
    except Exception as e:
        logger.error(f"Lỗi khi tải Vietnamese Students Feedback: {e}")

    # 2. Vietnamese Sentiment Analyst (từ file train)
    if vsa_train_path:
        try:
            logger.info("Đang tải Vietnamese Sentiment Analyst train từ file...")
            vsa_data = pd.read_csv(vsa_train_path)  # Đọc data_data
            vsa_processed = vsa_data[['comment', 'label']].rename(columns={'comment': 'text'})
            vsa_processed['label'] = vsa_processed['label'].map({'NEG': 0, 'NEU': 1, 'POS': 2})
            combined_data.append(vsa_processed)
            logger.info(f"Đã tải {len(vsa_processed)} mẫu từ Vietnamese Sentiment Analyst (train)")
        except Exception as e:
            logger.error(f"Lỗi khi tải Vietnamese Sentiment Analyst train từ file: {e}")

    # 3. AIVIVN 2019 (từ file train)
    if aivivn_train_path:
        try:
            logger.info("Đang tải AIVIVN 2019 train từ file...")
            aivivn_data = pd.read_csv(aivivn_train_path)  # Đọc train.csv
            aivivn_processed = aivivn_data[['comment', 'label']].rename(columns={'comment': 'text'})
            aivivn_processed['label'] = aivivn_processed['label'].map({0: 0, 1: 2})
            combined_data.append(aivivn_processed)
            logger.info(f"Đã tải {len(aivivn_processed)} mẫu từ AIVIVN 2019 (train)")
        except Exception as e:
            logger.error(f"Lỗi khi tải AIVIVN 2019 train từ file: {e}")

    if combined_data:
        all_data = pd.concat(combined_data, ignore_index=True)
        all_data['processed_text'] = all_data['text'].apply(
            lambda x: preprocess_text(x, restore_accent=True, abbreviations=abbreviations)
        )
        logger.info(f"Tổng số mẫu dữ liệu huấn luyện: {len(all_data)}")
        return all_data
    else:
        logger.error("Không có dữ liệu huấn luyện!")
        return pd.DataFrame(columns=['text', 'label', 'processed_text'])


# Tải dữ liệu kiểm tra
def load_test_datasets(abbreviations_path=None, vsa_test_path=None, aivivn_test_path=None):
    combined_data = []
    abbreviations = load_abbreviations(abbreviations_path) if abbreviations_path else {}

    # 1. Vietnamese Sentiment Analyst (từ file test)
    if vsa_test_path:
        try:
            logger.info("Đang tải Vietnamese Sentiment Analyst test từ file...")
            vsa_data = pd.read_csv(vsa_test_path)  # Đọc data
            vsa_processed = vsa_data[['content', 'label']].rename(columns={'comment': 'text'})
            vsa_processed['label'] = vsa_processed['label'].map({'NEG': 0, 'NEU': 1, 'POS': 2})
            combined_data.append(vsa_processed)
            logger.info(f"Đã tải {len(vsa_processed)} mẫu từ Vietnamese Sentiment Analyst (test)")
        except Exception as e:
            logger.error(f"Lỗi khi tải Vietnamese Sentiment Analyst test từ file: {e}")

    # 2. AIVIVN 2019 (từ file test)
    if aivivn_test_path:
        try:
            logger.info("Đang tải AIVIVN 2019 test từ file...")
            aivivn_data = pd.read_csv(aivivn_test_path)  # Đọc test.csv
            aivivn_processed = aivivn_data[['comment', 'label']].rename(columns={'comment': 'text'})
            aivivn_processed['label'] = aivivn_processed['label'].map({0: 0, 1: 2})
            combined_data.append(aivivn_processed)
            logger.info(f"Đã tải {len(aivivn_processed)} mẫu từ AIVIVN 2019 (test)")
        except Exception as e:
            logger.error(f"Lỗi khi tải AIVIVN 2019 test từ file: {e}")

    if combined_data:
        all_data = pd.concat(combined_data, ignore_index=True)
        all_data['processed_text'] = all_data['text'].apply(
            lambda x: preprocess_text(x, restore_accent=True, abbreviations=abbreviations)
        )
        logger.info(f"Tổng số mẫu dữ liệu kiểm tra: {len(all_data)}")
        return all_data
    else:
        logger.error("Không có dữ liệu kiểm tra!")
        return pd.DataFrame(columns=['text', 'label', 'processed_text'])


# Dataset cho PyTorch
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# Huấn luyện mô hình
def train_model(model, train_dataloader, val_dataloader, device, num_epochs=5, evaluation_steps=100, patience=3):
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.01)
    total_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)

    best_f1 = 0
    epochs_no_improve = 0
    best_model_path = os.path.join(models_dir, 'phobert_best_model.pt')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': total_train_loss / (step + 1)})

            if (step + 1) % evaluation_steps == 0:
                val_metrics = evaluate_model(model, val_dataloader, device)
                logger.info(f"Validation - F1: {val_metrics['f1']:.4f}")

                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    epochs_no_improve += 1

                model.train()

        val_metrics = evaluate_model(model, val_dataloader, device)
        logger.info(f"Epoch {epoch + 1} - F1: {val_metrics['f1']:.4f}")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping tại epoch {epoch + 1}")
            break

    return best_model_path, {'f1': best_f1}


# Đánh giá mô hình
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return {
        'loss': total_loss / len(dataloader),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'report': classification_report(all_labels, all_preds)
    }


# Dự đoán cảm xúc
def predict_emotion(text, model, tokenizer, device, abbreviations=None):
    processed_text = preprocess_text(text, restore_accent=True, abbreviations=abbreviations)
    encoding = tokenizer.encode_plus(
        processed_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()

    emotion_map = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
    probabilities = predictions[0].cpu().numpy()

    return {
        "text": text,
        "emotion": emotion_map.get(predicted_class, "Unknown"),
        "confidence": float(probabilities[predicted_class])
    }


# Hàm chính
def main(abbreviations_path='abbreviations.csv',
         vsa_train_path='Vietnamese_Sentiment_Analyst/data_data',
         aivivn_train_path='AIVIVN_2019/train.csv',
         vsa_test_path='Vietnamese_Sentiment_Analyst/data',
         aivivn_test_path='AIVIVN_2019/test.csv'):
    logger.info("Chương trình bắt đầu chạy!")
    # Tải dữ liệu huấn luyện
    train_data = load_train_datasets(abbreviations_path, vsa_train_path, aivivn_train_path)
    if len(train_data) == 0:
        logger.error("Không có dữ liệu huấn luyện để tiếp tục!")
        return

    # Chia tập train thành train và validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_data['processed_text'].values,
        train_data['label'].values,
        test_size=0.1,
        random_state=42,
        stratify=train_data['label']
    )

    # Tải dữ liệu kiểm tra
    test_data = load_test_datasets(abbreviations_path, vsa_test_path, aivivn_test_path)
    if len(test_data) == 0:
        logger.warning("Không có dữ liệu kiểm tra, chỉ huấn luyện trên tập train/validation.")

    # Khởi tạo tokenizer và mô hình
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        "vinai/phobert-base",
        num_labels=len(train_data['label'].unique())
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Thiết bị đang sử dụng: {device}")
    model = model.to(device)

    # Tạo dataloader cho train và validation
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    # Huấn luyện mô hình
    best_model_path, best_metrics = train_model(
        model, train_dataloader, val_dataloader, device, num_epochs=5, patience=3
    )

    # Lưu tokenizer
    tokenizer.save_pretrained("./models/phobert_tokenizer")
    logger.info(f"Mô hình tốt nhất: {best_model_path}, F1-score trên validation: {best_metrics['f1']}")

    # Đánh giá trên tập test nếu có
    if len(test_data) > 0:
        test_texts = test_data['processed_text'].values
        test_labels = test_data['label'].values
        test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=32)

        logger.info("Đánh giá mô hình trên tập test...")
        test_metrics = evaluate_model(model, test_dataloader, device)
        logger.info(f"Test - F1: {test_metrics['f1']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Classification Report trên tập test:\n{test_metrics['report']}")

    return best_model_path, tokenizer, model, device


if __name__ == "__main__":
    try:
        best_model_path, tokenizer, model, device = main(
            abbreviations_path='dataset/abbreviations.csv',
            vsa_train_path='dataset/Vietnamese_Sentiment_Analyst/data_data.csv',
            aivivn_train_path='dataset/AIVIVN_2019/train.csv',
            vsa_test_path='dataset/Vietnamese_Sentiment_Analyst/data.csv',
            aivivn_test_path='dataset/AIVIVN_2019/test.csv'
        )
        logger.info("Huấn luyện và đánh giá hoàn tất!")
    except Exception as e:
        logger.error(f"Lỗi: {e}")
