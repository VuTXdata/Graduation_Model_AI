import csv
import json
import logging
import os
import re

import emoji
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Các ánh xạ nhãn
SENTIMENT_MAP = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
TASK_MAP = {
    1: "Chất lượng", 2: "Đóng gói", 3: "Giao hàng", 4: "Giá thành", 5: "Hỗ trợ khách hàng",
    6: "Đặt hàng", 7: "Trả hàng", 8: "Khuyến mãi", 9: "Khác"
}


# Tiền xử lý văn bản
def preprocess_text(text, abbreviations=None):
    if not isinstance(text, str):
        return ""
    text = emoji.demojize(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.]', ' ', text).lower()
    if abbreviations:
        words = text.split()
        normalized_words = [abbreviations.get(word, word) for word in words]
        text = ' '.join(normalized_words)
    return text.strip()


# Tải danh sách từ viết tắt từ file (nếu có)
def load_abbreviations(file_path):
    abbr_dict = {}
    try:
        if os.path.exists(file_path):
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
                        if ':' in line:
                            key, value = line.strip().split(':')
                            abbr_dict[key] = value
            return abbr_dict
        else:
            logger.warning(f"File từ viết tắt không tồn tại: {file_path}")
            return {}
    except Exception as e:
        logger.warning(f"Không thể tải từ viết tắt: {e}. Sử dụng danh sách trống.")
        return {}


# Tải dữ liệu từ file CSV
def load_dataset(file_path, abbreviations=None):
    try:
        logger.info(f"Đang tải dữ liệu từ {file_path}...")
        data = pd.read_csv(file_path)

        required_columns = ['id', 'comment', 'label', 'task']
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            logger.error(f"Thiếu các cột: {', '.join(missing)}")
            return None

        if not data['label'].isin([0, 1, 2]).all():
            logger.error("Nhãn cảm xúc không hợp lệ (phải là 0, 1, 2)")
            return None
        if not data['task'].isin(range(1, 10)).all():
            logger.error("Nhãn task không hợp lệ (phải từ 1 đến 9)")
            return None

        data['processed_text'] = data['comment'].apply(
            lambda x: preprocess_text(x, abbreviations=abbreviations)
        )

        logger.info(f"Đã tải {len(data)} mẫu dữ liệu")
        return data
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu: {e}")
        return None


# Dataset cho PyTorch với nhiều nhãn
class MultiTaskDataset(Dataset):
    def __init__(self, texts, sentiment_labels, task_labels, tokenizer, max_len=128):
        self.texts = texts
        self.sentiment_labels = sentiment_labels
        self.task_labels = [label - 1 for label in task_labels]  # Chuyển task từ 1-9 thành 0-8
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
            'sentiment_label': torch.tensor(self.sentiment_labels[idx], dtype=torch.long),
            'task_label': torch.tensor(self.task_labels[idx], dtype=torch.long)
        }


# Mô hình đa nhiệm vụ
class MultiTaskModel(nn.Module):
    def __init__(self, base_model_name, num_sentiment_labels, num_task_labels, alpha=0.45, beta=0.55):
        super(MultiTaskModel, self).__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size
        self.sentiment_classifier = nn.Linear(hidden_size, num_sentiment_labels)
        self.task_classifier = nn.Linear(hidden_size, num_task_labels)
        self.alpha = alpha
        self.beta = beta

    def forward(self, input_ids, attention_mask, sentiment_label=None, task_label=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Lấy biểu diễn của token [CLS]

        sentiment_logits = self.sentiment_classifier(pooled_output)
        task_logits = self.task_classifier(pooled_output)

        loss = None
        if sentiment_label is not None and task_label is not None:
            sentiment_loss = nn.CrossEntropyLoss()(sentiment_logits, sentiment_label)
            task_loss = nn.CrossEntropyLoss()(task_logits, task_label)
            loss = self.alpha * sentiment_loss + self.beta * task_loss

        return {
            'loss': loss,
            'sentiment_logits': sentiment_logits,
            'task_logits': task_logits
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
    best_model_path = os.path.join(models_dir, 'phobert_multitask_best_model_V2.pt')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)
            task_labels = batch['task_label'].to(device)

            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentiment_label=sentiment_labels,
                task_label=task_labels
            )
            loss = outputs['loss']
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': total_train_loss / (step + 1)})

            if (step + 1) % evaluation_steps == 0:
                val_metrics = evaluate_model(model, val_dataloader, device)
                logger.info(
                    f"Validation - Sentiment F1: {val_metrics['sentiment_f1']:.4f}, Task F1: {val_metrics['task_f1']:.4f}")

                avg_f1 = (val_metrics['sentiment_f1'] + val_metrics['task_f1']) / 2
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    epochs_no_improve = 0
                    torch.save({'state_dict': model.state_dict()}, best_model_path)
                    logger.info(f"Đã lưu mô hình tốt nhất với Avg F1: {avg_f1:.4f}")
                else:
                    epochs_no_improve += 1

                model.train()

        val_metrics = evaluate_model(model, val_dataloader, device)
        avg_f1 = (val_metrics['sentiment_f1'] + val_metrics['task_f1']) / 2
        logger.info(
            f"Epoch {epoch + 1} - Sentiment F1: {val_metrics['sentiment_f1']:.4f}, Task F1: {val_metrics['task_f1']:.4f}, Avg F1: {avg_f1:.4f}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            epochs_no_improve = 0
            torch.save({'state_dict': model.state_dict()}, best_model_path)
            logger.info(f"Đã lưu mô hình tốt nhất với Avg F1: {avg_f1:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping tại epoch {epoch + 1}")
            break

    return best_model_path, {'avg_f1': best_f1}


# Đánh giá mô hình
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_sentiment_preds, all_sentiment_labels = [], []
    all_task_preds, all_task_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)
            task_labels = batch['task_label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentiment_label=sentiment_labels,
                task_label=task_labels
            )
            total_loss += outputs['loss'].item()

            sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=1).cpu().numpy()
            task_preds = torch.argmax(outputs['task_logits'], dim=1).cpu().numpy()

            all_sentiment_preds.extend(sentiment_preds)
            all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
            all_task_preds.extend(task_preds)
            all_task_labels.extend(task_labels.cpu().numpy())

    sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)
    sentiment_f1 = f1_score(all_sentiment_labels, all_sentiment_preds, average='weighted')
    task_accuracy = accuracy_score(all_task_labels, all_task_preds)
    task_f1 = f1_score(all_task_labels, all_task_preds, average='weighted')

    return {
        'loss': total_loss / len(dataloader),
        'sentiment_accuracy': sentiment_accuracy,
        'sentiment_f1': sentiment_f1,
        'task_accuracy': task_accuracy,
        'task_f1': task_f1,
        'sentiment_report': classification_report(all_sentiment_labels, all_sentiment_preds,
                                                  target_names=list(SENTIMENT_MAP.values())),
        'task_report': classification_report(all_task_labels, all_task_preds,
                                             target_names=[TASK_MAP[i + 1] for i in range(len(TASK_MAP))]),
        'sentiment_cm': confusion_matrix(all_sentiment_labels, all_sentiment_preds),
        'task_cm': confusion_matrix(all_task_labels, all_task_preds)
    }


# Trực quan hóa kết quả đánh giá
def visualize_results(test_metrics, output_dir='./results'):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        test_metrics['sentiment_cm'],
        annot=True, fmt='d', cmap='Blues',
        xticklabels=list(SENTIMENT_MAP.values()),
        yticklabels=list(SENTIMENT_MAP.values())
    )
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Confusion Matrix - Sentiment')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_confusion_matrix.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        test_metrics['task_cm'],
        annot=True, fmt='d', cmap='Blues',
        xticklabels=[TASK_MAP[i + 1] for i in range(len(TASK_MAP))],
        yticklabels=[TASK_MAP[i + 1] for i in range(len(TASK_MAP))]
    )
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Confusion Matrix - Task')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/task_confusion_matrix.png")
    plt.close()

    with open(f"{output_dir}/classification_reports.txt", 'w') as f:
        f.write("Sentiment Classification Report:\n")
        f.write(test_metrics['sentiment_report'])
        f.write("\n\nTask Classification Report:\n")
        f.write(test_metrics['task_report'])

    metrics_summary = {
        'sentiment_accuracy': test_metrics['sentiment_accuracy'],
        'sentiment_f1': test_metrics['sentiment_f1'],
        'task_accuracy': test_metrics['task_accuracy'],
        'task_f1': test_metrics['task_f1']
    }
    with open(f"{output_dir}/metrics_summary.json", 'w') as f:
        json.dump(metrics_summary, f, indent=4)

    logger.info(f"Đã lưu kết quả trực quan hóa và báo cáo vào thư mục {output_dir}")


# Dự đoán cảm xúc và task
def predict(text, model, tokenizer, device, abbreviations=None):
    processed_text = preprocess_text(text, abbreviations=abbreviations)
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
        sentiment_predictions = torch.softmax(outputs['sentiment_logits'], dim=1)
        task_predictions = torch.softmax(outputs['task_logits'], dim=1)

        sentiment_class = torch.argmax(sentiment_predictions, dim=1).item()
        task_class = torch.argmax(task_predictions, dim=1).item()  # 0-8

        sentiment_confidence = float(sentiment_predictions[0][sentiment_class])
        task_confidence = float(task_predictions[0][task_class])

    return {
        "text": text,
        "processed_text": processed_text,
        "sentiment": SENTIMENT_MAP.get(sentiment_class, "Unknown"),
        "sentiment_confidence": sentiment_confidence,
        "task": TASK_MAP.get(task_class + 1, "Unknown"),  # Chuyển từ 0-8 về 1-9
        "task_confidence": task_confidence
    }


# Hàm tạo dịch vụ dự đoán
def create_prediction_service(model, tokenizer, device, abbreviations=None):
    def predict_service(text):
        return predict(text, model, tokenizer, device, abbreviations)

    return predict_service


# Hàm chính
def main(data_path, abbreviations_path=None, test_size=0.15, val_size=0.15, num_epochs=5, batch_size=16, patience=3):
    logger.info("Chương trình bắt đầu chạy!")

    abbreviations = load_abbreviations(abbreviations_path) if abbreviations_path else {}

    data = load_dataset(data_path, abbreviations)
    if data is None or len(data) == 0:
        logger.error("Không có dữ liệu để tiếp tục!")
        return

    train_val_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=42,
        stratify=data[['label', 'task']]
    )

    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size / (1 - test_size),
        random_state=42,
        stratify=train_val_data[['label', 'task']]
    )

    logger.info(f"Phân chia dữ liệu: Train {len(train_data)}, Validation {len(val_data)}, Test {len(test_data)}")

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

    train_dataset = MultiTaskDataset(
        train_data['processed_text'].values,
        train_data['label'].values,
        train_data['task'].values,
        tokenizer
    )

    val_dataset = MultiTaskDataset(
        val_data['processed_text'].values,
        val_data['label'].values,
        val_data['task'].values,
        tokenizer
    )

    test_dataset = MultiTaskDataset(
        test_data['processed_text'].values,
        test_data['label'].values,
        test_data['task'].values,
        tokenizer
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size * 2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size * 2)

    num_sentiment_labels = len(SENTIMENT_MAP)
    num_task_labels = len(TASK_MAP)

    model = MultiTaskModel("vinai/phobert-base", num_sentiment_labels, num_task_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Thiết bị đang sử dụng: {device}")
    model = model.to(device)

    best_model_path, best_metrics = train_model(
        model,
        train_dataloader,
        val_dataloader,
        device,
        num_epochs=num_epochs,
        evaluation_steps=100,
        patience=patience
    )

    tokenizer_path = "models/phobert_multitask_tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Đã lưu tokenizer tại {tokenizer_path}")

    model.load_state_dict(torch.load(best_model_path)['state_dict'])
    logger.info(f"Đã tải mô hình tốt nhất từ {best_model_path}")

    logger.info("Đánh giá mô hình trên tập test...")
    test_metrics = evaluate_model(model, test_dataloader, device)

    logger.info(f"Test - Sentiment F1: {test_metrics['sentiment_f1']:.4f}, Task F1: {test_metrics['task_f1']:.4f}")
    logger.info(
        f"Test - Sentiment Accuracy: {test_metrics['sentiment_accuracy']:.4f}, Task Accuracy: {test_metrics['task_accuracy']:.4f}")
    logger.info(f"Sentiment Report:\n{test_metrics['sentiment_report']}")
    logger.info(f"Task Report:\n{test_metrics['task_report']}")

    visualize_results(test_metrics)

    example_text = "Chất lượng sản phẩm rất tốt, giao hàng nhanh"
    prediction = predict(example_text, model, tokenizer, device, abbreviations)
    logger.info(f"Ví dụ dự đoán: {json.dumps(prediction, ensure_ascii=False, indent=2)}")

    predict_service = create_prediction_service(model, tokenizer, device, abbreviations)

    return {
        'model': model,
        'tokenizer': tokenizer,
        'device': device,
        'abbreviations': abbreviations,
        'metrics': test_metrics,
        'best_model_path': best_model_path,
        'predict_service': predict_service
    }


if __name__ == "__main__":
    try:
        result = main(
            data_path="./dataset/dataset_datn_csv.csv",
            abbreviations_path='./dataset/abbreviations.csv',
            test_size=0.15,
            val_size=0.15,
            num_epochs=5,
            batch_size=16,
            patience=3
        )
        if result:
            logger.info("Chương trình hoàn tất thành công!")
            # Ví dụ sử dụng predict_service
            predict_fn = result['predict_service']
            test_text = "Hỗ trợ khách hàng tệ quá"
            prediction = predict_fn(test_text)
            logger.info(f"Dự đoán từ dịch vụ: {json.dumps(prediction, ensure_ascii=False, indent=2)}")
    except Exception as e:
        logger.error(f"Lỗi trong quá trình chạy chương trình: {e}")
