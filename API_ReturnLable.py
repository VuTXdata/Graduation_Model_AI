import json
import logging
import os
from typing import List
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Định nghĩa mô hình (phải khớp với mô hình đã huấn luyện)
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


# Định nghĩa request body
class TextItem(BaseModel):
    id_text: str  # Có thể là string hoặc int, tùy dữ liệu của bạn
    text: str


class TextRequest(BaseModel):
    items: List[TextItem]  # Danh sách các cặp {id_text, text}


# Khởi tạo FastAPI
app = FastAPI(title="MultiTask Sentiment and Task Classification API")

# Khởi tạo các biến toàn cục
model = None
tokenizer = None
device = None


# Hàm load mô hình và tokenizer
def load_model_and_tokenizer(model_path="./models/phobert_multitask_best_model_V2.pt",
                             tokenizer_path="./models/phobert_multitask_tokenizer"):
    global model, tokenizer, device

    try:
        # Xác định thiết bị
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Thiết bị đang sử dụng: {device}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Đã tải tokenizer từ {tokenizer_path}")

        # Khởi tạo mô hình
        model = MultiTaskModel("vinai/phobert-base", num_sentiment_labels=3, num_task_labels=9, alpha=0.4, beta=0.6)

        # Load trọng số mô hình
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()  # Chuyển sang chế độ đánh giá
        logger.info(f"Đã tải mô hình từ {model_path}")

    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình hoặc tokenizer: {e}")
        raise Exception("Không thể tải mô hình hoặc tokenizer")


# Hàm dự đoán batch
def predict_batch(items: List[TextItem], model, tokenizer, device, max_len=128):
    # Tách danh sách text để tokenize
    texts = [item.text for item in items]
    id_texts = [item.id_text for item in items]

    # Tokenize tất cả các câu cùng lúc
    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        sentiment_logits = outputs['sentiment_logits']
        task_logits = outputs['task_logits']

        # Tính xác suất
        sentiment_probs = torch.softmax(sentiment_logits, dim=1)
        task_probs = torch.softmax(task_logits, dim=1)

        # Lấy nhãn dự đoán và độ tin cậy
        sentiment_preds = torch.argmax(sentiment_probs, dim=1).cpu().numpy()
        task_preds = torch.argmax(task_probs, dim=1).cpu().numpy()
        sentiment_confidences = torch.max(sentiment_probs, dim=1).values.cpu().numpy()
        task_confidences = torch.max(task_probs, dim=1).values.cpu().numpy()

    # Tạo danh sách kết quả
    results = []
    for i in range(len(items)):
        result = {
            "id_text": id_texts[i],
            "sentiment": int(sentiment_preds[i]),  # Trả về dạng số (0, 1, 2)
            "sentiment_confidence": float(sentiment_confidences[i]),
            "task": int(task_preds[i] + 1),  # Trả về dạng số (1-9)
            "task_confidence": float(task_confidences[i])
        }
        results.append(result)

    return results


# Load mô hình khi khởi động API
@app.on_event("startup")
async def startup_event():
    load_model_and_tokenizer()


# Định nghĩa endpoint phân loại batch
@app.post("/predict", response_model=List[dict])
async def predict_texts(request: TextRequest):
    try:
        if not request.items:
            raise HTTPException(status_code=400, detail="Danh sách câu trống")

        logger.info(f"Nhận yêu cầu phân loại cho {len(request.items)} câu")
        predictions = predict_batch(request.items, model, tokenizer, device)
        logger.info(f"Đã xử lý xong {len(predictions)} câu")
        return predictions

    except Exception as e:
        logger.error(f"Lỗi trong quá trình dự đoán: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")


# Endpoint kiểm tra trạng thái API
@app.get("/health")
async def health_check():
    return {"status": "API is running", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)