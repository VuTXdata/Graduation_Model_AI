import csv
import logging
import re
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import emoji
from pyvi import ViTokenizer

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

# Khởi tạo Flask app
app = Flask(__name__)

# Load mô hình và tokenizer
model_path = 'models/phobert_best_model.pt'
tokenizer_path = 'models/phobert_tokenizer'
abbreviations_path = 'dataset/abbreviations.csv'

logger.info("Đang tải mô hình và tokenizer...")
try:
    model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    abbreviations = load_abbreviations(abbreviations_path)
except Exception as e:
    logger.error(f"Lỗi khi tải mô hình hoặc tokenizer: {e}")
    raise

# Chuyển mô hình lên thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
logger.info(f"Thiết bị đang sử dụng: {device}")
logger.info("Mô hình đã sẵn sàng!")

# Hàm dự đoán batch
def predict_batch(texts, model, tokenizer, device, abbreviations):
    # Tiền xử lý tất cả văn bản
    processed_texts = [preprocess_text(text, restore_accent=True, abbreviations=abbreviations) for text in texts]

    # Token hóa batch
    encodings = tokenizer(
        processed_texts,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    # Dự đoán batch
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_classes = torch.argmax(predictions, dim=1).cpu().numpy()
        confidences = predictions[torch.arange(len(texts)), predicted_classes].cpu().numpy()

    # Ánh xạ kết quả
    emotion_map = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
    results = [
        {
            "text": texts[i],
            "processed_text": processed_texts[i],
            "emotion": emotion_map[predicted_classes[i]],
            "confidence": float(confidences[i])
        }
        for i in range(len(texts))
    ]
    return results

# API endpoint cho dự đoán đơn lẻ
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data['text']
    try:
        result = predict_batch([text], model, tokenizer, device, abbreviations)[0]  # Gọi batch với list 1 phần tử
        return jsonify(result)
    except Exception as e:
        logger.error(f"Lỗi khi dự đoán: {e}")
        return jsonify({"error": str(e)}), 500

# API endpoint cho dự đoán batch
@app.route('/predict_batch', methods=['POST'])
def predict_batch_endpoint():
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({"error": "Missing 'texts' field (list of texts required)"}), 400

    texts = data['texts']
    if not isinstance(texts, list):
        return jsonify({"error": "'texts' must be a list"}), 400

    try:
        results = predict_batch(texts, model, tokenizer, device, abbreviations)
        return jsonify({"predictions": results})
    except Exception as e:
        logger.error(f"Lỗi khi dự đoán batch: {e}")
        return jsonify({"error": str(e)}), 500

# API endpoint kiểm tra trạng thái
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OK", "device": str(device)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)