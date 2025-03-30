import os
import queue
import random
import shutil
import threading
import time
import emoji  # Thêm thư viện emoji
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# Khóa để đồng bộ hóa việc ghi file
file_lock = threading.Lock()

def initialize_webdriver(chromedriver_path=None):
    """Khởi tạo trình duyệt Selenium với cấu hình chống phát hiện bot"""
    """Khởi tạo trình duyệt Selenium với cấu hình chống phát hiện bot và chạy ngầm"""
    chrome_options = Options()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--headless")  # Chạy ngầm, không hiển thị giao diện
    chrome_options.add_argument("--no-sandbox")  # Cần thiết khi chạy headless trên một số hệ thống
    chrome_options.add_argument("--disable-dev-shm-usage")  # Khắc phục lỗi bộ nhớ khi chạy headless
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--allow-insecure-localhost")

    # chrome_options = Options()
    # chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    # chrome_options.add_argument("--start-maximized")
    # chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    # chrome_options.add_experimental_option("useAutomationExtension", False)
    # chrome_options.add_argument("--ignore-certificate-errors")
    # chrome_options.add_argument("--allow-insecure-localhost")
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
    ]
    chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")
    if chromedriver_path:
        driver = webdriver.Chrome(service=Service(chromedriver_path), options=chrome_options)
    else:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def process_text_with_copilot(comment, prompt_template, chromedriver_path):
    """Sử dụng Microsoft Copilot để xử lý văn bản"""
    driver = initialize_webdriver(chromedriver_path)
    try:
        driver.get("https://copilot.microsoft.com/chats")
        time.sleep(5)

        # Kiểm tra URL và chuyển hướng nếu cần
        current_url = driver.current_url
        if "https://copilot.microsoft.com/onboarding" in current_url:
            print(f"Trang hiện tại là onboarding ({current_url}). Chuyển hướng đến https://copilot.microsoft.com/chats...")
            driver.get("https://copilot.microsoft.com/chats")
            time.sleep(5)  # Đợi trang mới tải xong

        # Thử tìm ô input, nếu không thấy thì load lại trang
        wait = WebDriverWait(driver, 15)
        max_attempts = 2  # Thử tối đa 2 lần
        for attempt in range(max_attempts):
            try:
                input_area = wait.until(EC.presence_of_element_located((By.ID, "userInput")))
                break
            except (TimeoutException, NoSuchElementException) as e:
                if attempt == 0:
                    print(f"Không tìm thấy ô nhập prompt: {e}. Đang load lại trang...")
                    driver.get("https://copilot.microsoft.com/chats")
                    time.sleep(5)
                else:
                    print(f"Lần thử {attempt + 1} thất bại: {e}. Trả về giá trị mặc định.")
                    return comment, "1", "9"

        input_area.click()
        time.sleep(random.uniform(0.5, 1))
        input_area.clear()
        comment_with_text_emoji = emoji.demojize(comment)
        full_prompt = prompt_template + f'"{comment_with_text_emoji}"'
        input_area.send_keys(full_prompt[0])
        time.sleep(random.uniform(0.5, 1))
        input_area.send_keys(full_prompt[1:])
        time.sleep(random.uniform(0.5, 1))
        submit_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-testid='submit-button']")))
        submit_button.click()
        print(f"Đang đợi Copilot trả lời cho comment: {comment[:30]}... Đợi 15 giây để phản hồi ổn định.")
        time.sleep(10)
        wait = WebDriverWait(driver, 30)
        response_elements = wait.until(EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "span.font-ligatures-none")
        ))

        full_response = response_elements[0].text.strip()

        # Phân tích kết quả theo định dạng "câu(cảm xúc-task)"
        corrected_text = comment
        label = "1"  # Trung tính mặc định
        task_number = "9"  # Khác mặc định

        if '(' in full_response and ')' in full_response:
            corrected_text = full_response.split('(')[0].strip()
            sentiment_task = full_response.split('(')[1].split(')')[0].strip()
            if '-' in sentiment_task:
                sentiment, task = sentiment_task.split('-')
                if sentiment in ['0', '1', '2']:
                    label = sentiment
                if task in '123456789':
                    task_number = task

        print(f"Đã nhận phản hồi: '{corrected_text[:30]}...' - Label: {label} - Task: {task_number}")
        return corrected_text, label, task_number

    except (TimeoutException, NoSuchElementException) as e:
        print(f"Không tìm thấy phần tử HTML cần thiết: {e}")
        return comment, "1", "9"
    except Exception as e:
        print(f"Lỗi khi xử lý với Copilot: {e}")
        return comment, "1", "9"
    finally:
        driver.quit()

def split_sentences(comment):
    """Tách các câu trong một ô comment, giả định câu kết thúc bằng dấu chấm, chấm than hoặc hỏi"""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', comment.strip())
    return [s.strip() for s in sentences if s.strip()]

def worker(thread_id, data_queue, result_queue, prompt_template, chromedriver_path):
    """Hàm xử lý cho mỗi luồng"""
    print(f"Luồng {thread_id} đã bắt đầu")
    while True:
        try:
            row_index, sentence, row_data = data_queue.get(block=False)
            print(f"Luồng {thread_id} đang xử lý dòng {row_index}, câu: {sentence[:30]}...")
            corrected_comment, label, task = process_text_with_copilot(sentence, prompt_template, chromedriver_path)
            new_row = row_data.copy()
            new_row['comment'] = corrected_comment
            new_row['label'] = label
            new_row['task'] = task
            result_queue.put((row_index, new_row))
            data_queue.task_done()
            time.sleep(random.uniform(2, 5))

        except queue.Empty:
            break
        except Exception as e:
            print(f"Luồng {thread_id} gặp lỗi: {e}")
            data_queue.task_done()
    print(f"Luồng {thread_id} đã kết thúc")

def write_results_to_file(output_file, result_queue, total_sentences, original_data):
    processed_rows = []
    completed = 0
    print("Đang bắt đầu ghi kết quả vào file...")
    last_write_time = time.time()

    while completed < total_sentences:
        try:
            row_index, row_data = result_queue.get(timeout=1)
            processed_rows.append(row_data)
            completed += 1

            current_time = time.time()
            if current_time - last_write_time >= 10 or completed % 5 == 0:
                with file_lock:
                    updated_data = pd.DataFrame(processed_rows, columns=original_data.columns)
                    updated_data.to_csv(output_file, index=False)
                    last_write_time = current_time

            print(f"Đã xử lý {completed}/{total_sentences} câu. Xử lý dòng {row_index}:")
            print(f"  - Comment mới: {row_data['comment'][:50]}..." if len(row_data['comment']) > 50 else f"  - Comment mới: {row_data['comment']}")
            print(f"  - Label: {row_data['label']} - Task: {row_data['task']}")

        except queue.Empty:
            current_time = time.time()
            if current_time - last_write_time >= 10 and processed_rows:
                with file_lock:
                    updated_data = pd.DataFrame(processed_rows, columns=original_data.columns)
                    updated_data.to_csv(output_file, index=False)
                    last_write_time = current_time
            continue
        except Exception as e:
            print(f"Lỗi khi ghi kết quả: {e}")

    with file_lock:
        updated_data = pd.DataFrame(processed_rows, columns=original_data.columns)
        updated_data.to_csv(output_file, index=False)

    print("Đã hoàn thành việc ghi kết quả vào file!")

def check_previous_progress(output_file):
    try:
        if os.path.exists(output_file):
            print(f"Tìm thấy file đã xử lý trước đó: {output_file}")
            existing_data = pd.read_csv(output_file)
            if not existing_data.empty:
                print(f"Tìm thấy {len(existing_data)} câu đã xử lý trước đó")
                return existing_data
            else:
                print("Không tìm thấy câu nào đã xử lý trước đó")
    except Exception as e:
        print(f"Lỗi khi kiểm tra tiến trình trước đó: {e}")
    return pd.DataFrame(columns=['id', 'comment', 'label', 'task'])

def main():
    cache_dir = os.path.expanduser("~/.wdm")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("Đã xóa bộ nhớ cache cũ của webdriver_manager")

    chromedriver_path = ChromeDriverManager().install()
    print(f"ChromeDriver đã được cài đặt tại: {chromedriver_path}")

    print("Chọn loại file đầu vào:")
    print("1. CSV")
    print("2. Excel")
    file_type = input("Nhập lựa chọn (1 hoặc 2): ")

    delimiter = ','
    if file_type == '1':
        delimiter = input("Nhập ký tự phân cách (nhấn Enter để dùng dấu phẩy mặc định): ") or ','

    file_path = input("Nhập đường dẫn đến file: ")

    data = None
    try:
        if file_type == '1':
            data = pd.read_csv(file_path, delimiter=delimiter)
        else:
            data = pd.read_excel(file_path)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    required_columns = ['id', 'comment', 'label']
    for col in required_columns:
        if col not in data.columns:
            print(f"Lỗi: Thiếu cột {col} trong tập dữ liệu")
            return

    if 'task' not in data.columns:
        data['task'] = "9"

    output_file = os.path.splitext(file_path)[0] + "_processed.csv"
    processed_data = check_previous_progress(output_file)
    original_data = data.copy()

    prompt_template = """Tôi đang trong ngữ cảnh là xây dựng tập dataset của riêng mình phù hợp cho bài toán phân loại. Chủ đề ở đây là thương mại điện tử. Tôi có các task như sau tương ứng với văn bản nói về dịch vụ kinh doanh hay sản phẩm : 1. chất lượng, 2. đóng gói, 3. giao hàng, 4. giá thành, 5. Hỗ trợ khách hàng, 6. Đặt hàng, 7. Trả hàng, 8.Khuyến mãi, 9. Khác. Và các nhãn cảm xúc 0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực". Tôi đang cần bạn đầu tiên nếu câu tiếng việt này có từ viết tắt, viết sai chính tả thì hãy trả về cho tôi 1 câu tiếng việt đầy đủ và hoàn chỉnh, tiếp theo phân loại câu sau đây vào các nhãn trên cho đúng.  Hãy trả lời gắn gọn đúng trọng tâm. Nếu câu có nhiều hơn 2 ý nghĩa. Thì hãy để ý nghĩa chung nhất bao hàm toàn bộ ý của câu. Ví dụ: "sp nay co chat luong tot" đầu ra bạn chỉ cần trả lời chính xác theo định dạng sau (cảm xúc-task) duy nhất, không thêm bớt các câu khác vào để tôi dễ copy và chỉ chứa 1 cặp: câu đã sửa(cảm xúc-task) ví dụ: sản phẩm này có chất lượng tốt(2-1). Câu tôi cần bạn giúp là """

    num_threads = int(input("Nhập số luồng (ví dụ: 5, 10): "))
    num_threads = min(num_threads, 10)

    data_queue = queue.Queue()
    result_queue = queue.Queue()

    total_sentences = 0
    processed_sentences = set(processed_data['comment'].tolist()) if not processed_data.empty else set()

    for index, row in data.iterrows():
        sentences = split_sentences(row['comment'])
        for sentence in sentences:
            if sentence not in processed_sentences:
                data_queue.put((index, sentence, row.to_dict()))
                total_sentences += 1

    if data_queue.empty():
        print("Tất cả các câu đã được xử lý trước đó. Không có công việc mới.")
        return

    print(f"Cần xử lý {total_sentences} câu mới")

    threads = []
    for i in range(min(num_threads, total_sentences)):
        thread = threading.Thread(
            target=worker,
            args=(i + 1, data_queue, result_queue, prompt_template, chromedriver_path)
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)

    writer_thread = threading.Thread(
        target=write_results_to_file,
        args=(output_file, result_queue, total_sentences, original_data)
    )
    writer_thread.daemon = True
    writer_thread.start()

    for thread in threads:
        thread.join()

    writer_thread.join()

    print(f"Đã hoàn thành! Kết quả được lưu tại: {output_file}")

if __name__ == "__main__":
    main()