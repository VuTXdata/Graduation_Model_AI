import os
import queue
import random
import shutil
import threading
import time
import emoji
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
import re

# Khóa để đồng bộ hóa việc ghi file
file_lock = threading.Lock()


def clean_text(text):
    """Làm sạch văn bản bằng cách xóa dấu xuống dòng và chuẩn hóa khoảng trắng"""
    if not isinstance(text, str):
        return str(text)

    # Xóa ký tự xuống dòng
    text = text.replace('\n', ' ').replace('\r', ' ')

    # Thay thế nhiều khoảng trắng liên tiếp bằng một khoảng trắng
    text = re.sub(r'\s+', ' ', text)

    # Xóa khoảng trắng ở đầu và cuối chuỗi
    return text.strip()


def initialize_webdriver(chromedriver_path=None):
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


def process_text_with_copilot(comments_batch, prompt_template, chromedriver_path):
    """Sử dụng Microsoft Copilot để xử lý nhiều câu văn bản cùng lúc"""
    driver = initialize_webdriver(chromedriver_path)
    try:
        driver.get("https://copilot.microsoft.com/chats")
        time.sleep(5)

        # Kiểm tra URL và chuyển hướng nếu cần
        current_url = driver.current_url
        if "https://copilot.microsoft.com/onboarding" in current_url:
            print(
                f"Trang hiện tại là onboarding ({current_url}). Chuyển hướng đến https://copilot.microsoft.com/chats...")
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
                    return None

        input_area.click()
        time.sleep(random.uniform(0.5, 1))
        input_area.clear()

        # Chuyển đổi emoji trong danh sách comments
        comments_with_text_emoji = []
        for comment in comments_batch:
            comments_with_text_emoji.append(emoji.demojize(comment))

        # Tạo chuỗi đầu vào bằng cách nối các comment với dấu | thay vì dấu -
        input_comments = "|".join(comments_with_text_emoji)

        # In ra giá trị đã ghép
        print(f"Dữ liệu ghép 5 dòng gửi lên Copilot: {input_comments}")

        full_prompt = prompt_template + f'"{input_comments}"'

        # Nhập prompt từng ký tự để tránh phát hiện
        input_area.send_keys(full_prompt[0])
        time.sleep(random.uniform(0.5, 1))
        input_area.send_keys(full_prompt[1:])
        time.sleep(random.uniform(0.5, 1))

        submit_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-testid='submit-button']")))
        submit_button.click()

        print(f"Đang đợi Copilot trả lời cho {len(comments_batch)} câu... Đợi 30 giây để phản hồi ổn định.")
        time.sleep(20)  # Tăng thời gian chờ cho batch lớn

        wait = WebDriverWait(driver, 45)  # Tăng timeout cho batch lớn
        response_elements = wait.until(EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "span.font-ligatures-none")
        ))

        full_response = response_elements[0].text.strip()

        # In ra đầy đủ phản hồi từ Copilot
        print(f"Phản hồi đầy đủ từ Copilot: {full_response}")

        # Xử lý kết quả để tách các phần trả lời
        results = []
        if "|" in full_response:
            parts = full_response.split("|")
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                if "(" in part and ")" in part:
                    try:
                        corrected_text = part.split("(")[0].strip()
                        sentiment_task = part.split("(")[1].split(")")[0].strip()
                        if "-" in sentiment_task:
                            sentiment, task = sentiment_task.split("-")
                            results.append({
                                "corrected_text": corrected_text,
                                "label": sentiment if sentiment in ['0', '1', '2'] else "1",
                                "task": task if task in '123456789' else "9"
                            })
                        else:
                            results.append({
                                "corrected_text": corrected_text,
                                "label": "1",
                                "task": "9"
                            })
                    except Exception as e:
                        print(f"Lỗi khi phân tích phần tử '{part}': {e}")
                        results.append({
                            "corrected_text": part,
                            "label": "1",
                            "task": "9"
                        })
                else:
                    results.append({
                        "corrected_text": part,
                        "label": "1",
                        "task": "9"
                    })
        else:
            # Trường hợp không có dấu "|" - có thể là lỗi hoặc chỉ có một kết quả
            if "(" in full_response and ")" in full_response:
                try:
                    corrected_text = full_response.split("(")[0].strip()
                    sentiment_task = full_response.split("(")[1].split(")")[0].strip()
                    if "-" in sentiment_task:
                        sentiment, task = sentiment_task.split("-")
                        results.append({
                            "corrected_text": corrected_text,
                            "label": sentiment if sentiment in ['0', '1', '2'] else "1",
                            "task": task if task in '123456789' else "9"
                        })
                    else:
                        results.append({
                            "corrected_text": corrected_text,
                            "label": "1",
                            "task": "9"
                        })
                except Exception as e:
                    print(f"Lỗi khi phân tích phản hồi đơn: {e}")
                    results.append({
                        "corrected_text": full_response,
                        "label": "1",
                        "task": "9"
                    })
            else:
                results.append({
                    "corrected_text": full_response,
                    "label": "1",
                    "task": "9"
                })

        # Đảm bảo kết quả có đủ số phần tử so với đầu vào
        while len(results) < len(comments_batch):
            missing_index = len(results)
            print(f"Thiếu kết quả cho câu thứ {missing_index + 1}, sử dụng giá trị mặc định")
            results.append({
                "corrected_text": comments_batch[missing_index],
                "label": "1",
                "task": "9"
            })

        # Cắt bớt kết quả nếu nhiều hơn cần thiết
        if len(results) > len(comments_batch):
            results = results[:len(comments_batch)]

        return results

    except (TimeoutException, NoSuchElementException) as e:
        print(f"Không tìm thấy phần tử HTML cần thiết: {e}")
        return None
    except Exception as e:
        print(f"Lỗi khi xử lý với Copilot: {e}")
        return None
    finally:
        driver.quit()


def worker(thread_id, data_queue, result_queue, prompt_template, chromedriver_path):
    """Hàm xử lý cho mỗi luồng - xử lý batch comment"""
    print(f"Luồng {thread_id} đã bắt đầu")
    while True:
        try:
            batch_data = data_queue.get(block=False)
            if not batch_data:
                data_queue.task_done()
                break

            comments = []
            indices = []
            row_data_list = []

            for idx, comment, row_data in batch_data:
                comments.append(comment)
                indices.append(idx)
                row_data_list.append(row_data)

            print(f"Luồng {thread_id} đang xử lý batch với {len(comments)} comment...")

            results = process_text_with_copilot(comments, prompt_template, chromedriver_path)

            if results is None:
                # Nếu xử lý thất bại, gửi kết quả mặc định
                for i in range(len(comments)):
                    new_row = row_data_list[i].copy()
                    new_row['comment'] = comments[i]
                    new_row['label'] = "1"
                    new_row['task'] = "9"
                    result_queue.put((indices[i], new_row))
            else:
                # Xử lý kết quả thành công
                for i in range(len(comments)):
                    if i < len(results):
                        new_row = row_data_list[i].copy()
                        new_row['comment'] = results[i]['corrected_text']
                        new_row['label'] = results[i]['label']
                        new_row['task'] = results[i]['task']
                        result_queue.put((indices[i], new_row))
                    else:
                        # Phòng trường hợp thiếu kết quả
                        new_row = row_data_list[i].copy()
                        new_row['comment'] = comments[i]
                        new_row['label'] = "1"
                        new_row['task'] = "9"
                        result_queue.put((indices[i], new_row))

            data_queue.task_done()
            time.sleep(random.uniform(2, 5))

        except queue.Empty:
            break
        except Exception as e:
            print(f"Luồng {thread_id} gặp lỗi: {e}")
            # Đảm bảo task được đánh dấu là hoàn thành ngay cả khi có lỗi
            if 'batch_data' in locals():
                for idx, comment, row_data in batch_data:
                    new_row = row_data.copy()
                    new_row['comment'] = comment
                    new_row['label'] = "1"
                    new_row['task'] = "9"
                    result_queue.put((idx, new_row))
                data_queue.task_done()
    print(f"Luồng {thread_id} đã kết thúc")


def write_results_to_file(output_file, result_queue, total_rows, original_data):
    """Ghi kết quả vào file đầu ra"""
    processed_rows = []
    completed = 0
    print("Đang bắt đầu ghi kết quả vào file...")
    last_write_time = time.time()

    # Đọc dữ liệu hiện có từ file nếu file tồn tại
    existing_data = pd.DataFrame(columns=original_data.columns)
    if os.path.exists(output_file):
        try:
            existing_data = pd.read_csv(output_file)
            print(f"Đã đọc {len(existing_data)} dòng từ file hiện có")
        except Exception as e:
            print(f"Lỗi khi đọc file hiện có: {e}")

    while completed < total_rows:
        try:
            row_index, row_data = result_queue.get(timeout=1)
            processed_rows.append(row_data)
            completed += 1

            current_time = time.time()
            if current_time - last_write_time >= 10 or completed % 5 == 0:
                with file_lock:
                    # Tạo DataFrame mới từ các dòng đã xử lý
                    updated_data = pd.DataFrame(processed_rows, columns=original_data.columns)

                    # Nối với dữ liệu hiện có và lưu
                    if not existing_data.empty:
                        combined_data = pd.concat([existing_data, updated_data], ignore_index=True)
                        combined_data.to_csv(output_file, index=False)
                    else:
                        updated_data.to_csv(output_file, index=False)

                    last_write_time = current_time

            print(f"Đã xử lý {completed}/{total_rows} dòng. Xử lý dòng {row_index}:")
            print(f"  - Comment mới: {row_data['comment'][:50]}..." if len(
                row_data['comment']) > 50 else f"  - Comment mới: {row_data['comment']}")
            print(f"  - Label: {row_data['label']} - Task: {row_data['task']}")

        except queue.Empty:
            current_time = time.time()
            if current_time - last_write_time >= 10 and processed_rows:
                with file_lock:
                    updated_data = pd.DataFrame(processed_rows, columns=original_data.columns)
                    if not existing_data.empty:
                        combined_data = pd.concat([existing_data, updated_data], ignore_index=True)
                        combined_data.to_csv(output_file, index=False)
                    else:
                        updated_data.to_csv(output_file, index=False)
                    last_write_time = current_time
            continue
        except Exception as e:
            print(f"Lỗi khi ghi kết quả: {e}")

    with file_lock:
        updated_data = pd.DataFrame(processed_rows, columns=original_data.columns)
        if not existing_data.empty:
            combined_data = pd.concat([existing_data, updated_data], ignore_index=True)
            combined_data.to_csv(output_file, index=False)
        else:
            updated_data.to_csv(output_file, index=False)

    print("Đã hoàn thành việc ghi kết quả vào file!")


def check_previous_progress(output_file):
    """Kiểm tra tiến trình xử lý trước đó"""
    try:
        if os.path.exists(output_file):
            print(f"Tìm thấy file đã xử lý trước đó: {output_file}")
            existing_data = pd.read_csv(output_file)
            if not existing_data.empty:
                print(f"Tìm thấy {len(existing_data)} dòng đã xử lý trước đó")
                return existing_data
            else:
                print("Không tìm thấy dòng nào đã xử lý trước đó")
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

    # Yêu cầu người dùng nhập dòng bắt đầu xử lý
    start_row = int(input("Nhập số dòng bắt đầu xử lý (0 để bắt đầu từ đầu): "))

    # Yêu cầu người dùng nhập đường dẫn file đầu ra
    output_file = input("Nhập đường dẫn file đầu ra (Enter để sử dụng tên mặc định): ")
    if not output_file:
        output_file = os.path.splitext(file_path)[0] + "_processed.csv"

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

    # Làm sạch dữ liệu trong tất cả các cột
    for column in data.columns:
        data[column] = data[column].apply(clean_text)

    # Nếu start_row > 0, chỉ lấy các dòng từ start_row trở đi
    if start_row > 0:
        if start_row < len(data):
            data = data.iloc[start_row:].reset_index(drop=True)
            print(f"Bắt đầu xử lý từ dòng {start_row}, còn lại {len(data)} dòng")
        else:
            print(f"Lỗi: Số dòng bắt đầu ({start_row}) lớn hơn tổng số dòng trong file ({len(data)})")
            return

    processed_data = check_previous_progress(output_file)
    original_data = data.copy()

    # Cập nhật prompt mới theo yêu cầu
    prompt_template = """Tôi đang trong ngữ cảnh là xây dựng tập dataset của riêng mình phù hợp cho bài toán phân loại. Chủ đề ở đây là thương mại điện tử. Tôi có các task như sau tương ứng với văn bản nói về dịch vụ kinh doanh hay sản phẩm : 1. chất lượng, 2. đóng gói, 3. giao hàng, 4. giá thành, 5. Hỗ trợ khách hàng, 6. Đặt hàng, 7. Trả hàng, 8.Khuyến mãi, 9. Khác. Và các nhãn cảm xúc 0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực". Tôi đang cần bạn đầu tiên nếu câu tiếng việt này có từ viết tắt, viết sai chính tả thì hãy trả về cho tôi 1 câu tiếng việt đầy đủ và hoàn chỉnh, tiếp theo phân loại câu sau đây vào các nhãn trên cho đúng.  Hãy trả lời gắn gọn đúng trọng tâm. Đầu ra bạn chỉ cần trả lời chính xác theo định dạng sau (cảm xúc-task) duy nhất tương ứng với câu đó, không thêm bớt các câu khác vào để tôi dễ copy. Input đầu vào sẽ là dạng như sau cau1|cau2|cau3|cau4...., định dạng đầu ra sẽ là câu đã sửa1(cảm xúc1-task1)|câu đã sửa2(cảm xúc2-task2)| câu đã sửa3(cảm xúc3-task3)|....Nếu câu có nhiều hơn 2 ý nghĩa có thể tách câu ra theo chủ đề các task nhưng vẫn đảm bảo định dạng đầu ra như trên và đặc biệt trả về số lượng lớn hơn hoặc bằng với số câu tôi đã cung cấp cho bạn ví dụ: đầu vào là "sp nay chat luong tot|gia thanh hop ly|dong hang nhanh" định dạng đầu ra sẽ là sản phẩm này có chất lượng tốt(2-1)|giá thành hợp lý (2-4)|đóng hàng nhanh(2-2). Các câu tôi cần bạn giúp là """

    num_threads = int(input("Nhập số luồng (ví dụ: 5, 10): "))
    num_threads = min(num_threads, 10)

    data_queue = queue.Queue()
    result_queue = queue.Queue()

    # Tạo danh sách các comment đã xử lý (từ file hiện có)
    processed_ids = set()
    if not processed_data.empty and 'id' in processed_data.columns:
        processed_ids = set(processed_data['id'].tolist())
        print(f"Đã tìm thấy {len(processed_ids)} ID đã xử lý trước đó")

    # Nhóm các dòng thành các batch 5 dòng thay vì 10 dòng
    batch_size = 5
    current_batch = []
    batch_count = 0

    for idx, row in data.iterrows():
        row_id = row['id']
        if row_id not in processed_ids:
            current_batch.append((idx, row['comment'], row.to_dict()))

            if len(current_batch) >= batch_size:
                data_queue.put(current_batch)
                batch_count += 1
                current_batch = []

    # Thêm batch cuối cùng nếu còn
    if current_batch:
        data_queue.put(current_batch)
        batch_count += 1

    total_rows = batch_count * batch_size
    if data_queue.empty():
        print("Tất cả các dòng đã được xử lý trước đó. Không có công việc mới.")
        return

    print(f"Cần xử lý {total_rows} dòng mới, chia thành {batch_count} batch")

    threads = []
    for i in range(min(num_threads, batch_count)):
        thread = threading.Thread(
            target=worker,
            args=(i + 1, data_queue, result_queue, prompt_template, chromedriver_path)
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)

    writer_thread = threading.Thread(
        target=write_results_to_file,
        args=(output_file, result_queue, total_rows, original_data)
    )
    writer_thread.daemon = True
    writer_thread.start()

    for thread in threads:
        thread.join()

    writer_thread.join()

    print(f"Đã hoàn thành! Kết quả được lưu tại: {output_file}")


if __name__ == "__main__":
    main()