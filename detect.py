import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import torch
from PIL import Image
import requests
import io

app = FastAPI()

# Semaphore để giới hạn số lượng yêu cầu đồng thời
semaphore = asyncio.Semaphore(10)

# Hàng đợi yêu cầu
request_queue = asyncio.Queue()

# Schema để định nghĩa dữ liệu đầu vào và đầu ra
class ImageInput(BaseModel):
    url: str 

class DetectionResult(BaseModel):
    label: str  # Nhãn của đối tượng
    coordinates: Tuple[float, float, float, float]  # Bounding box (x_min, y_min, x_max, y_max)
    confidence: float  # Mức độ tin cậy

class DetectionResponse(BaseModel):
    count: int  # Tổng số người phát hiện được
    detections: List[DetectionResult]  # Danh sách các bounding box

# Tải mô hình YOLOv5
MODEL_PATH = "yolov5s.pt"  # Đường dẫn mô hình
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)  # Tải YOLOv5
print("Loading YOLOv5 model...")

# Xử lý hàng đợi trong background
MAX_QUEUE_SIZE = 1000  # Số lượng yêu cầu tối đa trong hàng đợi
async def process_queue():
    while True:
        if request_queue.qsize() > MAX_QUEUE_SIZE:
            print("Queue is full, waiting for space...")
            await asyncio.sleep(1)  # Tạm dừng để đợi không gian trong hàng đợi
        request_data = await request_queue.get()
        image_data, response_future = request_data
        async with semaphore:  # Chỉ cho phép 3 yêu cầu đồng thời
            try:
                # Tải ảnh từ URL
                response = requests.get(image_data.url)
                response.raise_for_status()  # Kiểm tra lỗi HTTP
                img = Image.open(io.BytesIO(response.content))  # Đọc ảnh từ bytes
                # Phân tích bằng YOLOv5
                results = model(img)

                # Lọc kết quả chỉ lấy nhãn "person"
                detections = []
                count = 0
                for det in results.xyxy[0]:  # Duyệt qua từng kết quả
                    x_min, y_min, x_max, y_max, conf, cls = det.tolist()
                    label = model.names[int(cls)]
                    if label == "person":  # Lọc chỉ lấy nhãn "person"
                        count += 1
                        detections.append(DetectionResult(
                            label=label,
                            coordinates=(x_min, y_min, x_max, y_max),
                            confidence=conf,
                        ))

                # Gửi kết quả về
                response = DetectionResponse(count=count, detections=detections)
                response_future.set_result(response)

            except Exception as e:
                response_future.set_exception(HTTPException(status_code=500, detail=f"Error processing image: {str(e)}"))

            # Đánh dấu yêu cầu đã được xử lý
            request_queue.task_done()

# Khởi chạy background task để xử lý hàng đợi
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue())

@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(image: ImageInput):
    # Tạo một Future để nhận kết quả
    print(f"Received URL: {image.url}")
    response_future = asyncio.get_event_loop().create_future()

    # Đưa yêu cầu vào hàng đợi
    await request_queue.put((image, response_future))

    # Chờ kết quả từ hàng đợi
    response = await response_future
    return response
