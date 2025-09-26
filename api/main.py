import os
import sys

# Ensure project root is on sys.path (Option B)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import re
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, List, Any
import motor.motor_asyncio
from bson import ObjectId
from pydantic_core import core_schema
from pydantic.json_schema import JsonSchemaValue
from pydantic._internal._generate_schema import GetJsonSchemaHandler
from contextlib import asynccontextmanager

import torch
from ultralytics import YOLO
from torchvision import transforms

from ocr.model import EnhancedCRNN
from ocr.utils import decode_output, correct_ocr_text, deskew_image, denoise_image

import cv2
import numpy as np
from PIL import Image

# Globals
yolo_model = None
crnn_model = None
ocr_transform = None
device = None

# MongoDB connection variables (defined outside of lifespan for global access)
MONGO_URL = "mongodb://localhost:27017"
client = None
database = None
collection = None

YOLO_CLASS_MAP = {
    0: "8051",
    1: "ARDUINO_NANO_ATMEGA328P",
    2: "ARMCORTEXM3",
    3: "ARMCORTEXM7",
    4: "ESP32_DEVKIT",
    5: "NODEMCU_ESP8266",
    6: "RASPBERRY_PI_3B_PLUS"
}


# Pydantic v2 compatible PyObjectId
class PyObjectId(str):
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: Any) -> core_schema.CoreSchema:
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema([
                core_schema.is_instance_schema(ObjectId),
                core_schema.chain_schema([
                    core_schema.str_schema(),
                    core_schema.no_info_plain_validator_function(cls.validate),
                ]),
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x), when_used="json"
            ),
        )

    @classmethod
    def validate(cls, value) -> ObjectId:
        if not ObjectId.is_valid(value):
            raise ValueError("Invalid ObjectId")
        return ObjectId(value)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return handler(core_schema.str_schema())


# Pydantic model for datasheet validation
class DatasheetModel(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    part_number: str
    manufacturer: str
    core: str
    flash: str
    ram: str
    max_clock: str
    datasheet_url: str
    features: List[str]

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str}
    }


async def find_datasheet_by_text(ocr_text: str):
    """
    Find datasheet in database using OCR text with fuzzy/simple matching
    """
    if not ocr_text or len(ocr_text.strip()) < 2:
        return None

    # 1) Exact (case-insensitive)
    exact_match = await collection.find_one({
        "part_number": {"$regex": f"^{re.escape(ocr_text)}$", "$options": "i"}
    })
    if exact_match:
        return DatasheetModel(**exact_match)

    # 2) Partial (limited)
    partial_matches = []
    cursor = collection.find({
        "$or": [
            {"part_number": {"$regex": ocr_text, "$options": "i"}},
            {"manufacturer": {"$regex": ocr_text, "$options": "i"}},
            {"core": {"$regex": ocr_text, "$options": "i"}}
        ]
    }).limit(5)

    async for doc in cursor:
        partial_matches.append(DatasheetModel(**doc))

    return partial_matches[0] if partial_matches else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, database, collection, yolo_model, crnn_model, ocr_transform, device
    try:
        # MongoDB
        client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
        database = client.microcontrollers
        collection = database.datasheets
        await client.admin.command('ping')
        print("✅ Successfully connected to MongoDB!")

        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {device}")

        # Models
        # Use path relative to project root
        yolo_weights_path = os.path.join(PROJECT_ROOT, "data", "runs", "detect", "train2", "weights", "best.pt")
        yolo_model = YOLO(yolo_weights_path)
        print("🔍 YOLO model loaded successfully!")

        crnn_model = EnhancedCRNN(img_height=32, num_channels=1, num_classes=38)
        crnn_weights_path = os.path.join(PROJECT_ROOT, "best_model.pth")
        if os.path.exists(crnn_weights_path):
            crnn_model.load_state_dict(torch.load(crnn_weights_path, map_location=device))
            print("📖 CRNN model weights loaded!")
        else:
            print("⚠️ No trained CRNN weights found, using random weights")

        crnn_model.to(device)
        crnn_model.eval()

        # OCR transform
        ocr_transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        print("🛠️ OCR transform pipeline ready!")

        count = await collection.count_documents({})
        print(f"📊 Found {count} datasheets in database")
        print("🚀 All systems ready!")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

    yield

    if client:
        client.close()
        print("🔌 Disconnected from MongoDB")


app = FastAPI(
    title="Microcontroller Datasheet API",
    description="API for retrieving microcontroller datasheets from MongoDB and AI OCR",
    version="1.0.0",
    lifespan=lifespan
)


async def process_image_with_ai(image_bytes: bytes):
    """
    Core AI processing: YOLO detection + CRNN OCR on uploaded image
    """
    try:
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if cv_image is None:
            raise ValueError("Invalid image format")

        # YOLO detection
        yolo_results = yolo_model(cv_image, verbose=False)
        detections = []

        if len(yolo_results) > 0 and hasattr(yolo_results[0], 'boxes') and len(yolo_results[0].boxes) > 0:
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                crop = cv_image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop_pil = Image.fromarray(crop_gray)

                # Preprocess (match training)
                crop_pil = deskew_image(crop_pil)
                crop_pil = denoise_image(crop_pil)
                crop_pil = crop_pil.resize((128, 32), Image.BILINEAR)

                # CRNN inference
                crop_tensor = ocr_transform(crop_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = crnn_model(crop_tensor)
                    raw_predictions = decode_output(logits)

                raw_text = raw_predictions[0] if raw_predictions else ""
                corrected_text = correct_ocr_text(raw_text)

                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": confidence,
                    "class_id": class_id,
                    "raw_ocr": raw_text,
                    "corrected_ocr": corrected_text
                })

        return detections

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Microcontroller Datasheet API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    # Optionally, actually test DB or return cached state
    return {"status": "healthy", "database": "connected"}


@app.get("/datasheet/{part_number}", response_model=DatasheetModel)
async def get_datasheet(part_number: str):
    datasheet = await collection.find_one({
        "part_number": {"$regex": f"^{re.escape(part_number)}$", "$options": "i"}
    })
    if datasheet:
        return DatasheetModel(**datasheet)
    raise HTTPException(status_code=404, detail=f"Datasheet for '{part_number}' not found")


@app.get("/datasheets/", response_model=List[DatasheetModel])
async def get_all_datasheets(skip: int = 0, limit: int = 10):
    cursor = collection.find().skip(skip).limit(limit)
    datasheets = await cursor.to_list(length=limit)
    return [DatasheetModel(**datasheet) for datasheet in datasheets]


@app.get("/datasheets/manufacturer/{manufacturer}", response_model=List[DatasheetModel])
async def get_datasheets_by_manufacturer(manufacturer: str):
    cursor = collection.find({
        "manufacturer": {"$regex": manufacturer, "$options": "i"}
    })
    datasheets = await cursor.to_list(length=None)
    return [DatasheetModel(**datasheet) for datasheet in datasheets]


@app.get("/search/", response_model=List[DatasheetModel])
async def search_datasheets(q: str):
    query = {
        "$or": [
            {"part_number": {"$regex": q, "$options": "i"}},
            {"manufacturer": {"$regex": q, "$options": "i"}},
            {"features": {"$regex": q, "$options": "i"}},
            {"core": {"$regex": q, "$options": "i"}}
        ]
    }
    cursor = collection.find(query)
    datasheets = await cursor.to_list(length=None)
    return [DatasheetModel(**datasheet) for datasheet in datasheets]


@app.post("/recognize")
async def recognize_microcontrollers(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    detections = await process_image_with_ai(image_bytes)

    return {
        "filename": file.filename,
        "detections_count": len(detections),
        "detections": detections
    }


@app.post("/recognize-and-resolve")
async def recognize_and_resolve(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    detections = await process_image_with_ai(image_bytes)

    enriched_detections = []
    for detection in detections:
        ocr_text = detection.get("corrected_ocr", "") or detection.get("raw_ocr", "")
        # Fallback: use YOLO class_id if OCR fails
        if not ocr_text:
            part_number = YOLO_CLASS_MAP.get(detection["class_id"], None)
        else:
            part_number = ocr_text
        datasheet = await find_datasheet_by_text(part_number) if part_number else None
        datasheet_dict = datasheet.dict() if datasheet else None
        # --- Fix: Convert _id to string if present ---
        if datasheet_dict and "_id" in datasheet_dict:
            datasheet_dict["_id"] = str(datasheet_dict["_id"])
        enriched_detection = {
            **detection,
            "datasheet": datasheet_dict,
            "datasheet_found": datasheet is not None
        }
        enriched_detections.append(enriched_detection)

    return {
        "filename": file.filename,
        "detections_count": len(enriched_detections),
        "detections": enriched_detections,
        "summary": {
            "total_detected": len(detections),
            "datasheets_found": sum(1 for d in enriched_detections if d["datasheet_found"])
        }
    }


if __name__ == "__main__":
    import uvicorn
    # Safe to run from project root or api/ with Option B path setup
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
