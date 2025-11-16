import os
import sys
# Ensure project root is on sys.path (Option B)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
import re
from contextlib import asynccontextmanager
from typing import Optional, List, Any, Dict


from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import motor.motor_asyncio
from bson import ObjectId
from pydantic_core import core_schema
from pydantic.json_schema import JsonSchemaValue
from pydantic._internal._generate_schema import GetJsonSchemaHandler
from dotenv import load_dotenv

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
name_to_part: Dict[str, str] = {}

# MongoDB connection variables
MONGO_URL = "mongodb://localhost:27017"
client = None
database = None
collection = None

load_dotenv()

API_KEY = os.getenv("API_KEY", "dev-key")
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
CONF_THRESH = float(os.getenv("CONF_THRESH", "0.85"))
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", os.path.join(PROJECT_ROOT, "data", "runs", "detect", "train2", "weights", "best.pt"))
CRNN_WEIGHTS = os.getenv("CRNN_WEIGHTS", os.path.join(PROJECT_ROOT, "best_model.pth"))
MIN_CONF_FOR_FALLBACK = CONF_THRESH

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
    }

    # Ensure the dumped dict is JSON-safe (convert ObjectId to str and use alias _id)
    def model_dump(self, **kwargs):
        if "by_alias" not in kwargs:
            kwargs["by_alias"] = True
        d = super().model_dump(**kwargs)
        if "_id" in d and isinstance(d["_id"], ObjectId):
            d["_id"] = str(d["_id"])
        if "id" in d and isinstance(d["id"], ObjectId):
            d["id"] = str(d["id"])
        return d

def _coerce_id_to_str(d: dict) -> dict:
    if "_id" in d and isinstance(d["_id"], ObjectId):
        d["_id"] = str(d["_id"])
    return d

async def find_datasheet_by_text(ocr_text: str):
    """Return a plain dict for JSON embedding; try exact then partial."""
    if not ocr_text or len(ocr_text.strip()) < 2:
        return None

    exact = await collection.find_one({
        "part_number": {"$regex": f"^{re.escape(ocr_text)}$", "$options": "i"}
    })
    if exact:
        return _coerce_id_to_str(exact)

    cursor = collection.find({
        "$or": [
            {"part_number": {"$regex": ocr_text, "$options": "i"}},
            {"manufacturer": {"$regex": ocr_text, "$options": "i"}},
            {"core": {"$regex": ocr_text, "$options": "i"}}
        ]
    }).limit(5)

    async for doc in cursor:
        return _coerce_id_to_str(doc)

    return None

# helpers
def normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())

def canon_pn(s: str) -> str:
    s = s.upper().strip()
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s

def vendor_variants(pn: str):
    # Example heuristics, expand per vendor
    v = [pn]
    # STM32 package suffix trim (e.g., T6)
    v.append(re.sub(r'(T[0-9A-Z]+)$', '', pn))
    # ESP32 modules collapse E/D suffix
    v.append(re.sub(r'(-32[E|D])$', '-32', pn))
    # Deduplicate and keep non-empty
    return list(dict.fromkeys([x for x in v if x]))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, database, collection, yolo_model, crnn_model, ocr_transform, device, name_to_part
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
        print("🔗 Using weights:", YOLO_WEIGHTS, CRNN_WEIGHTS)
        yolo_model = YOLO(YOLO_WEIGHTS)
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

        # Build dynamic YOLO→DB mapping (no hard-coded map)
        rows = await collection.find().to_list(length=None)
        part_nums = [r["part_number"] for r in rows]
        name_to_part = {normalize(p): p for p in part_nums}
        for cid, cname in yolo_model.names.items():
            key = normalize(cname)
            if key not in name_to_part:
                print(f"⚠️ No DB part_number match for YOLO class '{cname}' (id {cid})")

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

# CORS for mobile app and web clients; restrict origins later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async def require_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

async def process_image_with_ai(image_bytes: bytes):
    """Core AI processing: YOLO detection + CRNN OCR on uploaded image"""
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
                # Extract box coordinates, confidence, class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = yolo_model.names.get(class_id, f"Class_{class_id}")
                # Crop and prepare for OCR
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
                    "label": class_name, 
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
    db_status = "connected"
    try:
        if client is None:
            db_status = "disconnected"
        else:
            await client.admin.command("ping")
            db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return {"status": "healthy", "database": db_status}

@app.get("/datasheet/{part_number}")
async def get_datasheet(part_number: str):
    datasheet = await collection.find_one({
        "part_number": {"$regex": f"^{re.escape(part_number)}$", "$options": "i"}
    })
    if datasheet:
        return _coerce_id_to_str(datasheet)
    raise HTTPException(status_code=404, detail=f"Datasheet for '{part_number}' not found")

@app.get("/datasheets/")
async def get_all_datasheets(skip: int = 0, limit: int = 10):
    cursor = collection.find().skip(skip).limit(limit)
    datasheets = await cursor.to_list(length=limit)
    return [_coerce_id_to_str(d) for d in datasheets]

@app.get("/datasheets/manufacturer/{manufacturer}")
async def get_datasheets_by_manufacturer(manufacturer: str):
    cursor = collection.find({
        "manufacturer": {"$regex": manufacturer, "$options": "i"}
    })
    datasheets = await cursor.to_list(length=None)
    return [_coerce_id_to_str(d) for d in datasheets]

@app.get("/search/")
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
    return [_coerce_id_to_str(d) for d in datasheets]

@app.post("/recognize")
async def recognize_microcontrollers(file: UploadFile = File(...), _: None = Depends(require_api_key)):
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
async def recognize_and_resolve(file: UploadFile = File(...), _: None = Depends(require_api_key)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    detections = await process_image_with_ai(image_bytes)

    enriched_detections = []
    for detection in detections:
        # 1) Try OCR text first
        ocr_text = detection.get("corrected_ocr") or detection.get("raw_ocr") or ""
        part_number = ocr_text if ocr_text else None

        # 2) If no OCR text but high confidence, fallback to YOLO class name → DB mapping
        if not part_number and detection["confidence"] >= MIN_CONF_FOR_FALLBACK:
            class_name = yolo_model.names[detection["class_id"]]
            part_number = name_to_part.get(normalize(class_name))

        # 3) Canonicalize and try DB lookups with variants
        datasheet = None
        if part_number:
            pn = canon_pn(part_number)
            # exact / partial
            datasheet = await find_datasheet_by_text(pn)
            # vendor variants
            if not datasheet:
                for v in vendor_variants(pn):
                    datasheet = await find_datasheet_by_text(v)
                    if datasheet:
                        break
            # prefix relax
            if not datasheet and len(pn) >= 6:
                prefix = pn[:8]
                cursor = collection.find({"part_number": {"$regex": f"^{re.escape(prefix)}", "$options": "i"}}).limit(1)
                arr = await cursor.to_list(length=1)
                if arr:
                    datasheet = _coerce_id_to_str(arr[0])

        enriched_detections.append({
            **detection,
            "datasheet": datasheet,  # plain dict or None
            "datasheet_found": datasheet is not None
        })

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
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
