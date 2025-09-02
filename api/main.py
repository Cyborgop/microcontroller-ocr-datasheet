from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Any
import motor.motor_asyncio
from bson import ObjectId
from pydantic_core import core_schema
from pydantic.json_schema import JsonSchemaValue
from pydantic._internal._generate_schema import GetJsonSchemaHandler
from contextlib import asynccontextmanager

# MongoDB connection variables (defined outside of lifespan for global access)
MONGO_URL = "mongodb://localhost:27017"
client = None
database = None
collection = None

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

# Lifespan event handler using the new async context manager approach
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize MongoDB connection
    global client, database, collection
    
    try:
        # Create MongoDB connection
        client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
        database = client.microcontrollers
        collection = database.datasheets
        
        # Test the connection
        await client.admin.command('ping')
        print("✅ Successfully connected to MongoDB!")
        
        # Check if we have data in our collection
        count = await collection.count_documents({})
        print(f"📊 Found {count} datasheets in database")
        
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
    
    # Yield control to the application
    yield
    
    # Shutdown: Close MongoDB connection
    if client:
        client.close()
        print("🔌 Disconnected from MongoDB")

# FastAPI app instance with lifespan event handler
app = FastAPI(
    title="Microcontroller Datasheet API",
    description="API for retrieving microcontroller datasheets from MongoDB",
    version="1.0.0",
    lifespan=lifespan  # Use the new lifespan parameter
)

# API Routes
@app.get("/")
async def root():
    return {"message": "Microcontroller Datasheet API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "connected"}

@app.get("/datasheet/{part_number}", response_model=DatasheetModel)
async def get_datasheet(part_number: str):
    """Get datasheet by part number"""
    # Search for the part number (case-insensitive)
    datasheet = await collection.find_one({
        "part_number": {"$regex": f"^{part_number}$", "$options": "i"}
    })
    
    if datasheet:
        return DatasheetModel(**datasheet)
    
    raise HTTPException(status_code=404, detail=f"Datasheet for '{part_number}' not found")

@app.get("/datasheets/", response_model=List[DatasheetModel])
async def get_all_datasheets(skip: int = 0, limit: int = 10):
    """Get all datasheets with pagination"""
    cursor = collection.find().skip(skip).limit(limit)
    datasheets = await cursor.to_list(length=limit)
    return [DatasheetModel(**datasheet) for datasheet in datasheets]

@app.get("/datasheets/manufacturer/{manufacturer}", response_model=List[DatasheetModel])
async def get_datasheets_by_manufacturer(manufacturer: str):
    """Get all datasheets by manufacturer"""
    cursor = collection.find({
        "manufacturer": {"$regex": manufacturer, "$options": "i"}
    })
    datasheets = await cursor.to_list(length=None)
    return [DatasheetModel(**datasheet) for datasheet in datasheets]

@app.get("/search/", response_model=List[DatasheetModel])
async def search_datasheets(q: str):
    """Search datasheets by part number, manufacturer, or features"""
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)  

