from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

# Example in-memory database (replace with MongoDB later)
DATABASE = {
    "LPC1768": {
        "manufacturer": "NXP",
        "datasheet_url": "https://www.nxp.com/docs/en/data-sheet/LPC1768_66.pdf"
    },
    "8051": {
        "manufacturer": "Intel",
        "datasheet_url": "https://www.ti.com/lit/ds/symlink/mcs51.pdf"
    }
}

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.get("/datasheet/{part_number}")
async def get_datasheet(part_number: str):
    part = DATABASE.get(part_number.upper())
    if part:
        return part
    return {"error": "Part not found"}
