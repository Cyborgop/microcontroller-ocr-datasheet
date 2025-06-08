# Microcontroller OCR & Datasheet Lookup

Welcome!  
This project helps you **recognize microcontroller boards from images** and instantly **fetch their datasheets** using deep learning (PyTorch), FastAPI, and MongoDB.  
Perfect for students, engineers, and makers who want to quickly identify and learn about microcontrollers in the lab or in the field.

---

## ✨ Features

- **OCR for microcontroller codes** (CRNN+CTC in PyTorch)
- **FastAPI backend** for instant datasheet queries
- **MongoDB database** for storing and retrieving microcontroller info
- **Modular code**: Easy to extend, adapt, and learn from

---

## 🗂️ Project Structure

microcontroller-ocr-datasheet/
├── ocr/ # OCR model and training code
├── api/ # FastAPI backend
├── db/ # MongoDB init scripts
├── data/ # Images and labels (sample data)
├── README.md
├── .gitignore
└── LICENSE


---

## ⚡ Quick Start

1. **Clone the repo**
    git clone https://github.com/Cyborgop/microcontroller-ocr-datasheet.git
    cd microcontroller-ocr-datasheet.


2. **Install dependencies**
    pip install -r requirements.txt
    pip install -r api/requirements.txt

3. **Prepare your data**
- Place your images and label files in the `data/` folder.

4. **Initialize MongoDB**
- Make sure MongoDB is running.
- Run:
  ```
  mongosh --file db/datasheet_init.js
  ```

5. **Train the OCR model**

  cd ocr
  python train.py

6. **Start the API server**

  cd ../api
  python -m uvicorn main:app --reload

 7. **Test the API**
  - Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive docs.
  
  ---
  
  ## 🧪 Example Usage
  
  **Get datasheet info for a recognized part:**
  GET /datasheet/LPC1768

  _Response:_
{
"manufacturer": "NXP",
"datasheet_url": "https://www.nxp.com/docs/en/data-sheet/LPC1768_66.pdf"
}

**Why did I build this?**  
As a student and maker, I wanted a smarter, faster way to identify microcontroller boards in the lab and get their datasheets without manual searching. This project combines AI, databases, and modern APIs to make that possible for everyone.

  
