# Microcontroller OCR & Datasheet Lookup System

An end-to-end system for **detecting microcontroller boards, performing OCR on chip markings, and retrieving relevant datasheet information**.  
Designed with a focus on **Computer Vision, OCR pipelines, and backend service integration**, with future scope for **embedded and edge deployment**.

---

## 📌 Overview
This project combines **computer vision, deep learning, and backend services** to automate the process of identifying microcontroller components and mapping them to their corresponding datasheets.

The system is useful in scenarios such as:
- Component identification from images
- Automated datasheet lookup
- Inventory analysis and verification
- Vision-assisted embedded workflows

---

## 🧠 System Architecture
The pipeline follows a modular design:

1. **Image Input**
2. **Microcontroller Detection**
3. **OCR on Chip Markings**
4. **Text Post-processing & Cleaning**
5. **Datasheet Lookup via Backend API**
6. **Structured Output (JSON / DB)**

Architecture diagrams are available in the `architecture_diagrams/` directory.

---

## 🔍 Core Features
- Microcontroller detection using deep learning
- OCR-based text extraction from chip surfaces
- Robust preprocessing and dataset cleaning
- Backend API for datasheet retrieval
- Modular design for experimentation and scaling

---

## 🛠 Tech Stack

### 🔹 Computer Vision & AI
- PyTorch
- OpenCV
- Custom OCR pipelines

### 🔹 Backend Services
- FastAPI
- RESTful APIs
- MongoDB for metadata and lookup

### 🔹 Programming Languages
- Python
- C / C++ (for embedded-oriented experimentation)

### 🔹 Tools & Platforms
- Git
- Linux
- Dataset annotation tools (VGG Image Annotator)

---

## 📂 Repository Structure
```text
.
├── api/                     # FastAPI backend services
├── ocr/                     # OCR models and pipelines
├── data/                    # Raw and processed datasets
├── datasets_used/           # Curated / golden datasets
├── Helper Python Files/     # Utility and helper scripts
├── architecture_diagrams/   # System design and architecture
├── runs/                    # Training and inference outputs
├── db/                      # Database-related code
└── README.md
