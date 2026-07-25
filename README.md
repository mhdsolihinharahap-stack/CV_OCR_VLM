# 🚗 CV_OCR_VLM

Computer Vision-based Indonesian License Plate Recognition using OCR and Vision Language Model (VLM).

## 📌 Overview

This project performs automatic Indonesian vehicle license plate recognition using Computer Vision and OCR techniques. The system processes vehicle images, extracts license plate regions, recognizes the plate text, and evaluates the recognition performance.

## ✨ Features

- Indonesian license plate recognition
- OCR-based text extraction
- Label normalization
- Character Error Rate (CER) evaluation
- CSV prediction export
- Dataset support for Indonesian license plates

## 📂 Project Structure

```
CV_OCR_VLM/
│
├── dataset/
│   ├── Indonesian License Plate Dataset/
│   └── Indonesian License Plate Recognition Dataset/
│
├── output/
│   └── prediction.csv
│
├── main.py
├── README.md
└── LICENSE
```

## 🛠️ Requirements

- Python 3.14+
- OpenAI SDK
- Jiwer
- Pillow
- Pandas

Install dependencies:

```bash
pip install openai jiwer pandas pillow python-dotenv
```

Or:

```bash
pip install -r requirements.txt
```

## ▶️ Usage

Run the project:

```bash
python main.py
```

## 📊 Evaluation

The project evaluates OCR performance using:

- Character Error Rate (CER)

Predictions are saved in:

```
output/prediction.csv
```

## 📁 Dataset

The project uses Indonesian License Plate datasets for evaluation and testing.

## 👨‍💻 Author

**Muhammad Solihin Harahap**

- Politeknik Negeri Batam
- Computer Vision & AI Research
- GitHub: https://github.com/mhdsolihinharahap-stack

## 📄 License

This project is licensed under the MIT License.