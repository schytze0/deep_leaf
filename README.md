# 🌱 Deep Leaf - Plant Disease Classification MLOps Pipeline

## 📌 Overview
**Deep Leaf** is a deep learning-based **image classification pipeline** for detecting plant diseases using **Transfer Learning (VGG16)**. It follows **MLOps best practices**, enabling:
- **Automated dataset handling from Kaggle**
- **Efficient model training & logging**

---

## 📂 Repository Structure
| File/Folder            | Description |
|------------------------|-------------|
| `config.py`           | Stores **global configuration** (paths, credentials, model settings). |
| `data_loader.py`      | Handles **dataset downloading & preprocessing**. |
| `model.py`            | Defines the **VGG16 transfer learning model**. |
| `train.py`            | **Trains the model** in two phases and saves training history. |
| `predict.py`          | **Makes predictions** on single images or folders. |
| `utils.py`            | Loads & **plots training history** (accuracy & loss). |
| `requirements.txt`    | Lists **dependencies** for setting up the environment. |
| `logs/` _(Folder)_    | Stores **training history (`history_*.json`)**. |
| `models/` _(Folder)_  | Stores **trained models (`.h5`)**. |

---

## 🚀 **Setting Up Deep Leaf for New Developers**
Follow these steps to get started:

### **1️⃣ Fork & Clone the Repositorry**
```sh
git clone https://github.com/your-org/deep-leaf.git
cd deep-leaf


### **2️⃣Fo Create a virtual environment**
Depending on your OS.


### **3️⃣ Install Dependencieses**
```sh
pip install -r requirements.txt



### **4️⃣ Set Up Kaggle API Accecess**
Each team member must store their own Kaggle credentials as GitHub repository secrets.

Step 1: Get Your Kaggle API Key

Go to Kaggle Account Settings.
Click "Create New API Token", which downloads kaggle.json.
Step 2: Add Credentials as GitHub Secrets

Go to GitHub Repo → Settings → Secrets → Actions → New Repository Secret
For each team member, add:

Secret Name	Value
KAGGLE_USERNAME_YOURNAME -> "your-kaggle-username"
KAGGLE_KEY_YOURNAME -> "your-kaggle-api-key"


## **🔄 Training the Model**

To train the model, run:
```sh
python train.py

✔ Downloads dataset from Kaggle.
✔ Trains model in two phases.
✔ Saves best model to models/.
✔ Logs training history in logs/history_*.json.


## **🔍 Making Predictions**

### **1️⃣ Predict a Single Imagege**
```sh
python predict.py --image path/to/image.jpg

### **2️⃣ Predict a Single Imagege**
```sh
python predict.py --folder path/to/folder.jpg

## **📊 Visualizing Training Performance**
```sh
python utils.py
