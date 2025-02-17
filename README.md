# 🌱 Deep Leaf - Plant Disease Classification MLOps Pipeline

## 📌 Overview
**Deep Leaf** is a deep learning-based **image classification pipeline** for detecting plant diseases using **Transfer Learning (VGG16)**. It follows **MLOps best practices**, enabling:
- **Automated dataset handling from Kaggle**
- **Efficient model training & logging**


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
| `models/` _(Folder)_  | Stores **trained models (`.keras`)**. |

---

## 🚀 **Setting Up Deep Leaf for New Developers**
Follow these steps to get started:

### **1️⃣ Fork & Clone the Repositorry**
```sh
git clone https://github.com/schytze0/deep_leaf.git
cd deep_leaf
```

### **2️⃣Fo Create a virtual environment**
Depending on your OS (for example with conda).
```sh
conda create -n my_env python=3.10  
```

### **3️⃣ Install Dependencieses**
```sh
pip install -r requirements.txt
```


### **4️⃣ Set Up Kaggle API Access**
Each team member must store their own Kaggle credentials as GitHub repository secrets.

Step 1: Get Your Kaggle API Key

Go to Kaggle Account Settings.
Click "Create New API Token", which downloads kaggle.json.

Step 2: Add Credentials as GitHub Secrets

Go to GitHub Repo → Settings → Secrets → Actions → New Repository Secret

For each team member, add:

Secret Name	|	Value

KAGGLE_USERNAME_YOURNAME -> "your-kaggle-username"

KAGGLE_KEY_YOURNAME -> "your-kaggle-api-key"


## **🔑 Setting Up the .env File for Automated Environment Setup**

To avoid manually setting environment variables every time, store them in a .env file.

### **1️⃣ Create the .env v File**
Inside the project folder, create a .env file:
```sh
vim .env
```

### **2️⃣ Add the Following Variables to .envnv**
```ini
# User Configuration
GITHUB_ACTOR=your_github_username

# Kaggle API Credentials
KAGGLE_USERNAME_YOURNAME=your_kaggle_username
KAGGLE_KEY_YOURNAME=your_kaggle_api_key
```

## **✅Run the test_config.py file to check the setup**
```sh
python test_config.py
```

There might appear some Tensorflow related warnings (depending on your machine and GPU/CUDA support). The script  should print "Configuration check complete." at the end of the output.

## **🔄 Training the Model**

To train the model, run:
```sh
python train.py
```

✔ Downloads dataset from Kaggle.
✔ Trains model in two phases.
✔ Saves best model to models/.
✔ Logs training history in logs/history_*.json.


## **🔍 Making Predictions**

### **1️⃣ Predict a Single Imagege**
```sh
python predict.py --image path/to/image.jpg
```

### **2️⃣ Predict a Single Imagege**
```sh
python predict.py --folder path/to/folder.jpg
```

## **📊 Visualizing Training Performance**
```sh
python utils.py
```
