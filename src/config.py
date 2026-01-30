import os
from dotenv import load_dotenv

# 1. Load the secrets from .env file
load_dotenv()

# 2. API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ Error: GOOGLE_API_KEY is missing in .env file!")

# 3. Model Settings (Using Gemini 2.5 Flash as requested)
MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 4. Folder Paths
# Get the absolute path of the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
IMAGE_DIR = os.path.join(STORAGE_DIR, "page_images")
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "faiss_index")

# 5. Create directories if they don't exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

print("✅ Configuration Loaded: Gemini 2.5 Flash Ready")