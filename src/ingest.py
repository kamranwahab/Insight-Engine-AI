import os
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import config  # We import the settings we just made

def process_pdfs():
    print(f"🚀 Starting Ingestion process...")
    print(f"📂 Looking for PDFs in: {config.DATA_DIR}")
    
    all_documents = []
    
    # 1. Loop through all PDF files in the data folder
    files = [f for f in os.listdir(config.DATA_DIR) if f.endswith(".pdf")]
    
    if not files:
        print("❌ No PDF files found! Please put your 50 papers in the 'data' folder.")
        return

    for filename in files:
        pdf_path = os.path.join(config.DATA_DIR, filename)
        print(f"📄 Processing: {filename}...")
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # A. Save Page as Image (For Gemini to "See")
            # We zoom in 2x (matrix=2,2) to make text/charts clear
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
            image_filename = f"{filename}_p{page_num}.png"
            image_path = os.path.join(config.IMAGE_DIR, image_filename)
            pix.save(image_path)
            
            # B. Extract Text (For Retrieval)
            text = page.get_text()
            
            # C. Create the Document Object
            # We attach the 'image_path' so the AI knows which image matches this text
            doc_obj = Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page_num,
                    "image_path": image_path  # <--- The Critical Link
                }
            )
            all_documents.append(doc_obj)
            
    print(f"✅ Processed {len(all_documents)} pages from {len(files)} files.")
    return all_documents

def create_vector_db(documents):
    if not documents:
        print("⚠️ No documents to index.")
        return

    print("🧠 Loading Embedding Model (This might take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    
    print("🗂️ Creating Vector Database...")
    vector_db = FAISS.from_documents(documents, embeddings)
    
    print(f"💾 Saving Index to {config.FAISS_INDEX_PATH}...")
    vector_db.save_local(config.FAISS_INDEX_PATH)
    print("🎉 Success! Database built.")

if __name__ == "__main__":
    # This runs when you type 'python src/ingest.py'
    docs = process_pdfs()
    create_vector_db(docs)