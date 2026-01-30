import os
import base64
import time
import json
import re
import config
from typing import List, Optional, Any, Dict, Union
from collections import defaultdict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# ==========================================
# 0. CONFIGURATION & KNOWLEDGE GRAPH
# ==========================================

PAPERS_BY_CATEGORY = {
    "NLP": [
        "01_Attention_Is_All_You_Need.pdf", "02_BERT_Pretraining.pdf", "03_GPT3_Language_Models.pdf",
        "04_T5_Text_to_Text_Transfer.pdf", "05_LLaMA_Open_Foundation_Models.pdf", "06_LoRA_Low_Rank_Adaptation.pdf",
        "07_Chain_of_Thought_Prompting.pdf", "08_ALBERT_Lite_BERT.pdf", "09_Reformer_Efficient_Transformer.pdf",
        "10_Longformer_Long_Document.pdf"
    ],
    "Vision": [
        "11_ResNet_Deep_Residual.pdf", "12_VGG_Deep_Networks.pdf", "13_YOLO_Object_Detection.pdf",
        "14_Mask_RCNN.pdf", "15_Vision_Transformers_ViT.pdf", "16_EfficientNet.pdf",
        "17_DenseNet.pdf", "18_UNet_Biomedical_Seg.pdf", "19_CLIP_OpenAI.pdf", "20_FPN_Feature_Pyramid.pdf"
    ],
    "Generative": [
        "21_GANs_Goodfellow.pdf", "22_DDPM_Diffusion_Models.pdf", "23_Stable_Diffusion_High_Res.pdf",
        "24_Pix2Pix_Image_Translation.pdf", "25_CycleGAN.pdf", "26_StyleGAN.pdf",
        "27_VAE_Variational_Autoencoders.pdf", "28_DALLE2_Hierarchical.pdf", "29_InfoGAN.pdf",
        "30_Generative_Modeling_Estimators.pdf"
    ],
    "RL": [
        "31_DQN_Atari.pdf", "32_PPO_Proximal_Policy.pdf", "33_TRPO_Trust_Region.pdf",
        "34_DDPG_Continuous_Control.pdf", "35_A3C_Asynchronous_RL.pdf", "36_Rainbow_DQN.pdf",
        "37_SAC_Soft_Actor_Critic.pdf", "38_MARL_Multi_Agent.pdf", "39_Dreamer_World_Models.pdf",
        "40_OpenAI_Gym.pdf"
    ],
    "Optimization": [
        "41_Adam_Optimizer.pdf", "42_Word2Vec_Mikolov.pdf", "43_Dropout_Srivastava.pdf",
        "44_Layer_Normalization.pdf", "45_Batch_Normalization.pdf", "46_Capsule_Networks.pdf",
        "47_ELMo_Embeddings.pdf", "48_GCN_Graph_Convolution.pdf", "49_Self_Normalizing_NN.pdf",
        "50_Fast_Gradient_Sign_Attack.pdf"
    ]
}

# FULLY EXPANDED Relationship Graph (Covering Logic for All 50 Papers)
PAPER_RELATIONSHIPS = {
    # NLP
    "01_Attention_Is_All_You_Need.pdf": ["02_BERT_Pretraining.pdf", "03_GPT3_Language_Models.pdf", "09_Reformer_Efficient_Transformer.pdf"],
    "02_BERT_Pretraining.pdf": ["08_ALBERT_Lite_BERT.pdf", "47_ELMo_Embeddings.pdf", "06_LoRA_Low_Rank_Adaptation.pdf"],
    "03_GPT3_Language_Models.pdf": ["07_Chain_of_Thought_Prompting.pdf", "05_LLaMA_Open_Foundation_Models.pdf", "04_T5_Text_to_Text_Transfer.pdf"],
    "04_T5_Text_to_Text_Transfer.pdf": ["02_BERT_Pretraining.pdf", "01_Attention_Is_All_You_Need.pdf"],
    "05_LLaMA_Open_Foundation_Models.pdf": ["03_GPT3_Language_Models.pdf", "06_LoRA_Low_Rank_Adaptation.pdf"],
    "06_LoRA_Low_Rank_Adaptation.pdf": ["02_BERT_Pretraining.pdf", "03_GPT3_Language_Models.pdf"],
    "07_Chain_of_Thought_Prompting.pdf": ["03_GPT3_Language_Models.pdf", "05_LLaMA_Open_Foundation_Models.pdf"],
    "08_ALBERT_Lite_BERT.pdf": ["02_BERT_Pretraining.pdf"],
    "09_Reformer_Efficient_Transformer.pdf": ["01_Attention_Is_All_You_Need.pdf", "10_Longformer_Long_Document.pdf"],
    "10_Longformer_Long_Document.pdf": ["09_Reformer_Efficient_Transformer.pdf", "01_Attention_Is_All_You_Need.pdf"],

    # Vision
    "11_ResNet_Deep_Residual.pdf": ["12_VGG_Deep_Networks.pdf", "17_DenseNet.pdf", "16_EfficientNet.pdf"],
    "12_VGG_Deep_Networks.pdf": ["11_ResNet_Deep_Residual.pdf"],
    "13_YOLO_Object_Detection.pdf": ["14_Mask_RCNN.pdf", "20_FPN_Feature_Pyramid.pdf"],
    "14_Mask_RCNN.pdf": ["13_YOLO_Object_Detection.pdf", "20_FPN_Feature_Pyramid.pdf"],
    "15_Vision_Transformers_ViT.pdf": ["01_Attention_Is_All_You_Need.pdf", "11_ResNet_Deep_Residual.pdf", "19_CLIP_OpenAI.pdf"],
    "16_EfficientNet.pdf": ["11_ResNet_Deep_Residual.pdf", "17_DenseNet.pdf"],
    "17_DenseNet.pdf": ["11_ResNet_Deep_Residual.pdf"],
    "18_UNet_Biomedical_Seg.pdf": ["14_Mask_RCNN.pdf"],
    "19_CLIP_OpenAI.pdf": ["15_Vision_Transformers_ViT.pdf", "28_DALLE2_Hierarchical.pdf"],
    "20_FPN_Feature_Pyramid.pdf": ["13_YOLO_Object_Detection.pdf", "14_Mask_RCNN.pdf"],

    # Generative
    "21_GANs_Goodfellow.pdf": ["26_StyleGAN.pdf", "25_CycleGAN.pdf", "29_InfoGAN.pdf", "24_Pix2Pix_Image_Translation.pdf"],
    "22_DDPM_Diffusion_Models.pdf": ["23_Stable_Diffusion_High_Res.pdf", "28_DALLE2_Hierarchical.pdf", "27_VAE_Variational_Autoencoders.pdf"],
    "23_Stable_Diffusion_High_Res.pdf": ["22_DDPM_Diffusion_Models.pdf", "28_DALLE2_Hierarchical.pdf"],
    "24_Pix2Pix_Image_Translation.pdf": ["21_GANs_Goodfellow.pdf", "25_CycleGAN.pdf"],
    "25_CycleGAN.pdf": ["21_GANs_Goodfellow.pdf", "24_Pix2Pix_Image_Translation.pdf"],
    "26_StyleGAN.pdf": ["21_GANs_Goodfellow.pdf"],
    "27_VAE_Variational_Autoencoders.pdf": ["21_GANs_Goodfellow.pdf", "22_DDPM_Diffusion_Models.pdf"],
    "28_DALLE2_Hierarchical.pdf": ["19_CLIP_OpenAI.pdf", "22_DDPM_Diffusion_Models.pdf"],
    "29_InfoGAN.pdf": ["21_GANs_Goodfellow.pdf"],
    "30_Generative_Modeling_Estimators.pdf": ["21_GANs_Goodfellow.pdf"],

    # RL
    "31_DQN_Atari.pdf": ["36_Rainbow_DQN.pdf", "35_A3C_Asynchronous_RL.pdf", "34_DDPG_Continuous_Control.pdf"],
    "32_PPO_Proximal_Policy.pdf": ["33_TRPO_Trust_Region.pdf", "35_A3C_Asynchronous_RL.pdf", "37_SAC_Soft_Actor_Critic.pdf"],
    "33_TRPO_Trust_Region.pdf": ["32_PPO_Proximal_Policy.pdf"],
    "34_DDPG_Continuous_Control.pdf": ["31_DQN_Atari.pdf", "37_SAC_Soft_Actor_Critic.pdf"],
    "35_A3C_Asynchronous_RL.pdf": ["31_DQN_Atari.pdf", "32_PPO_Proximal_Policy.pdf"],
    "36_Rainbow_DQN.pdf": ["31_DQN_Atari.pdf"],
    "37_SAC_Soft_Actor_Critic.pdf": ["34_DDPG_Continuous_Control.pdf", "32_PPO_Proximal_Policy.pdf"],
    "38_MARL_Multi_Agent.pdf": ["32_PPO_Proximal_Policy.pdf"],
    "39_Dreamer_World_Models.pdf": ["40_OpenAI_Gym.pdf"],
    "40_OpenAI_Gym.pdf": ["31_DQN_Atari.pdf", "32_PPO_Proximal_Policy.pdf"],

    # Optimization
    "41_Adam_Optimizer.pdf": ["45_Batch_Normalization.pdf", "43_Dropout_Srivastava.pdf"],
    "42_Word2Vec_Mikolov.pdf": ["47_ELMo_Embeddings.pdf", "48_GCN_Graph_Convolution.pdf"],
    "43_Dropout_Srivastava.pdf": ["45_Batch_Normalization.pdf", "44_Layer_Normalization.pdf"],
    "44_Layer_Normalization.pdf": ["45_Batch_Normalization.pdf", "01_Attention_Is_All_You_Need.pdf"],
    "45_Batch_Normalization.pdf": ["44_Layer_Normalization.pdf", "49_Self_Normalizing_NN.pdf"],
    "46_Capsule_Networks.pdf": ["11_ResNet_Deep_Residual.pdf"],
    "47_ELMo_Embeddings.pdf": ["02_BERT_Pretraining.pdf", "42_Word2Vec_Mikolov.pdf"],
    "48_GCN_Graph_Convolution.pdf": ["42_Word2Vec_Mikolov.pdf"],
    "49_Self_Normalizing_NN.pdf": ["45_Batch_Normalization.pdf"],
    "50_Fast_Gradient_Sign_Attack.pdf": ["21_GANs_Goodfellow.pdf"]
}

def build_comprehensive_concept_map():
    m = {}
    # Extensive regex patterns for ALL 50 papers
    patterns = [
        # NLP
        (".*attention.*", ["transformer", "attention", "self-attention"]),
        (".*bert.*", ["bert", "masked", "bidirectional"]),
        (".*gpt.*", ["gpt", "few-shot", "in-context"]),
        (".*t5.*", ["t5", "text-to-text"]),
        (".*llama.*", ["llama", "foundation"]),
        (".*lora.*", ["lora", "fine-tuning"]),
        (".*chain.*", ["chain of thought", "reasoning", "cot"]),
        (".*albert.*", ["albert", "parameter reduction"]),
        (".*reformer.*", ["reformer", "efficient attention"]),
        (".*longformer.*", ["longformer", "long context"]),
        # Vision
        (".*resnet.*", ["resnet", "residual", "skip connection"]),
        (".*vgg.*", ["vgg", "convolutional"]),
        (".*yolo.*", ["yolo", "object detection"]),
        (".*mask.*rcnn.*", ["mask r-cnn", "instance segmentation"]),
        (".*vit.*", ["vit", "vision transformer", "patches"]),
        (".*efficientnet.*", ["efficientnet", "scaling"]),
        (".*densenet.*", ["densenet", "dense connections"]),
        (".*unet.*", ["unet", "segmentation", "biomedical"]),
        (".*clip.*", ["clip", "contrastive", "multimodal"]),
        (".*fpn.*", ["fpn", "feature pyramid"]),
        # Generative
        (".*gan.*", ["gan", "adversarial", "generator"]),
        (".*ddpm.*", ["ddpm", "diffusion"]),
        (".*stable.*diffusion.*", ["stable diffusion", "latent diffusion"]),
        (".*pix2pix.*", ["pix2pix", "image translation"]),
        (".*cyclegan.*", ["cyclegan", "unpaired"]),
        (".*stylegan.*", ["stylegan", "style mixing"]),
        (".*vae.*", ["vae", "variational autoencoder"]),
        (".*dalle.*", ["dalle", "text-to-image"]),
        (".*infogan.*", ["infogan", "disentangled"]),
        (".*generative.*estimator.*", ["generative modeling"]),
        # RL
        (".*dqn.*", ["dqn", "q-learning", "atari"]),
        (".*ppo.*", ["ppo", "proximal", "policy gradient"]),
        (".*trpo.*", ["trpo", "trust region"]),
        (".*ddpg.*", ["ddpg", "continuous control"]),
        (".*a3c.*", ["a3c", "asynchronous"]),
        (".*rainbow.*", ["rainbow", "distributional rl"]),
        (".*sac.*", ["sac", "soft actor-critic"]),
        (".*marl.*", ["marl", "multi-agent"]),
        (".*dreamer.*", ["dreamer", "world models"]),
        (".*gym.*", ["openai gym", "benchmark"]),
        # Optimization
        (".*adam.*", ["adam", "momentum", "optimizer"]),
        (".*word2vec.*", ["word2vec", "embedding"]),
        (".*dropout.*", ["dropout", "regularization"]),
        (".*layer.*norm.*", ["layer normalization"]),
        (".*batch.*norm.*", ["batch normalization"]),
        (".*capsule.*", ["capsule network", "dynamic routing"]),
        (".*elmo.*", ["elmo", "contextual"]),
        (".*gcn.*", ["gcn", "graph convolutional"]),
        (".*self.*norm.*", ["selu", "self-normalizing"]),
        (".*fast.*gradient.*", ["fgsm", "adversarial attack"])
    ]
    
    # Flatten paper list and build map
    all_papers = []
    for papers in PAPERS_BY_CATEGORY.values():
        all_papers.extend(papers)
        
    for filename in all_papers:
        for pat, tags in patterns:
            if re.search(pat, filename.lower()):
                for tag in tags: m[tag] = filename
    return m

CONCEPT_MAP = build_comprehensive_concept_map()
SEARCH_CACHE = {}

# ==========================================
# 1. CORE CLASSES
# ==========================================

class SimpleResponse:
    def __init__(self, content):
        self.content = content

class PerformanceMetrics:
    def __init__(self):
        self.metrics = {"queries": 0, "complex": 0, "api_calls": 0}
    
    def log(self, is_complex=False):
        self.metrics["queries"] += 1
        if is_complex: self.metrics["complex"] += 1
    
    def api_hit(self):
        self.metrics["api_calls"] += 1

    def summary(self):
        return f"📊 Stats: {self.metrics['queries']} Queries | {self.metrics['complex']} Complex | {self.metrics['api_calls']} API Calls"

class ConversationMemory:
    def __init__(self):
        self.history = []
    
    def add(self, q, a, papers):
        self.history.append({"role": "user", "content": q})
        self.history.append({"role": "assistant", "content": a, "papers": papers})
    
    def get_context(self):
        return "\n".join([f"{h['role'].upper()}: {h['content'][:300]}..." for h in self.history[-4:]])
    
    def export(self, filename="report.md"):
        with open(filename, "w") as f:
            f.write("# Research Session Report\n\n")
            for h in self.history:
                if h["role"] == "user": f.write(f"## Q: {h['content']}\n\n")
                else: f.write(f"**A:** {h['content']}\n\n*Sources: {', '.join(h['papers'])}*\n\n---\n")
        return filename

# ==========================================
# 2. UTILS & LOADING
# ==========================================

METRICS = PerformanceMetrics()

def cached_similarity_search(db, query, k=3):
    cache_key = f"{query}_{k}"
    if cache_key in SEARCH_CACHE: return SEARCH_CACHE[cache_key]
    results = db.similarity_search(query, k=k)
    SEARCH_CACHE[cache_key] = results
    return results

def smart_truncate(text, limit=1500):
    if len(text) <= limit: return text
    cut = text[:limit]
    last_dot = cut.rfind('. ')
    return cut[:last_dot+1] + "..." if last_dot > 0 else cut + "..."

def encode_image(path):
    if not path or not os.path.exists(path): return None
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode('utf-8')

def safe_google_call(llm, messages, retries=1):
    METRICS.api_hit()
    for i in range(retries + 1):
        try: 
            return llm.invoke(messages)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if i < retries:
                    wait = 60
                    print(f"   (⚠️ Quota limit. Cooling down {wait}s... Attempt {i+1}/{retries})")
                    time.sleep(wait)
                else:
                    return SimpleResponse("I apologize, but the Google API is currently overloaded. Please try again in 2 minutes.")
            else: 
                return SimpleResponse(f"Error: {str(e)}")
    return SimpleResponse("API Error: Unknown failure.")

def get_llm_with_fallback():
    try:
        return ChatGoogleGenerativeAI(model=config.MODEL_NAME, temperature=0.1, google_api_key=config.GOOGLE_API_KEY)
    except Exception as e:
        print(f"   (⚠️ Gemini Error: {e})")
        print("   (⚠️ Falling back to local Ollama...)")
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model="llama3.2", temperature=0.1)
        except:
            return None

def load_system():
    print("⏳ Loading System...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    if not os.path.exists(config.FAISS_INDEX_PATH): return None, None
    db = FAISS.load_local(config.FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = get_llm_with_fallback()
    if llm: print("✅ System Ready.")
    return db, llm

# ==========================================
# 3. BONUS FEATURES
# ==========================================

def generate_mermaid_diagram(description, llm):
    print("   (🎨 Generating visual architecture...)")
    prompt = f"""
    Based on this description, generate a valid Mermaid.js 'graph TD' code block.
    Return ONLY the code inside mermaid tags.
    DESCRIPTION: {description[:2000]}
    """
    res = safe_google_call(llm, [HumanMessage(content=prompt)])
    if "Error" in res.content: return ""
    code = res.content.replace("```mermaid", "").replace("```", "").strip()
    return f"\n```mermaid\n{code}\n```\n"

def stream_google_call(llm, messages):
    """
    Streams the response from Google Gemini chunk by chunk.
    This allows the UI to show the text typing out in real-time.
    """
    try:
        # We use .stream() instead of .invoke()
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        yield f"Error during streaming: {str(e)}"


def compare_alternatives(design_text, llm):
    print("   (⚖️ Running comparative analysis...)")
    prompt = f"""
    Compare the proposed design below with: Monolithic (GPT-4) and Modular (MoE).
    Output a Markdown Table comparing Adaptability, Cost, and Modality.
    DESIGN: {design_text[:2000]}
    """
    res = safe_google_call(llm, [HumanMessage(content=prompt)])
    if "Error" in res.content: return ""
    return f"\n### ⚖️ Comparative Analysis\n{res.content}\n"

def ask_followup(db, llm):
    # Only for CLI mode, not used in Streamlit directly
    pass

# ==========================================
# 4. PIPELINE
# ==========================================

def get_related_papers(source_papers, db):
    extras = []
    for p in source_papers:
        if p in PAPER_RELATIONSHIPS:
            for related in PAPER_RELATIONSHIPS[p]:
                extras.extend(cached_similarity_search(db, related, k=1))
    return extras

def decompose_query(query, llm):
    print("   (🤔 Decomposing...)")
    prompt = (
        "You are an AI Librarian with 50 specific AI papers.\n"
        "Break the user's question into 3-4 specific search queries that target these papers.\n"
        "Output ONLY a JSON list of strings."
    )
    res = safe_google_call(llm, [SystemMessage(content=prompt), HumanMessage(content=query)])
    
    if "apologize" in res.content or "Error" in res.content:
        return [query]
        
    try:
        return json.loads(res.content.replace("```json", "").replace("```", "").strip())
    except: return [query]

def handle_complex(query, db, llm, mem):
    print("   (🧠 Complex Mode Activated)")
    METRICS.log(is_complex=True)
    
    sub_qs = decompose_query(query, llm)
    
    # Error Check: If API failed and return message is the query or error text
    if isinstance(sub_qs, list) and len(sub_qs) == 1 and ("apologize" in sub_qs[0] or "Error" in sub_qs[0]):
         return sub_qs[0], []

    docs_map = {}
    
    for q in sub_qs:
        raw_docs = cached_similarity_search(db, q, k=3)
        for d in raw_docs:
            score = 0
            if d.metadata["source"] in PAPER_RELATIONSHIPS: score += 2
            if any(k in q.lower() for k in CONCEPT_MAP if CONCEPT_MAP[k] == d.metadata["source"]): score += 5
            docs_map[d.metadata["source"]] = (score, d)

    if "universal" in query.lower() or "design" in query.lower():
        found_cats = {c for c, ps in PAPERS_BY_CATEGORY.items() if any(p in docs_map for p in ps)}
        missing = set(PAPERS_BY_CATEGORY.keys()) - found_cats
        if missing:
            print(f"   (⚠️ Fetching missing categories: {missing})")
            for cat in missing:
                rep = PAPERS_BY_CATEGORY[cat][0]
                d = cached_similarity_search(db, rep, k=1)[0]
                docs_map[rep] = (10, d)

    final_docs = [d for _, d in sorted(docs_map.values(), key=lambda x: x[0], reverse=True)][:6]
    context = ""
    img = None
    
    if not final_docs: return "Could not find relevant papers.", []

    for i, d in enumerate(final_docs):
        context += f"\n[PAPER: {d.metadata['source']}]\n{smart_truncate(d.page_content)}\n"
        if i==0: img = encode_image(d.metadata.get("image_path"))

    prompt = f"""
    ROLE: Chief AI Architect.
    TASK: Synthesize a design/answer based on the papers below.
    USER: {query}
    PREVIOUS CONTEXT: {mem.get_context()}
    PAPERS: {context}
    REQUIREMENTS:
    1. If designing, describe architecture in text.
    2. Cite papers for every decision.
    3. Explain component interactions.
    """
    
    payload = [{"type": "text", "text": prompt}]
    if img: payload.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
    
    res = safe_google_call(llm, [HumanMessage(content=payload)])
    
    if "apologize" in res.content or "Error" in res.content:
         return res.content, []
    
    main_answer = res.content
    diagram = generate_mermaid_diagram(main_answer, llm)
    comparison = compare_alternatives(main_answer, llm)
    
    return f"{main_answer}\n\n### 📐 Visual Architecture\n{diagram}\n{comparison}", [d.metadata["source"] for d in final_docs]

# ==========================================
# 5. LIBRARY MODE
# ==========================================
pass