import streamlit as st
import time
import base64
from backend import load_system, handle_complex, cached_similarity_search, get_related_papers, CONCEPT_MAP, smart_truncate, encode_image, safe_google_call, ConversationMemory, PerformanceMetrics
from langchain_core.messages import HumanMessage

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Insight Engine AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Extreme Beauty"
st.markdown("""
<style>
    /* Main Background & Font */
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        font-family: 'Inter', sans-serif;
    }
    
    /* Chat Message Bubbles */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f0f2f6;
        font-weight: 700;
    }
    
    /* Source Paper Tags */
    .paper-tag {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    
    /* Complex Mode Badge */
    .complex-badge {
        background-color: #FF5722;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE MANAGEMENT
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_loaded" not in st.session_state:
    with st.spinner("🚀 Booting up Insight Engine Neural Core..."):
        db, llm = load_system()
        st.session_state.db = db
        st.session_state.llm = llm
        st.session_state.memory = ConversationMemory()
        st.session_state.system_loaded = True

# ==========================================
# 3. SIDEBAR DASHBOARD
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=60)
    st.title("Insight Engine")
    st.caption("v4.0 Ultimate Edition")
    
    st.divider()
    
    # Live Metrics
    st.subheader("📊 System Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Indexed Papers", "50")
    with col2:
        st.metric("Categories", "5")
        
    st.divider()
    
    # Export Chat
    if st.button("💾 Export Conversation"):
        filename = st.session_state.memory.export("chat_history.md")
        with open(filename, "r") as f:
            st.download_button("Download Report", f, file_name="research_report.md")
    
    st.info("💡 Tip: Ask 'Design a...' to trigger Complex Synthesis Mode.")

# ==========================================
# 4. MAIN CHAT INTERFACE
# ==========================================
st.title("🧠 Insight Engine Research Assistant")
st.markdown("Ask deep technical questions about **NLP, Vision, RL, Generative AI, or Optimization**.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Cited Sources"):
                for src in message["sources"]:
                    st.markdown(f"<span class='paper-tag'>{src}</span>", unsafe_allow_html=True)

# User Input
if prompt := st.chat_input("Ask a question (e.g., 'Design a Universal Task Solver')..."):
    # 1. Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Generate Response
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        status_placeholder = st.status("🧠 Thinking...", expanded=True)
        
        try:
            db = st.session_state.db
            llm = st.session_state.llm
            mem = st.session_state.memory
            
            # CHECK COMPLEX MODE
            is_complex = any(w in prompt.lower() for w in ["design", "compare", "synthesize", "universal"])
            
            answer = ""
            papers = []
            
            if is_complex:
                status_placeholder.write("🔥 Complex Mode Detected: Decomposing Query...")
                # We reuse your backend logic here
                answer, papers = handle_complex(prompt, db, llm, mem)
                status_placeholder.write("✅ Synthesis Complete!")
            else:
                status_placeholder.write("🔍 Searching Vector Database...")
                docs = cached_similarity_search(db, prompt, k=3)
                related = get_related_papers([d.metadata["source"] for d in docs], db)
                docs.extend(related)
                
                # Rerank
                status_placeholder.write("⚖️ Reranking Results...")
                docs = sorted(docs, key=lambda x: 5 if x.metadata["source"] in CONCEPT_MAP.values() else 1, reverse=True)[:3]
                
                # Context Build
                ctx = "\n".join([f"[{d.metadata['source']}]\n{smart_truncate(d.page_content)}" for d in docs])
                img = encode_image(docs[0].metadata.get("image_path")) if docs else None
                
                prompt_payload = [{"type": "text", "text": f"Q: {prompt}\nContext:\n{ctx}"}]
                if img:
                    status_placeholder.write("👁️ Analyzing Visual Context...")
                    prompt_payload.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
                
                res = safe_google_call(llm, [HumanMessage(content=prompt_payload)])
                answer = res.content if hasattr(res, 'content') else res
                papers = [d.metadata["source"] for d in docs]
                status_placeholder.write("✅ Answer Generated!")
            
            status_placeholder.update(label="Process Complete", state="complete", expanded=False)
            
            # Display Answer
            response_placeholder.markdown(answer)
            
            # Show Sources nicely
            if papers:
                with st.expander("📚 References & Citations"):
                    st.markdown("**Based on the following papers:**")
                    for p in papers:
                        st.markdown(f"- 📄 `{p}`")
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": papers})
            mem.add(prompt, answer, papers)
            
        except Exception as e:
            status_placeholder.update(label="System Error", state="error")
            st.error(f"An error occurred: {str(e)}")