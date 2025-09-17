import os
import gradio as gr
import logging
from config import CONFIG
from generation import ideas, outline, expand, draft
from rag import retrieve

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def query_rag(query: str, hint_title: str = ""):
    """Query the RAG system and return formatted results."""
    if not query.strip():
        return "Please enter a query."
    
    try:
        contexts, category = retrieve(query, hint_title)
        
        if not contexts:
            return f"No relevant information found for query in {category} category."
        
        response = f"**Category:** {category}\n\n"
        response += f"**Found {len(contexts)} relevant sources:**\n\n"
        
        for i, (content, metadata) in enumerate(contexts, 1):
            source = metadata.get('filename', f'Source {i}')
            response += f"**[{i}] {source}:**\n{content[:300]}{'...' if len(content) > 300 else ''}\n\n"
        
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"Error processing query: {str(e)}"

def generate_ideas(beat: str, constraints: str):
    """Generate article ideas."""
    if not beat.strip():
        return "-"
    
    try:
        return ideas(beat, constraints or "No specific constraints")
    except Exception as e:
        logger.error(f"Error generating ideas: {e}")
        return f"Error generating ideas: {str(e)}"

def generate_outline(title: str, context: str):
    """Generate article outline."""
    if not title.strip():
        return "-"
    
    try:
        return outline(title, context or "No additional context provided")
    except Exception as e:
        logger.error(f"Error generating outline: {e}")
        return f"Error generating outline: {str(e)}"

def generate_draft_article(title: str, notes: str):
    """Generate draft with RAG context."""
    if not title.strip():
        return "-"
    
    try:
        contexts, category = retrieve(title, title, top_k=8)
        retrieved_snippets = ""
        for i, (content, metadata) in enumerate(contexts, 1):
            source = metadata.get('filename', f'Source {i}')
            retrieved_snippets += f"[{i}] {source}: {content}\n\n"
        
        article = draft(title, notes or "No additional notes", retrieved_snippets)
        return f"**Category:** {category}\n\n{article}"
    except Exception as e:
        logger.error(f"-: {e}")
        return f"-: {str(e)}"

def run_ingest_process():
    """-."""
    try:
        from ingest import run_ingest
        run_ingest()
        return "Document ingestion completed successfully! The knowledge base has been updated."
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        return f"Error during ingestion: {str(e)}"

def check_system_status():
    """Check system status."""
    try:
        import requests
        from pathlib import Path
        
        status_parts = []
        
        try:
            response = requests.get(f"{CONFIG['OLLAMA_BASE_URL']}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if CONFIG["LLM_MODEL"] in model_names:
                    status_parts.append(f"✅ Ollama: Running with {CONFIG['LLM_MODEL']}")
                else:
                    status_parts.append(f"⚠️ Ollama: Running but {CONFIG['LLM_MODEL']} not found")
            else:
                status_parts.append("❌ Ollama: Not responding")
        except Exception as e:
            status_parts.append(f"❌ Ollama: Connection failed - {str(e)}")
        
        raw_path = Path(CONFIG["DATA_RAW_PATH"])
        chroma_path = Path(CONFIG["CHROMA_PATH"])
        
        if raw_path.exists():
            pdf_count = len(list(raw_path.rglob("*.pdf")))
            status_parts.append(f"✅ Raw data: {pdf_count} PDF files found")
        else:
            status_parts.append("❌ Raw data directory not found")
        
        if chroma_path.exists():
            status_parts.append("✅ ChromaDB: Data directory exists")
        else:
            status_parts.append("⚠️ ChromaDB: No vector database found - run ingestion")
        
        status_parts.append(f"📊 Config:")
        status_parts.append(f"   - Geo embedding: {CONFIG['EMBED_GEO_MODEL']}")
        status_parts.append(f"   - Tech embedding: {CONFIG['EMBED_TECH_MODEL']}")
        status_parts.append(f"   - Reranker: {CONFIG['RERANK_MODEL']}")
        
        return "\n".join(status_parts)
    except Exception as e:
        logger.error(f"Error checking system status: {e}")
        return f"Error checking system status: {str(e)}"

# Build Gradio UI with tabs
with gr.Blocks(title="-", theme=gr.themes.Soft()) as app:
    gr.Markdown("# -")
    gr.Markdown("*-*")
    
    with gr.Row():
        with gr.Column(scale=3):
            status_btn = gr.Button("🔍 Check System Status", variant="secondary")
            ingest_btn = gr.Button("📥 Run Document Ingestion", variant="primary")
        with gr.Column(scale=1):
            pass
    
    status_output = gr.Textbox(label="System Status", lines=8, max_lines=15)
    
    status_btn.click(check_system_status, outputs=status_output)
    ingest_btn.click(run_ingest_process, outputs=status_output)
    
    gr.Markdown("---")
    
    with gr.Tabs():
        with gr.TabItem("-y"):
            gr.Markdown("-")
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(
                        label="-",
                        placeholder="-.",
                        lines=2
                    )
                    hint_input = gr.Textbox(
                        label="-",
                        placeholder="-.",
                        lines=1
                    )
                    query_btn = gr.Button("🔍 Search Documents", variant="primary")
                
                with gr.Column():
                    query_output = gr.Textbox(
                        label="-",
                        lines=15,
                        max_lines=20,
                        placeholder="-..."
                    )
            
            query_btn.click(
                fn=query_rag,
                inputs=[query_input, hint_input],
                outputs=query_output
            )
        
        with gr.TabItem("-n"):
            gr.Markdown("#-")
            with gr.Row():
                with gr.Column():
                    beat_input = gr.Textbox(
                        label="-",
                        placeholder="-..",
                        lines=1
                    )
                    constraints_input = gr.Textbox(
                        label="-s",
                        placeholder="A-..",
                        lines=3
                    )
                    ideas_btn = gr.Button("-s", variant="primary")
                
                with gr.Column():
                    ideas_output = gr.Textbox(
                        label="-",
                        lines=15,
                        max_lines=20,
                        placeholder="-.."
                    )
            
            ideas_btn.click(
                fn=generate_ideas,
                inputs=[beat_input, constraints_input],
                outputs=ideas_output
            )
        
        with gr.TabItem("--"):
            gr.Markdown("#-")
            with gr.Row():
                with gr.Column():
                    outline_title_input = gr.Textbox(
                        label="-",
                        placeholder="-..",
                        lines=1
                    )
                    outline_context_input = gr.Textbox(
                        label="-",
                        lines=5,
                        placeholder="-.."
                    )
                    outline_btn = gr.Button("📝 Generate Outline", variant="primary")
                
                with gr.Column():
                    outline_output = gr.Textbox(
                        label="Article Outline",
                        lines=15,
                        max_lines=20,
                        placeholder="-.."
                    )
            
            outline_btn.click(
                fn=generate_outline,
                inputs=[outline_title_input, outline_context_input],
                outputs=outline_output
            )
        
        with gr.TabItem("-g"):
            gr.Markdown("#-")
            with gr.Row():
                with gr.Column():
                    draft_title_input = gr.Textbox(
                        label="-",
                        placeholder="-..",
                        lines=1
                    )
                    draft_notes_input = gr.Textbox(
                        label="-",
                        lines=8,
                        placeholder="-.."
                    )
                    draft_btn = gr.Button("📄 Generate Draft", variant="primary")
                
                with gr.Column():
                    draft_output = gr.Textbox(
                        label="-",
                        lines=20,
                        max_lines=30,
                        placeholder="-."
                    )
            
            draft_btn.click(
                fn=generate_draft_article,
                inputs=[draft_title_input, draft_notes_input],
                outputs=draft_output
            )

    gr.Markdown("---")
    gr.Markdown("*Built with Ollama, dual-category embeddings, and LlamaIndex RAG*")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)
