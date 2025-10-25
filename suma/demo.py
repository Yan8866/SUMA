# The demo below will open in a browser on http://localhost:7860 if running from a file


from typing import List, Optional

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

import os, json
import boto3
from botocore.exceptions import ClientError

# Cache so we don’t hit Secrets Manager on every request
__OPENAI_KEY_CACHE = None

def get_openai_api_key(
    secret_name=os.getenv("OPENAI_SECRET_NAME", "suma/openai"),
    region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-1",
):
    """Return the OpenAI API key from AWS Secrets Manager (JSON secret with key 'OPENAI_API_KEY')."""
    global __OPENAI_KEY_CACHE
    if __OPENAI_KEY_CACHE:
        return __OPENAI_KEY_CACHE

    client = boto3.client("secretsmanager", region_name=region_name)
    try:
        resp = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        # Optional fallback: allow local ENV during dev
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            __OPENAI_KEY_CACHE = env_key
            return __OPENAI_KEY_CACHE
        raise RuntimeError(f"Unable to read secret {secret_name}: {e}") from e

    # SecretString if stored as text/JSON; SecretBinary if stored as binary
    if "SecretString" in resp:
        s = resp["SecretString"]
    else:
        s = resp["SecretBinary"].decode("utf-8")

    try:
        # Expecting {"OPENAI_API_KEY": "..."} — adjust if you stored a plain string
        key = json.loads(s).get("OPENAI_API_KEY", s)
    except json.JSONDecodeError:
        # If you stored the key as a plain string
        key = s

    if not key:
        raise RuntimeError(f"Secret {secret_name} did not contain OPENAI_API_KEY")

    __OPENAI_KEY_CACHE = key
    return key


# ---- URL scraper ----
try:
    from src.scraper import fetch_website_contents
except Exception:
    # fallback if your module isn't available
    import requests
    from bs4 import BeautifulSoup
    def fetch_website_contents(url: str) -> str:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # basic cleanup: remove nav/aside/script/style
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text

# ---- environment / client ----
OPENAI_API_KEY = get_openai_api_key()   # fetch from Secrets Manager
client = OpenAI(api_key=OPENAI_API_KEY)

# ---- file readers ----
def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    # uses pypdf (pure-python)
    from pypdf import PdfReader
    reader = PdfReader(path)
    chunks = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)

def read_docx(path: str) -> str:
    # .docx only (not legacy .doc)
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

EXT_READERS = {
    ".txt": read_txt,
    ".pdf": read_pdf,
    ".docx": read_docx,
}

def read_any_file(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext in EXT_READERS:
        return EXT_READERS[ext](path)
    raise ValueError(f"Unsupported file type: {ext}")

def collect_files_text(files: Optional[List[gr.File]]) -> str:
    if not files:
        return ""
    texts = []
    for f in files:
        try:
            texts.append(read_any_file(f.name))
        except Exception as e:
            texts.append(f"[Error reading {os.path.basename(f.name)}: {e}]")
    return "\n\n".join(texts)

# ---- prompting helpers ----
SYSTEM_SUMMARY = (
    "You are a concise, slightly snarky assistant. Summarize the provided content, "
    "ignoring boilerplate navigation. Respond in Markdown without code fences."
)
SYSTEM_QA = (
    "You answer questions strictly using the provided content. "
    "If the answer is not present, say 'I don't know based on the provided content.' "
    "Respond concisely in Markdown."
)

def summarize_content(content: str) -> str:
    # keep prompt sizes reasonable
    max_chars = 12000
    content = content[:max_chars]
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_SUMMARY},
            {"role": "user", "content": f"Summarize this:\n\n{content}"},
        ],
    )
    return resp.choices[0].message.content

def answer_question(content: str, question: str) -> str:
    max_chars = 12000
    content = content[:max_chars]
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_QA},
            {
                "role": "user",
                "content": f"Context:\n{content}\n\nQuestion: {question}",
            },
        ],
    )
    return resp.choices[0].message.content

# ---- gradio functions ----
def make_context(url: str, files: List[gr.File]) -> str:
    parts = []
    if url and url.strip():
        try:
            parts.append(fetch_website_contents(url.strip()))
        except Exception as e:
            parts.append(f"[Error fetching URL: {e}]")
    file_text = collect_files_text(files)
    if file_text:
        parts.append(file_text)
    if not parts:
        return "[No content provided. Enter a URL or upload a document.]"
    return "\n\n".join(parts)

def on_summarize(url: str, files: List[gr.File]) -> str:
    content = make_context(url, files)
    try:
        return summarize_content(content)
    except Exception as e:
        return f"**Error:** {type(e).__name__}: {e}"

def on_qa(url: str, files: List[gr.File], question: str) -> str:
    if not question.strip():
        return "Please enter a question."
    content = make_context(url, files)
    try:
        return answer_question(content, question.strip())
    except Exception as e:
        return f"**Error:** {type(e).__name__}: {e}"

# ---- UI (Blocks + CSS for layout) ----
CSS = """

#header {            /* the container for the title/subtitle */
  display: flex;
  flex-direction: column;
  align-items: center;   /* centers horizontally */
  gap: 6px;
}
#app-title h1 {      /* the Markdown -> <h1> */
  text-align: center;
  margin: 0;
}
#subtitle { text-align: center; }


.gradio-container {
  background-color: #d9f5fb
}

html, body { background: transparent !important; }

/* Centered shell */
#shell { max-width: 1100px; margin: 0 auto; padding: 32px 20px 56px; }

/* Card look for the output */
.card {
  background: #ffffffcc;           /* slight translucency over bg image */
  backdrop-filter: blur(2px);
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 6px 20px rgba(0,0,0,.06);
}

/* Make the two buttons blue (by elem_id) */
#btn-sum button, #btn-qa button {
  background-color: #2563eb !important;   /* blue-600 */
  border-color: #2563eb !important;
  color: #ffffff !important;
}
#btn-sum button:hover, #btn-qa button:hover {
  background-color: #1d4ed8 !important;   /* blue-700 */
  border-color: #1d4ed8 !important;
}
"""

theme = gr.themes.Ocean(primary_hue="slate")   # hue doesn't matter since we override with CSS

with gr.Blocks(css=CSS, theme=theme, title="Doc & Web Summarizer") as demo:
    with gr.Column(elem_id="shell"):
        gr.Markdown("# SUMA", elem_id="app-title")
        gr.Markdown(
            "Paste a URL and/or upload files (`.pdf`, `.txt`, `.docx`). "
            "Click **Summarize** or ask a question and click **Answer Questions**."
        )

        with gr.Row():
            with gr.Column(scale=5, min_width=350):
                url_in = gr.Textbox(label="URL", placeholder="https://example.com")
                files_in = gr.File(
                    label="Upload documents",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".docx"],
                )
                question_in = gr.Textbox(
                    label="Question (for Answer Questions)",
                    placeholder="e.g., What are the main findings?"
                )

                with gr.Row():
                    btn_sum = gr.Button("Summarize", variant="primary", elem_id="btn-sum")
                    btn_qa  = gr.Button("Answer Questions", elem_id="btn-qa")

            with gr.Column(scale=7, min_width=420):
                out_md = gr.Markdown(label="Output", elem_classes=["card"])

        btn_sum.click(on_summarize, inputs=[url_in, files_in], outputs=out_md)
        btn_qa.click(on_qa, inputs=[url_in, files_in, question_in], outputs=out_md)

demo.launch(server_name="0.0.0.0", server_port=8080, allowed_paths=["src"])




