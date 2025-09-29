import os, re, pathlib, json
import numpy as np
import pandas as pd

import torch
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM
import gradio as gr

PROJECT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "sample_telugu.csv"

SAMPLE_ROWS = [
    {"id":"te1","language":"te","context":"తెలంగాణ రాష్ట్ర రాజధాని హైదరాబాదు. ఈ నగరం ఐటి పరిశ్రమకు ప్రసిద్ధి.","question":"తెలంగాణ రాష్ట్ర రాజధాని ఏది?","answer_text":"హైదరాబాదు"},
    {"id":"te2","language":"te","context":"తెలుగు భాష ద్రావిడ భాషా కుటుంబానికి చెందినది. దాని లిపి తెలుగు లిపి.","question":"తెలుగు భాష ఏ లిపిని ఉపయోగిస్తుంది?","answer_text":"తెలుగు లిపి"},
    {"id":"te3","language":"te","context":"సీతాకోక చిలుకలకు రెండు రెక్కలు ఉంటాయి. ఇవి పూల మకరందం తాగుతాయి.","question":"సీతాకోక చిలుకకు ఎన్ని రెక్కలు ఉన్నాయి?","answer_text":"రెండు"},
    {"id":"te4","language":"te","context":"విశాఖపట్నం ఒక తీర నగరం. ఇది ఆంధ్రప్రదేశ్‌లోని ప్రముఖ నౌకాశ్రయం.","question":"విశాఖపట్నం ఏ రకమైన నగరం?","answer_text":"తీర నగరం"},
    {"id":"te5","language":"te","context":"చార్మినార్ హైదరాబాద్ లో ఉంది. ఇది చారిత్రక స్మారక చిహ్నం.","question":"చార్మినార్ ఎక్కడ ఉంది?","answer_text":"హైదరాబాద్"},
]

def ensure_sample_csv(path: pathlib.Path):
    if not path.exists():
        df = pd.DataFrame(SAMPLE_ROWS)
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"[init] Wrote sample Telugu data to {path}")

ensure_sample_csv(CSV_PATH)

_ZW = r"\u200b\u200c\u200d\ufeff"
ZW_RE = re.compile(f"[{_ZW}]")

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u0964", "।")
    s = ZW_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df = pd.read_csv(CSV_PATH, encoding="utf-8")
df["context_norm"] = df["context"].apply(normalize_text)
CORPUS = df["context_norm"].tolist()

EMB_MODEL_NAME = "intfloat/multilingual-e5-base"
emb_model = SentenceTransformer(EMB_MODEL_NAME)
emb_model.eval()

def encode_queries(texts):
    texts = [normalize_text(t) for t in texts]
    prefixed = [f"query: {t}" for t in texts]
    with torch.inference_mode():
        vecs = emb_model.encode(prefixed, normalize_embeddings=True)
    return vecs

def encode_passages(texts):
    texts = [normalize_text(t) for t in texts]
    prefixed = [f"passage: {t}" for t in texts]
    with torch.inference_mode():
        vecs = emb_model.encode(prefixed, normalize_embeddings=True)
    return vecs

PASSAGE_EMBS = encode_passages(CORPUS)

def retrieve_top_k(query: str, k: int = 3):
    if not query or not query.strip():
        return []
    qv = encode_queries([query])[0]
    sims = np.dot(PASSAGE_EMBS, qv)
    idxs = np.argsort(-sims)[:k]
    results = []
    for rank, i in enumerate(idxs):
        results.append({"rank": int(rank+1), "similarity": float(sims[i]), "context": CORPUS[i]})
    return results

READER_MODEL = "deepset/xlm-roberta-large-squad2"
device = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(READER_MODEL, use_fast=True)
qa = pipeline("question-answering", model=READER_MODEL, tokenizer=tokenizer, device=device)

# --- Telugu -> English translator (offline, NLLB-200) ---
# Model: facebook/nllb-200-distilled-600M
# Language codes: Telugu = 'tel_Telu', English = 'eng_Latn'
NLLB_ID = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_ID)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_ID)
trans_te_en = pipeline(
    "translation",
    model=nllb_model,
    tokenizer=nllb_tokenizer,
    src_lang="tel_Telu",
    tgt_lang="eng_Latn",
    device=device
)

def te_to_en(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    out = trans_te_en(text, max_length=256)
    return out[0]["translation_text"].strip()


def answer_with_context(question: str, context: str):
    question = normalize_text(question)
    context = normalize_text(context)
    if not question or not context:
        return {"answer": "", "score": 0.0}
    out = qa(question=question, context=context)
    ans = out.get("answer", "").strip()
    score = float(out.get("score", 0.0))
    return {"answer": ans, "score": score}

def no_context_flow(question: str, top_k: int = 3):
    cands = retrieve_top_k(question, k=top_k)
    if not cands:
        return {"answer": "", "score": 0.0, "used_context": "", "retrieved": []}
    best = {"answer": "", "score": -1.0, "used_context": ""}
    for c in cands:
        out = answer_with_context(question, c["context"])
        if out["score"] > best["score"]:
            best = {"answer": out["answer"], "score": out["score"], "used_context": c["context"]}
    return {"answer": best["answer"], "score": best["score"], "used_context": best["used_context"], "retrieved": cands}

INTRO_MD = """
### ShabdaAI
- **మోడ్ 1:** నేను ఇచ్చే ప్యాసేజ్ (context) పై సమాధానం ఇవ్వు  
- **మోడ్ 2:** ప్యాసేజ్ ఇవ్వకపోతే — చిన్న తెలుగు కార్పస్‌లో *సెర్చ్ → రీడ్* చేసి సమాధానం ఇవ్వు  

> Models: **intfloat/multilingual-e5-base** (retrieval) + **deepset/xlm-roberta-large-squad2** (extractive QA)
"""

def ui_answer(mode, translate_outputs_en, translate_inputs_en, question, user_context, top_k):
    question = question or ""
    user_context = user_context or ""

    # Optional English translations of inputs
    q_en = te_to_en(question) if translate_inputs_en and question else ""
    ctx_en = te_to_en(user_context) if translate_inputs_en and user_context else ""

    if mode == "With my context":
        res = answer_with_context(question, user_context)
        ans_te = res["answer"]
        ans_en = te_to_en(ans_te) if translate_outputs_en and ans_te else ""
        return ans_te, ans_en, f"{res['score']:.3f}", user_context, ctx_en or "—", q_en or "—", "—"

    else:
        res = no_context_flow(question, top_k=int(top_k))
        ans_te = res["answer"]
        ans_en = te_to_en(ans_te) if translate_outputs_en and ans_te else ""
        retrieved_tbl = "\n".join(
            [f"{r['rank']}. (sim={r['similarity']:.3f}) {r['context']}" for r in res.get("retrieved", [])]
        ) or "—"
        return ans_te, ans_en, f"{res['score']:.3f}", res["used_context"], ctx_en or "—", q_en or "—", retrieved_tbl


with gr.Blocks() as demo:
    gr.Markdown(INTRO_MD)

    with gr.Row():
        mode = gr.Radio(
            choices=["With my context", "No context (search sample data)"],
            value="With my context",
            label="Mode"
        )
        top_k = gr.Slider(1, 5, value=3, step=1, label="Top-K passages (for No-context mode)")
    with gr.Row():
        translate_outputs_en = gr.Checkbox(value=True, label="Translate ANSWER (Telugu → English)")
        translate_inputs_en  = gr.Checkbox(value=True, label="Translate INPUTS (Question/Context → English)")

    question = gr.Textbox(label="ప్రశ్న (Question)", placeholder="ఉదా: చార్మినార్ ఎక్కడ ఉంది?")
    user_context = gr.Textbox(label="ప్యాసేజ్ / కాంటెక్స్ట్ (optional)", lines=4)

    btn = gr.Button("Answer")

    # Answers
    answer_te = gr.Textbox(label="Answer (Telugu)")
    answer_en = gr.Textbox(label="Answer (English)")

    # Confidence + contexts
    score = gr.Textbox(label="Confidence score")
    used_ctx = gr.Textbox(label="Used context (Telugu)")
    ctx_en_box = gr.Textbox(label="Used context (English)")
    q_en_box = gr.Textbox(label="Question (English)")

    retrieved = gr.Textbox(label="Top-K retrieved passages (Telugu)", lines=4)

    btn.click(
        fn=ui_answer,
        inputs=[mode, translate_outputs_en, translate_inputs_en, question, user_context, top_k],
        outputs=[answer_te, answer_en, score, used_ctx, ctx_en_box, q_en_box, retrieved]
    )

if __name__ == "__main__":
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
