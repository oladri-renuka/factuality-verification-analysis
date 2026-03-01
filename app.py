import gradio as gr
import torch
import wikipedia
import re
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rank_bm25 import BM25Okapi

# ── Load RoBERTa-MNLI once at startup ─────────────────────────────────────────
print("Loading RoBERTa-Large-MNLI...")
nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
nli_model     = AutoModelForSequenceClassification.from_pretrained(
    "roberta-large-mnli"
)
nli_model.eval()
print("Model ready.")

# ── BM25 helpers ──────────────────────────────────────────────────────────────
def get_name_tokens(topic):
    stopwords = {"the","a","an","of","and","or","in","on","at",
                 "to","for","with","by","from","mr","mrs","dr","sir"}
    return set(topic.lower().split()) - stopwords

def bm25_retrieve(sentences, query, name_tokens):
    def clean(text):
        return [w for w in text.lower().split()
                if w not in name_tokens and len(w) > 2]
    corpus  = [clean(s) for s in sentences]
    q_clean = clean(query)
    if not any(corpus) or not q_clean:
        return sentences[0] if sentences else ""
    scores = BM25Okapi(corpus).get_scores(q_clean)
    return sentences[int(np.argmax(scores))]

def get_wikipedia_sentences(topic):
    try:
        results = wikipedia.search(topic, results=3)
        for r in results:
            try:
                page  = wikipedia.page(r, auto_suggest=False)
                sents = re.split(r'(?<=[.!?])\s+', page.content)
                return [s.strip() for s in sents if len(s.strip()) > 20][:300]
            except Exception:
                continue
    except Exception:
        pass
    return []

# ── NLI scoring ───────────────────────────────────────────────────────────────
def nli_entailment_prob(evidence, claim):
    enc = nli_tokenizer(
        evidence, claim,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        logits = nli_model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    # roberta-large-mnli: 0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT
    return probs[2].item()

# ── Core verification function ────────────────────────────────────────────────
def verify_biography(topic, biography):
    if not topic.strip():
        return "Please enter a person's name.", None

    if not biography.strip():
        return "Please enter a biography.", None

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', biography.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if not sentences:
        return "No sentences found.", None

    # Get Wikipedia content once
    wiki_sents  = get_wikipedia_sentences(topic)
    name_tokens = get_name_tokens(topic)

    if not wiki_sents:
        return (f"Could not find Wikipedia page for '{topic}'. "
                f"Try a more specific name."), None

    rows = []
    for sent in sentences:
        evidence = bm25_retrieve(wiki_sents, sent, name_tokens)
        prob     = nli_entailment_prob(evidence, sent)

        verdict_default    = "✅ Supported" if prob >= 0.50 else "❌ Not Supported"
        verdict_calibrated = "✅ Supported" if prob >= 0.10 else "❌ Not Supported"

        rows.append({
            "Sentence": sent[:120] + "..." if len(sent) > 120 else sent,
            "Entailment Prob": round(prob, 3),
            "Default (0.50)": verdict_default,
            "Calibrated (0.10) ★": verdict_calibrated,
            "Evidence (truncated)": evidence[:120] + "..." if len(evidence) > 120 else evidence,
        })

    df = pd.DataFrame(rows)

    # Summary
    n   = len(rows)
    n_d = sum(1 for r in rows if "✅" in r["Default (0.50)"])
    n_c = sum(1 for r in rows if "✅" in r["Calibrated (0.10) ★"])

    changed = n_c - n_d
    summary = (
        f"### {topic} — {n} sentences analyzed\n\n"
        f"| Threshold | Supported | Rate |\n"
        f"|-----------|-----------|------|\n"
        f"| Default (0.50) | {n_d}/{n} | {n_d/n:.0%} |\n"
        f"| **Calibrated (0.10) ★** | **{n_c}/{n}** | **{n_c/n:.0%}** |\n\n"
        f"**Key finding demonstrated:** Threshold calibration changes verdict "
        f"for **{changed} sentence(s)**. "
        f"The calibrated threshold recovers facts the default incorrectly rejects — "
        f"this is the main finding of the paper (+0.076 F1 improvement)."
    )

    return summary, df

# ── Example inputs ────────────────────────────────────────────────────────────
examples = [
    [
        "Errol Flynn",
        "Errol Flynn was an Australian actor born on June 20, 1909. "
        "He was known for his swashbuckling roles in Hollywood films. "
        "He starred in Captain Blood in 1935. "
        "He was notorious for his hard-partying lifestyle. "
        "He died in 1959 at the age of 50."
    ],
    [
        "Marie Curie",
        "Marie Curie was a Polish physicist and chemist. "
        "She was the first woman to win a Nobel Prize. "
        "She won the Nobel Prize in Physics in 1903. "
        "She was born in Warsaw in 1867. "
        "She discovered the elements polonium and radium."
    ],
    [
        "Harrison Ford",
        "Harrison Ford is an American actor born on July 13, 1942. "
        "He is best known for playing Han Solo in Star Wars. "
        "He also portrayed Indiana Jones in Raiders of the Lost Ark. "
        "Ford has been in films of several genres."
    ],
]

# ── Interface ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Atomic Factuality Verifier", theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div style='text-align:center; padding:20px 0 10px 0'>
        <h1>🔍 Atomic Factuality Verifier</h1>
        <p style='color:#555; max-width:720px; margin:0 auto; font-size:1.05em'>
        Paste an LLM-generated biography and verify each sentence against
        Wikipedia using <b>BM25 retrieval + RoBERTa-Large-MNLI</b>.<br><br>
        Demonstrates the key research finding:
        <b>lowering the NLI threshold from 0.50 → 0.10 improves F1 by +0.076</b>,
        a larger gain than switching verification strategies entirely.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            topic_box = gr.Textbox(
                label="Person Name",
                placeholder="e.g. Errol Flynn",
                lines=1
            )
            bio_box = gr.Textbox(
                label="Biography (LLM-generated)",
                placeholder="Paste a biography here. Each sentence will be verified independently.",
                lines=10
            )
            verify_btn = gr.Button("Verify Biography", variant="primary", size="lg")

            gr.Examples(
                examples=examples,
                inputs=[topic_box, bio_box],
                label="Try these examples"
            )

        with gr.Column(scale=1):
            summary_out = gr.Markdown(label="Summary")
            table_out   = gr.DataFrame(
                label="Sentence-level Verification Results",
                wrap=True
            )

    gr.HTML("""
    <div style='margin:24px auto; padding:18px;
                background:#f0f4ff; border-radius:10px;
                max-width:900px'>
        <h3>📊 Research Results (14,525 human-annotated facts · 183 entities)</h3>
        <table style='width:100%; border-collapse:collapse; font-size:0.95em'>
            <tr style='background:#2C5F8A; color:white; text-align:center'>
                <th style='padding:8px; text-align:left'>Method</th>
                <th>F1</th><th>Precision</th><th>Recall</th><th>Kappa</th>
            </tr>
            <tr style='text-align:center'>
                <td style='padding:6px'>A: NLI (threshold=0.50, default)</td>
                <td>0.651</td><td>0.953</td><td>0.495</td><td>0.335</td>
            </tr>
            <tr style='background:#eaf4ea; font-weight:bold; text-align:center'>
                <td style='padding:6px'>A: NLI (threshold=0.10, calibrated) ★</td>
                <td>0.727</td><td>0.919</td><td>0.602</td><td>0.393</td>
            </tr>
            <tr style='text-align:center'>
                <td style='padding:6px'>B: Retrieval+LLM (BM25+Qwen2.5-3B)</td>
                <td>0.667</td><td>0.957</td><td>0.512</td><td>0.354</td>
            </tr>
            <tr style='text-align:center'>
                <td style='padding:6px'>C: Direct LLM (no retrieval)</td>
                <td>0.273</td><td>0.910</td><td>0.161</td><td>0.081</td>
            </tr>
        </table>
        <p style='margin-top:12px; font-size:0.88em; color:#555'>
            ★ Calibrated threshold achieves best F1 — the central finding of this work.<br>
            Four failure modes identified: inferential facts (~40%), retrieval failures (~25%),
            world knowledge facts (~20%), relational/compound facts (~15%).
        </p>
        <p style='margin-top:8px; font-size:0.88em'>
            <a href='https://github.com/oladri-renuka/factuality-verification-analysis'
               target='_blank'>📁 GitHub Repository</a> ·
            <em>Oladri Renuka · UMD Applied ML</em>
        </p>
    </div>
    """)

    verify_btn.click(
        fn=verify_biography,
        inputs=[topic_box, bio_box],
        outputs=[summary_out, table_out]
    )

if __name__ == "__main__":
    demo.launch()
