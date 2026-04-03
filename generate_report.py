from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

SCREENSHOT_DIR = "/Users/dudoan/.gemini/antigravity/brain/f41e723d-db42-4f5f-aa05-9cd4f900184a"
SCREENSHOTS = {
    "home":      f"{SCREENSHOT_DIR}/homepage_before_input_1775155875887.png",
    "results":   f"{SCREENSHOT_DIR}/analysis_results_high_risk_1775155910771.png",
    "full":      f"{SCREENSHOT_DIR}/full_analysis_results_high_risk_1775155920313.png",
    "adv1a":     f"{SCREENSHOT_DIR}/adversarial_test1_slippery_slope_part1_1775157003543.png",
    "adv1b":     f"{SCREENSHOT_DIR}/adversarial_test1_slippery_slope_part2_1775157007592.png",
    "adv2a":     f"{SCREENSHOT_DIR}/adversarial_test2_screenwriter_part1_1775157061831.png",
    "adv2b":     f"{SCREENSHOT_DIR}/adversarial_test2_screenwriter_part2_1775157066636.png",
    "adv3a":     f"{SCREENSHOT_DIR}/adversarial_test3_slang_part1_1775157122228.png",
    "adv3b":     f"{SCREENSHOT_DIR}/adversarial_test3_slang_part2_1775157127604.png",
}

doc = Document()
for section in doc.sections:
    section.top_margin = Inches(0.9); section.bottom_margin = Inches(0.9)
    section.left_margin = Inches(1);  section.right_margin  = Inches(1)

def h(doc, text, level=1, color=None):
    p = doc.add_heading(text, level=level)
    if color:
        for run in p.runs:
            run.font.color.rgb = RGBColor(*color)
    return p

def b(doc, text, bold=False, italic=False, size=10):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold; run.italic = italic; run.font.size = Pt(size)
    return p

def caption(doc, text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in p.runs:
        run.italic = True; run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(100, 100, 100)

def add_table(doc, headers, rows):
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.style = "Light Shading Accent 1"
    for i, h_text in enumerate(headers):
        tbl.rows[0].cells[i].text = h_text
        for run in tbl.rows[0].cells[i].paragraphs[0].runs:
            run.bold = True; run.font.size = Pt(9)
    for row_data in rows:
        row = tbl.add_row()
        for i, val in enumerate(row_data):
            row.cells[i].text = val
            for run in row.cells[i].paragraphs[0].runs:
                run.font.size = Pt(9)
    return tbl

def img(doc, key, width=5.5):
    import os
    path = SCREENSHOTS.get(key)
    if path and os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER

# ── TITLE ────────────────────────────────────────────────
title = doc.add_heading("SubstanceWatch AI", 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in title.runs:
    run.font.size = Pt(20); run.font.color.rgb = RGBColor(31, 73, 125)

for text, size in [
    ("Challenge 1: AI for Substance Abuse Risk Detection from Social Signals", 12),
    ("NSF NRT Research-A-Thon 2026  |  UMKC  |  Track A: AI Modeling & Reasoning", 10),
    ("Team: Joe Doan · Manan Koradiya · Ruixuan Hou · Aditya Naredla", 10),
]:
    p = doc.add_paragraph(text); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in p.runs:
        run.font.size = Pt(size)
        if size == 12: run.bold = True

doc.add_paragraph()

# ── §1 INTRODUCTION ──────────────────────────────────────
h(doc, "1. Introduction & Problem Statement")
b(doc, "Substance abuse causes over 100,000 overdose deaths annually in the U.S. Warning signals frequently surface in online communities before escalating to physical crises, yet traditional monitoring is delayed and retrospective. This project presents SubstanceWatch AI: an end-to-end AI pipeline that identifies substance abuse and emotional distress signals from anonymized social media data. Our core innovation is a Dual-Threshold Architecture that combines a high-Recall ML Tripwire with a high-Precision Gemini 2.5 LLM Evaluator backed by Retrieval-Augmented Generation (RAG), transforming noisy text into interpretable, evidence-grounded public health insights.")

# ── §2 DATASETS ──────────────────────────────────────────
h(doc, "2. Dataset & Data Strategy")
b(doc, "We curated an ensemble of three public, anonymized datasets:")
add_table(doc,
    ["Dataset", "Source / Size", "Purpose"],
    [
        ["Reddit Mental Health Classification", "HuggingFace · 1.1M posts", "Core: addiction, alcoholism, depression, PTSD, suicidewatch"],
        ["Mental Health Corpus", "Kaggle · 28K posts", "Binary baseline: normal (0) vs. emotional distress (1)"],
        ["KUC Hackathon 2018 (reference)", "Kaggle · 200K reviews", "Pre-approved medical drug review reference baseline"],
    ]
)
doc.add_paragraph()
b(doc, "Stratified Sub-sampling: We engineered a targeted 43,551-row subset — ALL addiction (~7.7k) and alcoholism (~5.5k) posts; 5,000 from each of depression, suicidewatch, anxiety, ptsd, mentalhealth; and 5,000 safe control posts — ensuring dense, well-separated clusters for both direct substance signals and underlying emotional distress.", italic=True)

# ── §3 ARCHITECTURE ──────────────────────────────────────
h(doc, "3. System Architecture: Dual-Threshold Pipeline")
add_table(doc,
    ["Layer", "Technology", "Role & Threshold"],
    [
        ["1. ML Tripwire", "TF-IDF + Logistic Regression", "Maximize Recall — cast a wide net. Threshold: 0.4. Output: 🟡 INITIAL SIGNAL DETECTED"],
        ["2. RAG Retrieval", "all-MiniLM-L6-v2 + ChromaDB", "Retrieve high-confidence context. Strict L2 Distance < 1.2 threshold blocks hallucination."],
        ["3. LLM Evaluator", "Gemini 2.5", "Temporal + intent reasoning: Active Risk vs. Retrospective/Recovery. Output: 🔴/🟢 FINAL AI ASSESSMENT"],
    ]
)
doc.add_paragraph()
b(doc, "Design Rationale: In public health, missing a genuine distress signal (False Negative) is more dangerous than a False Positive. The ML layer is tuned for maximum Recall (97.29%), while the LLM applies Precision-focused contextual reasoning to filter noise, including correctly classifying retrospective/recovery narratives and promotional content as low risk.", italic=True)
doc.add_paragraph()
h(doc, "Dashboard Interface", level=2)
img(doc, "home")
caption(doc, "Figure 1: SubstanceWatch AI Dashboard — Public Health Signal Detection interface.")
doc.add_paragraph()
img(doc, "results")
caption(doc, "Figure 2: Dual-Threshold output — ML Tripwire flags the input; Gemini 2.5 confirms 🔴 FINAL AI ASSESSMENT: HIGH RISK.")

# ── §4 EVALUATION ────────────────────────────────────────
h(doc, "4. Evaluation Results")
b(doc, "The ML pipeline was evaluated on an 80/20 train-test split of the 43,551-row stratified subset:")
add_table(doc,
    ["Class", "Precision", "Recall", "F1 Score", "Support"],
    [
        ["Safe (Control)", "0.82", "0.93", "0.87", "1,002"],
        ["High Risk", "0.9905", "0.9729", "0.9816", "7,709"],
        ["Weighted Avg", "0.97", "0.97", "0.97", "8,711"],
    ]
)
doc.add_paragraph()
img(doc, "full", width=5.5)
caption(doc, "Figure 3: Full LLM output — Risk Summary, direct quote-cited Supporting Evidence, and RAG-driven behavioral pattern matching.")

# ── §5 ADVERSARIAL TESTING ───────────────────────────────
h(doc, "5. Adversarial Testing (Stress-Test)")
b(doc, "To validate that the system reasons beyond keyword matching, we designed three adversarial test scenarios engineered to expose the failure modes of naive classifiers. All three passed with the expected outcomes:", size=10)
doc.add_paragraph()
add_table(doc,
    ["Scenario", "Trap Type", "ML Tripwire", "LLM Assessment", "Result"],
    [
        ["The Slippery Slope", "Recovery language hiding imminent relapse", "🟡 INITIAL SIGNAL", "🔴 HIGH RISK", "✅ PASS"],
        ["The Screenwriter", "Drug keywords inside fictional character dialogue", "🟡 INITIAL SIGNAL", "🟢 LOW RISK", "✅ PASS"],
        ["The Slang & Metaphor", "Cocaine use disguised as skiing slang", "🟡 INITIAL SIGNAL", "🔴 HIGH RISK", "✅ PASS"],
    ]
)
doc.add_paragraph()

# Scenario 1
h(doc, "Scenario 1 — The Slippery Slope", level=2)
b(doc, "Trap: The post opens with '5 years clean' — a recovery narrative — but ends with the user parked outside their old drug-buying neighborhood, actively rationalizing 'just one hit wouldn't reset the clock.' The LLM must detect that this is an active crisis, not a retrospective story.")
img(doc, "adv1a", width=5.3)
caption(doc, "Figure 4a: Test 1 — ML correctly flags INITIAL SIGNAL; LLM correctly escalates to 🔴 HIGH RISK despite recovery framing.")
img(doc, "adv1b", width=5.3)
caption(doc, "Figure 4b: Test 1 — LLM Supporting Evidence: 'I'm sitting in my car outside the old neighborhood' and 'just one hit wouldn't reset the clock' identified as active relapse signals.")

doc.add_paragraph()

# Scenario 2
h(doc, "Scenario 2 — The Screenwriter", level=2)
b(doc, "Trap: The post is saturated with high-risk terms — 'opioid', 'cravings', 'dealer', 'withdrawal' — but all are attributed to a fictional protagonist inside quotation marks. The LLM must extract that the author is a screenwriter seeking community feedback, not an individual in distress.")
img(doc, "adv2a", width=5.3)
caption(doc, "Figure 5a: Test 2 — ML flags INITIAL SIGNAL from drug keywords; LLM correctly downgrades to 🟢 LOW RISK after intent analysis.")
img(doc, "adv2b", width=5.3)
caption(doc, "Figure 5b: Test 2 — Evidence: 'I'm a writer working on a screenplay' and the quoted fictional dialogue correctly identified as non-personal creative research.")

doc.add_paragraph()

# Scenario 3
h(doc, "Scenario 3 — The Slang & Metaphor", level=2)
b(doc, "Trap: The post contains zero explicit drug keywords. Instead, 'hitting the slopes,' 'white noise,' and 'skiing' are street slang for cocaine use, embedded in a context of job loss and relationship breakdown. The LLM must decode the implicit semantic intent without explicit keyword matches.")
img(doc, "adv3a", width=5.3)
caption(doc, "Figure 6a: Test 3 — ML flags initial signals from distress language; LLM correctly escalates to 🔴 HIGH RISK by decoding slang metaphors.")
img(doc, "adv3b", width=5.3)
caption(doc, "Figure 6b: Test 3 — Evidence: 'hitting the slopes tonight just to numb it all out' and 'white noise' decoded as cocaine relapse signals in context of compounding stressors.")

# ── §6 INNOVATION & CONCLUSION ───────────────────────────
h(doc, "6. Innovation Highlights & Conclusion")
for title_text, desc in [
    ("Dual-Threshold Design:", "Separates the high-Recall 'catching' stage (ML) from the high-Precision 'filtering' stage (LLM+RAG) — a novel pattern for public health risk systems."),
    ("Active vs. Retrospective Classification:", "Gemini 2.5 correctly differentiated recovery narratives, creative/fictional content, and slang-based active crisis posts — nuances no binary classifier can capture."),
    ("Adversarial Test Results:", "All 3 stress-test scenarios passed with expected outcomes, demonstrating genuine contextual reasoning beyond keyword matching."),
    ("Grounded LLM Outputs:", "All LLM summaries are evidence-grounded via ChromaDB. Strict L2 < 1.2 threshold blocks hallucination on low-confidence inputs."),
    ("Ethical AI by Design:", "No PII at any stage. Population-level analysis only, with full data anonymization."),
]:
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(title_text + " "); run.bold = True; run.font.size = Pt(10)
    p.add_run(desc).font.size = Pt(10)

doc.add_paragraph()
b(doc, "Future Work: Integrate CDC/NIDA temporal data for population-level spike detection. Add BERTopic topic modeling for emerging theme discovery. Deploy a temporal dashboard for community-level behavioral shift analysis.")

# ── §7 REFERENCES ────────────────────────────────────────
h(doc, "7. References")
for i, ref in enumerate([
    "CDC National Drug Overdose Data: https://data.cdc.gov",
    "Reddit Mental Health Classification (HuggingFace): kamruzzaman-asif/reddit-mental-health-classification",
    "Mental Health Corpus (Kaggle): reihanenamdari/mental-health-corpus",
    "Reimers & Gurevych (2019). Sentence-BERT. arXiv:1908.10084",
    "Lewis et al. (2020). Retrieval-Augmented Generation. arXiv:2005.11401",
    "Google Gemini 2.5: https://deepmind.google/technologies/gemini/",
], 1):
    p = doc.add_paragraph(f"{i}. {ref}", style='List Number')
    for run in p.runs: run.font.size = Pt(9)

# ── SAVE ─────────────────────────────────────────────────
out = "/Users/dudoan/Documents/UMKC/Big Data Analytics/AI model project/SubstanceWatch_AI_Report.docx"
doc.save(out)
print(f"✅ Report saved: {out}")
