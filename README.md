# Document Extraction Comparison — AWS Textract vs Azure Document Intelligence vs Unstructured.io

A side-by-side benchmark of three Document AI engines run against the **same** PDF
(a fictional NHS patient discharge summary), followed by an LLM normalisation step
that turns raw extracted text into clean, structured JSON.

This is the reference implementation behind the article
**"Which Document AI tool is right for your stack?"** by
[nuvraxis](https://nuvraxis.com/articles/rag-pipeline).

> The PDF used here (`patient_discharge_summary.pdf`) is **fictional** and exists
> for demonstration purposes only — no real patient data is involved.

---

## What it does

[code.py](code.py) runs all three extractors on the same document and produces a
side-by-side comparison of what each tool can — and can't — extract:

| Stage | Engine | Notes |
|-------|--------|-------|
| 1. Extraction | **AWS Textract** | Async `StartDocumentAnalysis` with `TABLES` + `FORMS`; PDF uploaded to S3 first (async requires S3). Native key-value pair extraction. |
| 2. Extraction | **Azure Document Intelligence** | `prebuilt-layout` model, Markdown output (preserves table structure for RAG). Strong OCR on scanned/handwritten docs. |
| 3. Extraction | **Unstructured.io** | Open-source, self-hostable (GDPR/HIPAA-friendly). Runs locally via Docker. Returns typed elements (Title, NarrativeText, Table, ListItem…). |
| 4. Normalisation | **Claude (Anthropic)** | Post-processes any engine's output into a structured clinical JSON schema — the production RAG pattern: fast extraction + LLM normalisation. |

The key design idea: the extraction engine is an **interchangeable backend**. You
can swap Textract for Azure for Unstructured without changing a line of downstream
application code.

---

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/).

```bash
# install dependencies
uv sync

# copy and fill in credentials
cp .env.example .env
```

Required environment variables (see [.env.example](.env.example)):

```dotenv
# AWS Textract
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=eu-west-2
AWS_S3_BUCKET=your-textract-input-bucket

# Azure Document Intelligence
AZURE_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_KEY=...

# Unstructured.io (Docker local)
UNSTRUCTURED_API_URL=http://localhost:8000
UNSTRUCTURED_API_KEY=local

# LLM normalisation step
ANTHROPIC_API_KEY=sk-ant-...
```

To run Unstructured.io locally:

```bash
docker run -d -p 8000:8000 \
  downloads.unstructured.io/unstructured-io/unstructured-api:latest
```

## Run

```bash
uv run code.py
```

---

## Sample output

Running against `patient_discharge_summary.pdf` (3-page fictional discharge summary):

### Results summary

| Tool                        | Pages | Text chars | Tables | Key-Value pairs | Latency (s) | Error |
|-----------------------------|------:|-----------:|-------:|----------------:|------------:|:-----:|
| AWS Textract                |     3 |      6,525 |      6 |              45 |         7.0 |   —   |
| Azure Document Intelligence |     2 |      6,258 |      4 |               0 |         5.4 |   —   |
| Unstructured.io (local OSS) |     1 |      3,813 |      6 |               0 |         6.9 |   —   |

**Takeaways:**
- **AWS Textract** is the only engine with native key-value extraction (45 KVPs) — best for forms.
- **Azure** is fastest and returns clean Markdown; `prebuilt-layout` doesn't extract KVPs (use `prebuilt-document` for that).
- **Unstructured.io** is free/self-hostable but has weaker OCR; pair it with an LLM for structured fields.

### Key-value pairs extracted (sample)

**AWS Textract**
```
Age:                    → 72
Consultant:             → Mr. James R. Hadfield, FRCS
Discharge:              → Home with community nursing
Tel:                    → +44 20 7946 0800
Clinical summary:       → Mrs. Thornton, a 72-year-old female with a background of
                          T2DM, hypertension, and CKD Stage 3, presented to A&E on
                          09/11/2024 with a 3-day history of productive cough,
                          dyspnoea on exertion, and markedly elevated blood glucose
                          (CBG on admission: 24.3 mmol/L)…
```

**Azure Document Intelligence**
```
(prebuilt-layout does not extract KVPs — switch to prebuilt-document)
```

**Unstructured.io (local OSS)**
```
Note:                   → Unstructured does not extract KVPs natively.
                          Post-process with an LLM (Claude/GPT) for structured extraction.
Element types found:    → {'Title': 12, 'NarrativeText': 13, 'Text': 6, 'Header': 3,
                           'Table': 6, 'ListItem': 4, 'FigureCaption': 1}
```

### Raw text snippet (first 300 chars)

**AWS Textract**
```
Meridian General Hospital
CONFIDENTIAL — NHS PROTECTED
Document Class: Clinical - Discharge Summary
Department of Internal Medicine 123 Medical Drive, London EC1A 1BB
Tel: +44 20 7946 0800 | Fax: +44 20 7946 0801 www.meridianhospital.nhs.uk
FICTIONAL DOCUMENT - FOR DEMONSTRATION PURPOSES ONLY
```

**Azure Document Intelligence**
```
<!-- PageHeader="Meridian General Hospital" -->
<!-- PageHeader="Department of Internal Medicine | 123 Medical Drive, London EC1A 1BB
Tel: +44 20 7946 0800 | Fax: +44 20 7946 0801 | www.meridianhospital.nhs.uk" -->
<!-- PageHeader="CONFIDENTIAL - NHS PROTECTED Document Class: Clinical - Discharge Su…
```

**Unstructured.io (local OSS)**
```
Meridian General Hospital
Department of Internal Medicine | 123 Medical Drive, London EC1A 1BB
Tel: +44 20 7946 0800 | Fax: +44 20 7946 0801 | www.meridianhospital.nhs.uk
CONFIDENTIAL — NHS PROTECTED
Document Class: Clinical — Discharge Summary
FICTIONAL DOCUMENT — FOR DEMONSTRATION PURPOSES ONLY
```

### LLM structured extraction (Claude on Azure output)

After extraction, Claude normalises the raw text into a clinical JSON schema:

```json
{
  "patient_name": "Margaret Eleanor Thornton",
  "nhs_number": "485 777 3321",
  "date_of_birth": "14 March 1952",
  "admission_date": "09 November 2024",
  "discharge_date": "15 November 2024",
  "primary_diagnosis": "Type 2 Diabetes Mellitus - uncontrolled",
  "icd10_codes": ["E11.65", "I11.0", "N18.3", "J18.9", "E11.40"],
  "allergies": ["Penicillin - rash (documented 2019)"],
  "discharge_medications": [
    {
      "medication": "Amoxicillin 500mg",
      "dose": "500 mg",
      "route": "Oral",
      "frequency": "TDS",
      "notes": "Complete 7-day course (Days 1-7 post-discharge)"
    },
    {
      "medication": "Insulin Glargine (Lantus)",
      "dose": "24 units",
      "route": "SC",
      "frequency": "OD nocte",
      "notes": "Increased from 18u - review in 4 weeks"
    },
    {
      "medication": "NovoRapid (Insulin Aspart)",
      "dose": "4-8 units",
      "route": "SC",
      "frequency": "TDS with meals",
      "notes": "Sliding scale guidance provided"
    },
    {
      "medication": "Amlodipine",
      "dose": "10 mg",
      "route": "Oral",
      "frequency": "OD",
      "notes": "Ongoing - BP target < 130/80"
    },
    {
      "medication": "Ramipril",
      "dose": "5 mg",
      "route": "Oral",
      "frequency": "OD",
      "notes": "Monitor eGFR + K+ in 2 weeks"
    },
    {
      "medication": "Furosemide",
      "dose": "40 mg",
      "route": "Oral",
      "frequency": "OD morning",
      "notes": "Review at 2-week follow-up"
    },
    {
      "medication": "Atorvastatin",
      "dose": "40 mg",
      "route": "Oral",
      "frequency": "OD nocte",
      "notes": "Ongoing"
    },
    {
      "medication": "Aspirin",
      "dose": "75 mg",
      "route": "Oral",
      "frequency": "OD",
      "notes": "Ongoing - gastroprotection with PPI"
    },
    {
      "medication": "Omeprazole",
      "dose": "20 mg",
      "route": "Oral",
      "frequency": "OD",
      "notes": null
    }
  ],
  "follow_up_actions": [
    "Review insulin regimen in 4 weeks",
    "Monitor eGFR and potassium in 2 weeks",
    "Review at 2-week follow-up",
    "Community nursing support at home"
  ]
}
```

---

## Tool selection guide

| Your situation                          | Best choice                   |
|-----------------------------------------|-------------------------------|
| Scanned / handwritten clinical docs     | Azure Document Intelligence   |
| Mixed PDFs, need self-hosted / GDPR     | Unstructured.io (OSS)         |
| Already on AWS, need S3/Lambda pipeline | AWS Textract                  |
| Structured forms (claims, referrals)    | Azure prebuilt-document       |
| RAG pipeline, LangChain/LlamaIndex      | Unstructured.io + Qdrant      |
| Max accuracy, budget not a constraint   | Azure + LLM post-processing   |
| High volume, cost-sensitive             | Unstructured.io (self-hosted) |

---

## Project structure

```
rag-pipeline-comparison/
├── code.py                          # the comparison runner (all 3 engines + LLM step)
├── main.py                          # placeholder entrypoint
├── patient_discharge_summary.pdf    # fictional NHS discharge summary (demo input)
├── .env.example                     # credential template
├── pyproject.toml                   # uv project + dependencies
└── README.md
```

---

*nuvraxis builds production Document AI + RAG pipelines.
Contact: hello@nuvraxis.com · [nuvraxis.com](https://nuvraxis.com)*
