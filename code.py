"""
Compare AWS Textract, Azure Document Intelligence and Unstructured.io on the
same PDF and print a side-by-side summary of what each one pulls out.

Then run the cleanest text output through Claude to get structured JSON, which
is how we'd actually use this in a real pipeline.

Test doc is a fake NHS discharge summary (patient_discharge_summary.pdf).

Install:
    uv add boto3 azure-ai-documentintelligence unstructured \
        python-dotenv pandas tabulate anthropic

See .env.example for the credentials you need.
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import boto3
from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv()

PDF_PATH = "patient_discharge_summary.pdf"


@dataclass
class ExtractionResult:
    tool: str
    pages: int = 0
    raw_text_chars: int = 0
    tables_found: int = 0
    key_value_pairs: int = 0
    latency_seconds: float = 0.0
    tables: list = field(default_factory=list)
    kvp_sample: dict = field(default_factory=dict)
    raw_text_snippet: str = ""
    full_text: str = ""
    error: Optional[str] = None

    def summary(self) -> dict:
        return {
            "Tool": self.tool,
            "Pages": self.pages,
            "Text chars": f"{self.raw_text_chars:,}",
            "Tables": self.tables_found,
            "Key-Value pairs": self.key_value_pairs,
            "Latency (s)": f"{self.latency_seconds:.1f}",
            "Error": self.error or "—",
        }


def extract_textract(pdf_path):
    # Textract async needs the file in S3 first. The sync API only takes a
    # single-page TIFF/JPEG, which is no good for multi-page PDFs.
    result = ExtractionResult(tool="AWS Textract")
    t0 = time.time()

    bucket = os.getenv("AWS_S3_BUCKET")
    region = os.getenv("AWS_REGION", "eu-west-2")
    s3_key = f"textract-input/{os.path.basename(pdf_path)}"

    try:
        s3 = boto3.client("s3", region_name=region)
        textract = boto3.client("textract", region_name=region)

        print("[Textract] Uploading PDF to S3…")
        s3.upload_file(pdf_path, bucket, s3_key)

        response = textract.start_document_analysis(
            DocumentLocation={"S3Object": {"Bucket": bucket, "Name": s3_key}},
            FeatureTypes=["TABLES", "FORMS"],
        )
        job_id = response["JobId"]
        print(f"[Textract] Job started: {job_id}")

        while True:
            status = textract.get_document_analysis(JobId=job_id)
            job_status = status["JobStatus"]
            if job_status in ("SUCCEEDED", "FAILED"):
                break
            print(f"[Textract] Status: {job_status} — waiting…")
            time.sleep(3)

        if job_status == "FAILED":
            result.error = status.get("StatusMessage", "Unknown Textract error")
            return result

        # Results come back paginated, so walk the NextToken chain.
        blocks = status.get("Blocks", [])
        next_token = status.get("NextToken")
        while next_token:
            page_resp = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
            blocks.extend(page_resp.get("Blocks", []))
            next_token = page_resp.get("NextToken")

        raw_text_parts = []
        tables_count = 0
        kvp_count = 0
        page_set = set()
        block_map = {b["Id"]: b for b in blocks}

        for block in blocks:
            btype = block.get("BlockType")
            if btype == "LINE":
                raw_text_parts.append(block.get("Text", ""))
                page_set.add(block.get("Page", 1))
            elif btype == "TABLE":
                tables_count += 1
            elif btype == "KEY_VALUE_SET" and block.get("EntityTypes") == ["KEY"]:
                kvp_count += 1
                # Only bother resolving the first few KVPs for the sample output.
                if kvp_count <= 5:
                    key_text = " ".join(
                        block_map[rel["Ids"][0]].get("Text", "")
                        for rel in block.get("Relationships", [])
                        if rel["Type"] == "CHILD"
                        for wid in rel["Ids"]
                        if block_map.get(wid, {}).get("BlockType") == "WORD"
                    )
                    val_text = ""
                    for rel in block.get("Relationships", []):
                        if rel["Type"] == "VALUE":
                            for vid in rel["Ids"]:
                                val_block = block_map.get(vid, {})
                                for vrel in val_block.get("Relationships", []):
                                    if vrel["Type"] == "CHILD":
                                        for wid in vrel["Ids"]:
                                            w = block_map.get(wid, {})
                                            if w.get("BlockType") == "WORD":
                                                val_text += w.get("Text", "") + " "
                    if key_text.strip():
                        result.kvp_sample[key_text.strip()] = val_text.strip()

        raw_text = "\n".join(raw_text_parts)
        result.raw_text_chars = len(raw_text)
        result.raw_text_snippet = raw_text[:400]
        result.full_text = raw_text
        result.tables_found = tables_count
        result.key_value_pairs = kvp_count
        result.pages = len(page_set) or status.get("DocumentMetadata", {}).get("Pages", 0)
        result.latency_seconds = time.time() - t0

    except Exception as exc:
        result.error = str(exc)
        result.latency_seconds = time.time() - t0

    return result


def extract_azure(pdf_path):
    # prebuilt-layout is fine for general clinical docs. For actual form fields
    # (claims, referrals) you'd switch to prebuilt-document or a custom model.
    # Asking for markdown output keeps the tables readable for the RAG step.
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    result = ExtractionResult(tool="Azure Document Intelligence")
    t0 = time.time()

    endpoint = os.getenv("AZURE_ENDPOINT")
    key = os.getenv("AZURE_KEY")

    try:
        client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

        print("[Azure] Submitting document for analysis…")
        with open(pdf_path, "rb") as f:
            poller = client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=f,
                content_type="application/octet-stream",
                output_content_format="markdown",
            )
        doc = poller.result()

        raw_text = doc.content or ""
        result.raw_text_chars = len(raw_text)
        result.raw_text_snippet = raw_text[:400]
        result.full_text = raw_text
        result.pages = len(doc.pages) if doc.pages else 0

        result.tables_found = len(doc.tables) if doc.tables else 0
        for tbl in (doc.tables or [])[:3]:  # keep first 3 tables as a sample
            rows = {}
            for cell in tbl.cells:
                rows.setdefault(cell.row_index, {})[cell.column_index] = cell.content
            if rows:
                max_col = max(c for r in rows.values() for c in r) + 1
                table_list = [[rows[r].get(c, "") for c in range(max_col)] for r in sorted(rows)]
                result.tables.append(table_list)

        # Note: prebuilt-layout returns no KVPs. Use prebuilt-document for that.
        kvps = doc.key_value_pairs or []
        result.key_value_pairs = len(kvps)
        for kvp in kvps[:5]:
            k = (kvp.key.content if kvp.key else "").strip()
            v = (kvp.value.content if kvp.value else "").strip()
            if k:
                result.kvp_sample[k] = v

        result.latency_seconds = time.time() - t0

    except Exception as exc:
        result.error = str(exc)
        result.latency_seconds = time.time() - t0

    return result


def extract_unstructured(pdf_path):
    # Open source and self-hostable, which is the main reason to use it for
    # GDPR/HIPAA work. Run the API locally with:
    #   docker run -d -p 8000:8000 \
    #     downloads.unstructured.io/unstructured-io/unstructured-api:latest
    # OCR is weaker than Azure/Textract and there's no native form extraction.
    result = ExtractionResult(tool="Unstructured.io (local OSS)")
    t0 = time.time()

    try:
        api_key = os.getenv("UNSTRUCTURED_API_KEY")
        api_url = os.getenv("UNSTRUCTURED_API_URL")

        if api_key and api_url:
            from unstructured.partition.api import partition_via_api
            print("[Unstructured] Using hosted API…")
            elements = partition_via_api(
                filename=pdf_path,
                api_key=api_key,
                api_url=api_url,
                strategy="hi_res",
                infer_table_structure=True,
            )
        else:
            from unstructured.partition.pdf import partition_pdf
            print("[Unstructured] Running locally (OSS mode)…")
            elements = partition_pdf(
                filename=pdf_path,
                strategy="hi_res",
                infer_table_structure=True,
                include_page_breaks=True,
            )

        from unstructured.documents.elements import Table, PageBreak

        raw_text_parts = []
        tables_count = 0
        page_count = 1
        element_type_counts = {}

        for el in elements:
            etype = type(el).__name__
            element_type_counts[etype] = element_type_counts.get(etype, 0) + 1

            if isinstance(el, PageBreak):
                page_count += 1
                continue
            if isinstance(el, Table):
                tables_count += 1
                result.tables.append(el.text)
            else:
                raw_text_parts.append(el.text)

        raw_text = "\n".join(raw_text_parts)
        result.raw_text_chars = len(raw_text)
        result.raw_text_snippet = raw_text[:400]
        result.full_text = raw_text
        result.tables_found = tables_count
        result.pages = page_count
        result.latency_seconds = time.time() - t0

        result.key_value_pairs = 0
        result.kvp_sample = {
            "Note": "Unstructured does not extract KVPs natively. "
                    "Post-process with an LLM (Claude/GPT) for structured extraction.",
            "Element types found": str(element_type_counts),
        }

    except Exception as exc:
        result.error = str(exc)
        result.latency_seconds = time.time() - t0

    return result


def llm_extract_structured(raw_text, provider="anthropic"):
    # Once we have text from any of the engines, hand it to an LLM to pull out
    # the fields we care about. The whole point is that the extractor is
    # swappable - this step doesn't change when you switch engines.
    schema = {
        "patient_name": None,
        "nhs_number": None,
        "date_of_birth": None,
        "admission_date": None,
        "discharge_date": None,
        "primary_diagnosis": None,
        "icd10_codes": [],
        "allergies": [],
        "discharge_medications": [],
        "follow_up_actions": [],
    }

    prompt = f"""
You are a clinical document parser. Extract the following fields from the
discharge summary below. Return ONLY valid JSON matching the schema.

Schema:
{json.dumps(schema, indent=2)}

Document:
\"\"\"
{raw_text[:6000]}
\"\"\"

Return only the JSON object. No explanation.
"""

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()

    # Models sometimes wrap the JSON in a code fence, so strip it before parsing.
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def run_comparison():
    print("\n" + "=" * 70)
    print("  Document Extraction Comparison")
    print("  AWS Textract  |  Azure Document Intelligence  |  Unstructured.io")
    print("=" * 70 + "\n")

    results = []

    print("── Running AWS Textract ──")
    results.append(extract_textract(PDF_PATH))

    print("\n── Running Azure Document Intelligence ──")
    results.append(extract_azure(PDF_PATH))

    print("\n── Running Unstructured.io ──")
    results.append(extract_unstructured(PDF_PATH))

    print("\n\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(tabulate([r.summary() for r in results], headers="keys", tablefmt="rounded_grid"))

    print("\n\nKEY-VALUE PAIRS EXTRACTED (sample)\n")
    for r in results:
        print(f"  [{r.tool}]")
        for k, v in list(r.kvp_sample.items())[:5]:
            print(f"    {k:40s} → {v}")
        print()

    print("\nRAW TEXT SNIPPET (first 300 chars)\n")
    for r in results:
        print(f"  [{r.tool}]")
        print(f"  {r.raw_text_snippet[:300]!r}")
        print()

    # Run the LLM step on Azure's output since the markdown text is cleanest.
    azure_result = next((r for r in results if "Azure" in r.tool), None)
    if azure_result and not azure_result.error and azure_result.full_text:
        print("\n" + "=" * 70)
        print("LLM STRUCTURED EXTRACTION (Claude on Azure output)")
        print("=" * 70)
        try:
            structured = llm_extract_structured(azure_result.full_text)
            print(json.dumps(structured, indent=2))
        except Exception as e:
            print(f"LLM extraction skipped: {e}")

    print("\n\n" + "=" * 70)
    print("TOOL SELECTION GUIDE")
    print("=" * 70)
    guide = [
        ["Scanned / handwritten clinical docs", "Azure Document Intelligence"],
        ["Mixed PDFs, need self-hosted / GDPR", "Unstructured.io (OSS)"],
        ["Already on AWS, need S3/Lambda pipeline", "AWS Textract"],
        ["Structured forms (claims, referrals)", "Azure prebuilt-document"],
        ["RAG pipeline, LangChain/LlamaIndex", "Unstructured.io + Qdrant"],
        ["Max accuracy, budget not a constraint", "Azure + LLM post-processing"],
        ["High volume, cost-sensitive", "Unstructured.io (self-hosted)"],
    ]
    print(tabulate(guide, headers=["Your situation", "Best choice"], tablefmt="rounded_grid"))


if __name__ == "__main__":
    run_comparison()
