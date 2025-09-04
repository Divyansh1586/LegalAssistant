# extract_with_gemini_sequential.py
import os
import json
import re
import asyncio
import random
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Constants and Configuration ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("⚠️ GOOGLE_API_KEY not set. Add it to your .env file.")

INPUT_FILE = "constitution_of_india.json"
OUTPUT_FILE = "graph_fragments.json"
LOG_FILE = "error_log.txt"

# Initialize the LLM with the specified model and settings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# Prompt for the knowledge graph extraction
PROMPT = """
Extract a knowledge graph from the following constitutional article.
Return JSON in the format:
{{
  "nodes": [
    {{"id": "Article:21A", "label": "Article", "properties": {{"number": "21A", "title": "...", "snippet": "..."}}}}
  ],
  "edges": [
    {{"source_id": "Article:21A", "target_id": "Article:45", "type": "REFERS_TO", "properties": {{"snippet": "..."}}}}
  ]
}}
Allowed node labels: Article, Part, Schedule, Amendment, CaseLaw, Concept, Subject.
Allowed edge types: CONTAINS, REFERS_TO, AMENDED_BY, INTERPRETS, MENTIONS, HAS_SUBJECT.
Keep JSON strictly valid and concise.
TEXT:
\"\"\"{text}\"\"\"
"""

# --- Utility Functions ---
def clean_json(raw: str) -> str:
    """Removes markdown backticks and leading/trailing whitespace from a JSON string."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw)
    raw = re.sub(r"```$", "", raw)
    return raw.strip()

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads JSON data from a file, handling file existence and corruption."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Warning: {file_path} is corrupt. Starting fresh.")
                return []
    return []

def save_json_data(data: List[Dict[str, Any]], file_path: str):
    """Saves JSON data to a file with indentation."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

async def call_with_retry(prompt: str, retries: int = 3):
    """Invokes the LLM with retry logic and exponential backoff."""
    for i in range(retries):
        try:
            resp = await llm.ainvoke(prompt)
            return resp
        except Exception as e:
            wait = (2 ** i) + random.random()
            print(f"⚠️ Error: {e}, retrying in {wait:.1f}s...")
            await asyncio.sleep(wait)
    raise RuntimeError("Failed after retries")

# --- Main Processing Logic ---
async def process_all_sequentially(limit: Optional[int] = None, cooldown: int = 8):
    """Main function to orchestrate the sequential processing of all articles."""
    articles = load_json_data(INPUT_FILE)
    if limit:
        articles = articles[:limit]

    results = load_json_data(OUTPUT_FILE)
    processed_ids = {node["nodes"][0]["id"] for node in results if "nodes" in node and node["nodes"]}

    print(f"▶️ Resuming: {len(processed_ids)} articles already processed.")

    # Iterate through articles sequentially
    for i, article in enumerate(articles, 1):
        article_id = f"Article:{article['article']}"
        if article_id in processed_ids:
            print(f"⏩ Skipping Article {article['article']} ({i}/{len(articles)})")
            continue

        text = f"Article {article['article']}: {article['title']}\n{article['description']}"
        prompt = PROMPT.format(text=text)

        try:
            resp = await call_with_retry(prompt)
            if resp is None or not resp.content.strip():
                raise ValueError("LLM returned empty content.")

            cleaned = clean_json(resp.content)
            data = json.loads(cleaned)
            
            # Save the result immediately
            results.append(data)
            save_json_data(results, OUTPUT_FILE)
            
            print(f"✅ Saved Article {article['article']} ({i}/{len(articles)})")

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error for Article {article['article']}: {e}")
            with open(LOG_FILE, "a") as log_file:
                log_file.write(f"JSON ERROR for Article {article['article']}:\n{e}\nRaw Content:\n{cleaned}\n\n")
            continue
        except Exception as e:
            print(f"⚠️ Skipped Article {article['article']} due to a general error: {e}")
            with open(LOG_FILE, "a") as log_file:
                log_file.write(f"GENERAL ERROR for Article {article['article']}: {e}\n\n")
            continue
        finally:
            # Cooldown between requests
            await asyncio.sleep(cooldown)

    print(f"🎉 Finished. Saved {len(results)} graph fragments to {OUTPUT_FILE}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Cooldown of 8 seconds to stay well below the rate limit
    asyncio.run(process_all_sequentially(limit=None, cooldown=1))
