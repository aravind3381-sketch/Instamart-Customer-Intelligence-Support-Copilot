# src/part_c_triage.py
import re
import json
import time
import random
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mock_llm

from mock_llm import (
    chat,
    est_tokens,
    reset
)


def redact_ticket(text):
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", text)
    text = re.sub(r"\b[6-9]\d{9}\b", "[PHONE]", text)
    text = re.sub(r"ORD-\d{6}", "[ORDER_ID]", text)
    return text

def parse_llm_response(content):
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("category") in mock_llm.CATS:
            return "classified_json", data["category"]
    except Exception:
        pass
    
    # Regex fallback to lowercase words
    words = set(re.findall(r"[a-z]+", content.lower()))
    matches = [c for c in mock_llm.CATS if c in words]
    if len(matches) == 1:
        return "classified_keyword", matches[0]
        
    return "flagged_invalid_output", None

def execute_with_retry(messages, ticket_id, base_delay=0.05, max_retries=3):
    for attempt in range(max_retries + 1):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt) * random.uniform(0.9, 1.15)
            # Log format: ticket id, retry number, wait
            print(f"[RETRY] Ticket={ticket_id} Attempt={attempt} Delay={delay:.4f}s")
            time.sleep(delay)

def run_triage_pipeline(few_shot=True, token_budget=100000):
    mock_llm.reset()
    tickets_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "tickets.csv"), keep_default_na=False)
    segments_df = pd.read_csv(os.path.join(PROJECT_ROOT, "outputs", "customer_segments.csv"))
    premium_lookup = dict(zip(segments_df["customer_id"], segments_df["is_premium"]))
    segment_lookup = dict(zip(segments_df["customer_id"], segments_df["segment_name"]))
    
    system_prompt = (
        "CLASSIFY support tickets into billing, delivery, technical, or fraud. "
        "Reply with JSON: {\"category\": \"...\", \"reason\": \"...\"}\n"
    )
    if few_shot:
        system_prompt += (
            "EXAMPLES:\n"
            "Ticket: Unauthorized charge on card.\n"
            "{\"category\": \"fraud\", \"reason\": \"unauthorized card activity\"}\n"
            "Ticket: The rider has not arrived yet.\n"
            "{\"category\": \"delivery\", \"reason\": \"order delivery inquiry\"}\n"
        )
        
    results = []
    running_tokens = 0
    
    for _, row in tickets_df.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        txt = row["text"]
        
        # 1. Input Guardrails
        if not txt.strip():
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "skipped_empty",
                "category": "", "segment_name": segment_lookup.get(cid, ""),
                "priority": "NORMAL", "route": "human_review", "tokens_used": 0
            })
            continue
            
        if mock_llm.est_tokens(txt) > 400:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "flagged_long",
                "category": "", "segment_name": segment_lookup.get(cid, ""),
                "priority": "NORMAL", "route": "human_review", "tokens_used": 0
            })
            continue
            
        # 2. Token Budget Check
        if running_tokens >= token_budget:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "budget_exhausted",
                "category": "", "segment_name": segment_lookup.get(cid, ""),
                "priority": "NORMAL", "route": "human_review", "tokens_used": 0
            })
            continue
            
        # 3. Redact
        clean_txt = redact_ticket(txt)
        assert not re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", clean_txt)
        assert not re.findall(r"\b[6-9]\d{9}\b", clean_txt)
        assert not re.findall(r"ORD-\d{6}", clean_txt)
        
        # 4. Model Call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": clean_txt}
        ]
        resp = execute_with_retry(messages, tid)
        used = resp.usage.total_tokens
        running_tokens += used
        
        # 5. Output Validation
        status, category = parse_llm_response(resp.choices[0].message.content)
        
        # 6. Routing
        is_prem = premium_lookup.get(cid, False)
        if status == "flagged_invalid_output":
            route, priority = "human_review", "NORMAL"
        elif category == "fraud":
            route, priority = "security_team", "HIGH"
        elif is_prem:
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"
            
        results.append({
            "ticket_id": tid, "customer_id": cid, "status": status,
            "category": category if category else "",
            "segment_name": segment_lookup.get(cid, ""),
            "priority": priority, "route": route, "tokens_used": used
        })
        
    return pd.DataFrame(results)

# Standard triage run
df_triage = run_triage_pipeline(few_shot=True, token_budget=100000)
df_triage.to_csv(os.path.join(PROJECT_ROOT, "outputs", "triage_results.csv"), index=False)
    
# Budget demo run
df_budget = run_triage_pipeline(few_shot=False, token_budget=6000)
df_budget.to_csv(os.path.join(PROJECT_ROOT, "outputs", "triage_budget_demo.csv"), index=False)