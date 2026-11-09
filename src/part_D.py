import os
import glob
import json
import sys
import time
import sqlite3
import pandas as pd
import mock_llm
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

POLICIES_DIR = os.path.join("data", "policies")
STATE_DIR = "state"
LEDGER_PATH = os.path.join("outputs", "refund_ledger.jsonl")
RUNS_PATH = os.path.join("outputs", "agent_runs.jsonl")
DB_PATH = os.path.join("data", "instamart.db")

os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# =====================================================================
# 1. Chunking & Policy RAG (Tasks 1, 2 & 3)
# =====================================================================

def chunk_policies(chunk_size=40, overlap=10):
    chunks = []
    step = max(1, chunk_size - overlap)
    for path in sorted(glob.glob(os.path.join(POLICIES_DIR, "*.txt"))):
        fname = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            words = f.read().split()
        for idx in range(0, len(words), step):
            chunk_words = words[idx:idx + chunk_size]
            chunks.append({"source": fname, "text": " ".join(chunk_words)})
            if idx + chunk_size >= len(words):
                break
    return chunks


class PolicyRAG:
    def __init__(self, chunk_size=40, overlap=10):
        self.chunks = chunk_policies(chunk_size, overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:k]
        return [
            {"source": self.chunks[i]["source"], "score": float(scores[i]), "text": self.chunks[i]["text"]}
            for i in top_indices
        ]

    def answer(self, question):
        docs = self.retrieve(question, k=3)
        context = "\n".join(f"[{d['source']}]\n{d['text']}" for d in docs)
        messages = [
            {"role": "system", "content": "POLICY ASSISTANT. Answer strictly from CONTEXT. If missing, say 'I could not find that information in the policy documents.'"},
            {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}
        ]
        resp = mock_llm.chat(messages)
        return resp.choices[0].message.content, [d["source"] for d in docs]


# =====================================================================
# 2. Tools & Specs (Task 5)
# =====================================================================

rag_engine = PolicyRAG(chunk_size=40, overlap=10)

def get_order(order_id):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
            (order_id,)
        ).fetchone()
    if row:
        return {"order_id": row[0], "status": row[1], "order_amount": float(row[2]), "delivery_minutes": row[3]}
    return {"error": "order not found"}

def search_policy(query):
    return {"results": rag_engine.retrieve(query, k=3)}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": float(order_amount)}
    return {"refund_inr": 0.0}

def _log_action(action, order_id, refund_inr):
    entry = {"timestamp": datetime.utcnow().isoformat(), "action": action, "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):
    return _log_action("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return _log_action("escalate_to_human", order_id, refund_inr)

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

TOOLS_SPEC = [
    {"type": "function", "function": {"name": "get_order", "description": "Lookup order in analytical DB.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "search_policy", "description": "Search operational policies.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "calculate_refund", "description": "Calculate refund if delivery exceeds 45 min.", "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": ["integer", "null"]}}, "required": ["order_amount", "delivery_minutes"]}}},
    {"type": "function", "function": {"name": "issue_refund", "description": "Issue refund <= INR 1000.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}}},
    {"type": "function", "function": {"name": "escalate_to_human", "description": "Escalate refund > INR 1000.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}}}
]


# =====================================================================
# 3. Unified ReAct Agent Engine (Tasks 6, 7 & 8)
# =====================================================================

def execute_agent_loop(state, max_iterations=6, token_limit=6000, time_limit_seconds=10, crash_after_tool=None):
    """Core execution loop shared by both run_agent and resume_agent."""
    start_time = time.monotonic()
    run_id = state["run_id"]
    scenario_id = state["scenario_id"]
    messages = state["messages"]
    steps = state["steps"]
    executed_tools = state["executed_tools"]
    tokens_used = state["tokens_used"]
    completed_calls = len([m for m in messages if m.get("role") == "assistant"])

    for i in range(completed_calls, max_iterations):
        # 1. Timeout Guard
        if (time.monotonic() - start_time) >= time_limit_seconds:
            return {"scenario_id": scenario_id, "status": "FALLBACK_TIMEOUT", "final_answer": "Timeout reached.", "llm_calls": i, "tokens_used": tokens_used, "steps": steps, "executed_tools": executed_tools}

        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        tokens_used += resp.usage.total_tokens
        msg = resp.choices[0].message

        # 2. Token Limit Guard
        if tokens_used > token_limit:
            return {"scenario_id": scenario_id, "status": "FALLBACK_TOKEN_LIMIT", "final_answer": "Token limit exceeded.", "llm_calls": i + 1, "tokens_used": tokens_used, "steps": steps, "executed_tools": executed_tools}

        # 3. Final Response Check
        if not msg.tool_calls:
            status = msg.content.split(":")[0].strip() if ":" in msg.content else "RESOLVED"
            return {"scenario_id": scenario_id, "status": status, "final_answer": msg.content, "llm_calls": i + 1, "tokens_used": tokens_used, "steps": steps, "executed_tools": executed_tools}

        # 4. Tool Execution
        tc = msg.tool_calls[0]
        tool_name = tc.name
        tool_args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)

        messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "name": tool_name, "arguments": tool_args}]})
        obs = TOOL_MAP[tool_name](**tool_args)
        executed_tools.append(tool_name)
        steps.append({"step": i + 1, "tool": tool_name, "arguments": tool_args, "observation": obs})
        messages.append({"role": "tool", "tool_call_id": tc.id, "name": tool_name, "content": json.dumps(obs)})

        # Save checkpoint
        state_data = {"run_id": run_id, "scenario_id": scenario_id, "messages": messages, "tokens_used": tokens_used, "steps": steps, "executed_tools": executed_tools}
        with open(os.path.join(STATE_DIR, f"{run_id}.json"), "w", encoding="utf-8") as f:
            json.dump(state_data, f)

        # Simulated Crash Injection
        if crash_after_tool and len(executed_tools) == crash_after_tool:
            raise RuntimeError(f"Simulated crash after tool: {tool_name}")

    return {"scenario_id": scenario_id, "status": "FALLBACK_MAX_ITERATIONS", "final_answer": "Loop limit reached.", "llm_calls": max_iterations, "tokens_used": tokens_used, "steps": steps, "executed_tools": executed_tools}


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, run_id=None, crash_after_tool=None):
    run_id = run_id or f"run_{scenario_id}_{int(time.time()*1000)}"
    initial_state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": [
            {"role": "system", "content": "AGENT. You resolve customer refund requests using available tools."},
            {"role": "user", "content": user_message}
        ],
        "steps": [],
        "executed_tools": [],
        "tokens_used": 0
    }
    return execute_agent_loop(initial_state, max_iterations, token_limit, time_limit_seconds, crash_after_tool)


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    ckpt_path = os.path.join(STATE_DIR, f"{run_id}.json")
    with open(ckpt_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return execute_agent_loop(state, max_iterations, token_limit, time_limit_seconds)


# =====================================================================
# 4. Evaluation & Execution Runner (Tasks 4, 6, 7 & 8)
# =====================================================================

def evaluate_rag():
    eval_df = pd.read_csv("data/raw/policy_eval.csv")
    for name, overlap in [("Overlap=10", 10), ("Overlap=0", 0)]:
        rag = PolicyRAG(40, overlap)
        doc_hits, fact_hits, fallback_hits = 0, 0, 0
        for _, row in eval_df.iterrows():
            ans, sources = rag.answer(row["question"])
            if row["expected_doc"] == "NONE":
                if ans.startswith("I could not find"):
                    fallback_hits += 1
            else:
                if row["expected_doc"] in sources:
                    doc_hits += 1
                if row["expected_fact"] in ans:
                    fact_hits += 1
        print(f"RAG {name:10} -> Doc Hits: {doc_hits}/12 | Fact Hits: {fact_hits}/12 | Fallback: {fallback_hits}/2")


if __name__ == "__main__":
    print("=== TASK 4: Policy RAG Evaluation ===")
    evaluate_rag()

    print("\n=== TASK 6: Executing 7 Scenarios ===")
    scenarios_df = pd.read_csv("data/raw/agent_scenarios.csv")
    if os.path.exists(LEDGER_PATH): os.remove(LEDGER_PATH)
    if os.path.exists(RUNS_PATH): os.remove(RUNS_PATH)

    for _, row in scenarios_df.iterrows():
        res = run_agent(row["message"], row["scenario_id"])
        with open(RUNS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
        print(f"[{res['scenario_id']}] Status: {res['status']:24} | Calls: {res['llm_calls']} | Tokens: {res['tokens_used']} | Answer: {res['final_answer']}")

    print("\n=== TASK 7: Guardrail Demos ===")
    demo_msg = scenarios_df.iloc[0]["message"]
    print("Token Limit Demo Status: ", run_agent(demo_msg, "S1_TOKEN", token_limit=200)["status"])
    print("Timeout Demo Status:     ", run_agent(demo_msg, "S1_TIME", time_limit_seconds=0)["status"])

    print("\n=== TASK 8: Crash-and-Resume Demo ===")
    run_id = "s1_crash_demo"
    try:
        run_agent(demo_msg, "S1", crash_after_tool=2, run_id=run_id)
    except RuntimeError:
        print("Simulated crash caught after tool 2.")

    resumed = resume_agent(run_id)
    print("Resumed Status:          ", resumed["status"])
    print("Executed Tools Sequence: ", " -> ".join(resumed["executed_tools"]))
    print("Final Answer:            ", resumed["final_answer"])