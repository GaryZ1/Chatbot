"""Structured RAG / text-to-SQL: for aggregate or analytical questions ("highest quarter",
"average revenue"), retrieval over text is the wrong tool. Give the LLM the schema (plus
column descriptions and examples), generate SQL, VALIDATE it (read-only, allow-listed
tables, row limit), execute, and answer from the result table. Hybrid systems route between
this and text RAG, or run both."""

import re
import sqlite3

from rag_kit import REVENUE_ROWS, get_llm, show

SCHEMA = """CREATE TABLE revenue (
  quarter TEXT,          -- e.g. '2024-Q3'
  revenue_m_eur REAL,    -- revenue in million euros
  operators INTEGER      -- number of operators using Skyline
);"""
PROMPT = """<schema>
{schema}
</schema>
<question>{q}</question>
Write one SQLite SELECT statement that answers the question. Return only the SQL."""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(SCHEMA)
    conn.executemany("INSERT INTO revenue VALUES (?, ?, ?)", REVENUE_ROWS)

    def authorizer(action, arg1, arg2, db, trigger):          # read-only, one table
        if action == sqlite3.SQLITE_SELECT or (action == sqlite3.SQLITE_READ and
                                               arg1 == "revenue"):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    conn.set_authorizer(authorizer)
    return conn


def validate(sql: str, max_rows: int = 100) -> str:
    sql = re.sub(r"^```(?:sql)?|```$", "", sql.strip(), flags=re.M).strip().rstrip(";")
    if ";" in sql or not re.match(r"(?is)^\s*(select|with)\b", sql):
        raise ValueError("only a single SELECT statement is allowed")
    if not re.search(r"(?i)\blimit\s+\d+\s*$", sql):
        sql += f" LIMIT {max_rows}"
    return sql


def text_to_sql(question: str) -> dict:
    llm, conn = get_llm(), connect()
    sql = validate(llm(PROMPT.format(schema=SCHEMA, q=question), task="sql"))
    try:
        cur = conn.execute(sql)
    except sqlite3.DatabaseError as exc:                       # feed the error back once
        sql = validate(llm(PROMPT.format(schema=SCHEMA, q=question) +
                           f"\nThe previous SQL failed with: {exc}", task="sql"))
        cur = conn.execute(sql)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    table = "\n".join([" | ".join(cols)] + [" | ".join(map(str, r)) for r in rows])
    answer = llm(f'<context><source id="sql">{table}</source></context>\n'
                 f"<question>{question}</question>", task="answer")
    return {"sql": sql, "rows": rows, "answer": answer}


def demo() -> None:
    show("Text-to-SQL", text_to_sql("Which quarter had the highest revenue?"))
    try:
        validate("DROP TABLE revenue")
    except ValueError as exc:
        show("Validator", f"blocked: {exc}")


if __name__ == "__main__":
    demo()
