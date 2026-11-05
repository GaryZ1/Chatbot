"""Command line: ``chatbot ingest | ask | serve | eval``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence

from chatbot.config import Settings
from chatbot.core.types import Principal


def _principal(args: argparse.Namespace) -> Principal:
    return Principal(
        user_id=args.user,
        groups=frozenset(g for g in args.groups.split(",") if g),
        tenant_id=args.tenant,
    )


async def _ingest(args: argparse.Namespace) -> int:
    from chatbot.service import ChatbotService

    svc = ChatbotService.from_settings(Settings.load(args.config))
    try:
        report = await svc.ingestion.ingest_paths(args.paths, full=args.full)
    finally:
        await svc.aclose()
    print(json.dumps(report.__dict__, indent=2, default=str))
    return 1 if report.failed else 0


async def _ask(args: argparse.Namespace) -> int:
    from chatbot.service import ChatbotService

    svc = ChatbotService.from_settings(Settings.load(args.config))
    principal = _principal(args)
    session_id: str | None = None
    try:
        questions = [args.question] if args.question else None
        while True:
            if questions is not None:
                if not questions:
                    break
                q = questions.pop(0)
            else:
                try:
                    q = input("you> ").strip()
                except EOFError:
                    break
                if not q:
                    continue
            answer, session = await svc.rag.chat(q, principal, session_id=session_id)
            session_id = session.session_id
            print(f"bot> {answer.text}\n     [{answer.status.value} via {answer.route}]")
            for c in answer.citations:
                where = f" p.{c.page}" if c.page else ""
                print(f"     [{c.index}] {c.title or c.doc_id}{where}")
    finally:
        await svc.aclose()
    return 0


async def _eval(args: argparse.Namespace) -> int:
    from chatbot.evaluation import run_eval
    from chatbot.service import ChatbotService

    settings = Settings.load(args.config)
    svc = ChatbotService.from_settings(settings)
    try:
        if args.corpus:
            await svc.ingestion.ingest_paths([args.corpus], full=True)
        principal = Principal(
            user_id="eval", groups=frozenset(args.groups.split(",")), tenant_id=args.tenant
        )
        judge_llm = svc.components.llm if args.judges else None
        outcome = await run_eval(
            svc.eval_system(principal),
            args.dataset,
            name=args.name,
            split=args.split,
            allow_test=args.split == "test",
            judge_llm=judge_llm,
            baseline=args.baseline,
            gate=args.gate,
            out_dir=args.out,
        )
    finally:
        await svc.aclose()
    print(f"report: {outcome.report_path}")
    if outcome.gate is not None:
        print("gate: " + ("PASS" if outcome.gate.passed else "FAIL"))
        for r in outcome.gate.reasons:
            print(f"  - {r}")
    return outcome.exit_code


def _serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    if args.config:
        os.environ["CHATBOT_CONFIG"] = args.config
    uvicorn.run(
        "chatbot.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chatbot")
    parser.add_argument("--config", default=None, help="YAML config (default: $CHATBOT_CONFIG)")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_identity(p: argparse.ArgumentParser) -> None:
        p.add_argument("--user", default="cli")
        p.add_argument("--groups", default="employees")
        p.add_argument("--tenant", default="default")

    p_ing = sub.add_parser("ingest", help="index one or more directories")
    p_ing.add_argument("paths", nargs="+")
    p_ing.add_argument("--full", action="store_true", help="re-process unchanged files too")

    p_ask = sub.add_parser("ask", help="ask a question (interactive without --question)")
    p_ask.add_argument("--question", "-q")
    add_identity(p_ask)

    p_srv = sub.add_parser("serve", help="run the HTTP API")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--workers", type=int, default=1)

    p_ev = sub.add_parser("eval", help="run the offline evaluation (+ optional CI gate)")
    p_ev.add_argument("--dataset", nargs="+", required=True)
    p_ev.add_argument(
        "--corpus", default="data/sample_corpus", help="ingest this first ('' to skip)"
    )
    p_ev.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p_ev.add_argument("--baseline", default=None, help="previous results JSON")
    p_ev.add_argument("--gate", default=None, help="gate YAML")
    p_ev.add_argument("--judges", action="store_true", help="use LLM judges (costs tokens)")
    p_ev.add_argument("--name", default="candidate")
    p_ev.add_argument("--out", default="eval/reports")
    add_identity(p_ev)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("chatbot.trace").setLevel(logging.WARNING)
    if args.cmd == "serve":
        return _serve(args)
    handler = {"ingest": _ingest, "ask": _ask, "eval": _eval}[args.cmd]
    return asyncio.run(handler(args))


if __name__ == "__main__":
    sys.exit(main())
