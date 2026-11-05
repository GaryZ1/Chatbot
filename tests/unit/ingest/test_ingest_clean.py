"""Cleaning: unicode, line repair, page furniture, whitespace, language, email."""

from __future__ import annotations

from chatbot.core.types import Block, BlockType, Document
from chatbot.ingest import (
    clean_document,
    clean_email,
    dehyphenate,
    detect_language,
    merge_broken_lines,
    normalize_unicode,
    normalize_whitespace,
    remove_repeated_lines,
)


def test_normalize_unicode() -> None:
    assert normalize_unicode("ﬁle ＡＢＣ​x­z\r\n") == "file ABCxz\n"


def test_dehyphenate() -> None:
    assert dehyphenate("infor-\nmation retrieval") == "information retrieval"
    assert dehyphenate("Jean-\nPierre") == "Jean-\nPierre"


def test_merge_broken_lines_keeps_markdown() -> None:
    text = (
        "This line is wrapped\nin the middle.\nNew sentence starts.\n"
        "# Heading\n- item one\n- item two\n| a | b |\n| 1 | 2 |\n```\ncode line\nmore code\n```"
    )
    out = merge_broken_lines(text).split("\n")
    assert out[0] == "This line is wrapped in the middle."
    assert out[1] == "New sentence starts."
    assert out[2:] == [
        "# Heading",
        "- item one",
        "- item two",
        "| a | b |",
        "| 1 | 2 |",
        "```",
        "code line",
        "more code",
        "```",
    ]


def test_remove_repeated_lines() -> None:
    pages = [f"ACME Corp Annual Report\nBody text {i} unique.\nPage {i}" for i in range(1, 6)]
    out = remove_repeated_lines(pages)
    assert all("ACME" not in p and "Page" not in p for p in out)
    assert out[2] == "Body text 3 unique."
    assert remove_repeated_lines(pages[:2]) == pages[:2]  # too few pages to judge


def test_normalize_whitespace_preserves_markdown() -> None:
    text = "a    b  \n\n\n\n  - nested   item\n```\nx    =   1\n```\n"
    assert normalize_whitespace(text) == "a b\n\n  - nested item\n```\nx    =   1\n```"


def test_detect_language() -> None:
    assert detect_language("The policy is that all of the payments are retried for you.")[0] == "en"
    assert detect_language("Die Zahlung wird nicht automatisch wiederholt, wenn der Kunde das ablehnt.")[0] == "de"
    assert detect_language("Le paiement est effectué dans les trois jours pour les clients qui sont en France.")[0] == "fr"
    assert detect_language("El pago se realiza en tres días para los clientes que no están en la ciudad.")[0] == "es"
    assert detect_language("これは日本語の文章です。支払いは三回再試行されます。")[0] == "ja"
    assert detect_language("Платёж будет повторён три раза для всех клиентов.")[0] == "ru"
    assert detect_language("xyz qrs")[0] == "und"


def test_clean_email() -> None:
    body = (
        "Hello,\n\nThe printer on floor 3 is jammed again.\n\nThanks,\nBob Smith\nIT Ops\n+1 555 0100\n\n"
        "-----Original Message-----\nFrom: Carol\nSent: Monday\nSubject: printer\n\nOld message"
    )
    out = clean_email(body)
    assert out == "Hello,\n\nThe printer on floor 3 is jammed again."
    assert clean_email("Fix applied.\n-- \nJane\nSent from my iPhone") == "Fix applied."
    assert clean_email("Answer here\n> quoted line\n> another") == "Answer here"


def test_clean_document_furniture_and_language() -> None:
    blocks = []
    for p in range(1, 5):
        blocks.append(Block(type=BlockType.PARAGRAPH, text="Contoso Confidential", page=p))
        blocks.append(
            Block(type=BlockType.PARAGRAPH, text=f"The quarterly results for the region {p} were strong-\nly positive.", page=p)
        )
        blocks.append(Block(type=BlockType.PARAGRAPH, text=f"- {p} -", page=p))
    blocks.append(Block(type=BlockType.CODE, text="x  =  1\n\n\n  y", page=4))
    doc = Document(doc_id="d", title="  Report​ ", blocks=blocks, language="und")
    out = clean_document(doc)
    assert "Contoso Confidential" not in out.text
    assert "- 2 -" not in out.text
    assert "strongly positive" in out.text
    assert out.blocks[-1].text == "x  =  1\n\n\n  y"  # code untouched
    assert out.language == "en"
    assert out.title == "Report"
    assert doc.blocks[0].text == "Contoso Confidential"  # input not mutated
