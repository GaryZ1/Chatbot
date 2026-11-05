from chatbot.guard.pii import PIIDetector, PIIType, iban_valid, luhn_valid, restore, ssn_valid


def test_luhn():
    assert luhn_valid("4111111111111111")
    assert luhn_valid("4111 1111 1111 1111")
    assert luhn_valid("378282246310005")  # Amex
    assert not luhn_valid("4111111111111112")
    assert not luhn_valid("1")


def test_iban():
    assert iban_valid("DE89 3704 0044 0532 0130 00")
    assert iban_valid("GB82WEST12345698765432")
    assert iban_valid("FR1420041010050500013M02606")
    assert not iban_valid("DE89 3704 0044 0532 0130 01")  # bad check digits
    assert not iban_valid("DE89370400440532013")  # wrong length for DE
    assert not iban_valid("hello")


def test_ssn():
    assert ssn_valid("123-45-6789")
    assert not ssn_valid("000-12-3456")
    assert not ssn_valid("666-12-3456")
    assert not ssn_valid("923-12-3456")
    assert not ssn_valid("123-00-6789")


def test_redaction_placeholders_and_consistency():
    d = PIIDetector()
    text = (
        "Contact jane@acme.io or jane@acme.io, card 4111 1111 1111 1111, "
        "other card 4111 1111 1111 1112, IBAN DE89 3704 0044 0532 0130 00, "
        "SSN 123-45-6789, host 10.0.0.12, bob@x.org"
    )
    r = d.redact(text)
    assert "[EMAIL_1]" in r.text and "[EMAIL_2]" in r.text
    assert r.text.count("[EMAIL_1]") == 2  # same value -> same placeholder
    assert "[CREDIT_CARD_1]" in r.text
    assert "4111 1111 1111 1112" in r.text  # fails Luhn -> not a card
    assert "[IBAN_1]" in r.text and "[SSN_1]" in r.text and "[IP_ADDRESS_1]" in r.text
    assert "jane@acme.io" not in r.text
    assert r.mapping == {}  # not reversible by default


def test_reversible_mapping_roundtrip():
    d = PIIDetector()
    text = "Email me at a.b@example.com, phone +44 20 7946 0958."
    r = d.redact(text, reversible=True)
    assert r.mapping["[EMAIL_1]"] == "a.b@example.com"
    assert restore(r.text, r.mapping) == text


def test_secrets_detected():
    d = PIIDetector()
    key_block = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\nabc\n-----END RSA PRIVATE KEY-----"
    text = (
        "aws AKIAIOSFODNN7EXAMPLE secret aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY "
        "openai sk-proj-abcdefghijklmnopqrstuvwx github ghp_"
        + "a" * 36
        + "\n"
        + key_block
        + "\npassword: hunter2hunter2"
    )
    r = d.redact(text)
    types = r.types
    assert {
        PIIType.AWS_ACCESS_KEY,
        PIIType.AWS_SECRET_KEY,
        PIIType.API_KEY,
        PIIType.PRIVATE_KEY,
        PIIType.SECRET,
    } <= types
    assert r.has_secrets
    assert "BEGIN RSA PRIVATE KEY" not in r.text
    assert "hunter2hunter2" not in r.text


def test_no_false_positives_on_plain_text():
    d = PIIDetector()
    text = "Version 1.2.3 released on 2024-01-15 at 12:30:45; see section 4.2 and step 10."
    assert d.detect(text) == []


def test_type_filter():
    d = PIIDetector(types=[PIIType.EMAIL])
    r = d.redact("x@y.com 4111 1111 1111 1111")
    assert r.text.startswith("[EMAIL_1]") and "4111" in r.text
