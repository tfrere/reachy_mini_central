"""Unit tests for the HF token resolver dependency in ``app.py``.

The resolver is the first line of defence for authenticated endpoints;
a regression here can silently re-introduce token leaks or accept
malformed Authorization schemes.

No test infrastructure exists on this HF Space yet. Run manually with::

    pip install pytest httpx fastapi
    python -m pytest test_resolver.py -v

Keep these tests here so whoever adds CI can wire them up without
writing the suite from scratch.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app import _resolve_hf_token


def _req(ip: str = "1.2.3.4") -> MagicMock:
    """Build a minimal ``Request``-shaped mock with .client.host."""
    request = MagicMock()
    request.client.host = ip
    return request


def _run(
    authorization: Optional[str] = None,
    token: str = "",
    ip: str = "1.2.3.4",
) -> str:
    """Invoke the resolver synchronously for a given input."""
    import asyncio

    return asyncio.run(_resolve_hf_token(_req(ip), authorization, token))


# ---- Header form (preferred) ----

def test_bearer_header_is_accepted():
    assert _run(authorization="Bearer hf_abc123") == "hf_abc123"


def test_bearer_header_case_insensitive_scheme():
    assert _run(authorization="bearer hf_abc123") == "hf_abc123"
    assert _run(authorization="BEARER hf_abc123") == "hf_abc123"


def test_bearer_header_trims_whitespace():
    assert _run(authorization="Bearer   hf_abc123  ") == "hf_abc123"


def test_bearer_header_empty_token_is_rejected():
    """'Bearer' with nothing after must 401, not return ''."""
    with pytest.raises(HTTPException) as exc:
        _run(authorization="Bearer")
    assert exc.value.status_code == 401


def test_bearer_header_whitespace_only_token_is_rejected():
    with pytest.raises(HTTPException) as exc:
        _run(authorization="Bearer    ")
    assert exc.value.status_code == 401


# ---- Unknown / malformed Authorization schemes ----

def test_basic_auth_scheme_is_rejected():
    """Basic auth must NOT leak through as a bare-string token."""
    with pytest.raises(HTTPException) as exc:
        _run(authorization="Basic dXNlcjpwdw==")
    assert exc.value.status_code == 401
    assert "Bearer" in exc.value.detail


def test_digest_scheme_is_rejected():
    with pytest.raises(HTTPException) as exc:
        _run(authorization="Digest username=foo")
    assert exc.value.status_code == 401


def test_bare_token_in_header_is_rejected():
    """A raw token with no scheme is not RFC 6750 shaped — reject it."""
    with pytest.raises(HTTPException) as exc:
        _run(authorization="hf_abc123")
    assert exc.value.status_code == 401


# ---- Query-string fallback ----

def test_query_token_is_accepted_and_deprecation_is_logged_once(caplog):
    # Clear the sampler state between tests — the module-level set
    # persists within a process.
    from app import _deprecation_warned_ips

    _deprecation_warned_ips.clear()
    with caplog.at_level("WARNING"):
        assert _run(token="hf_legacy") == "hf_legacy"
    warnings = [r for r in caplog.records if "deprecation" in r.message]
    assert len(warnings) == 1

    # Second call from the same IP does NOT re-log.
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert _run(token="hf_legacy") == "hf_legacy"
    warnings = [r for r in caplog.records if "deprecation" in r.message]
    assert len(warnings) == 0

    # But a different IP triggers its own one-shot warning.
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert _run(token="hf_legacy", ip="9.9.9.9") == "hf_legacy"
    warnings = [r for r in caplog.records if "deprecation" in r.message]
    assert len(warnings) == 1


# ---- Precedence (both forms present) ----

def test_header_takes_precedence_over_query():
    """If both are sent, the Authorization header wins and the query
    is ignored without logging deprecation (header users are compliant)."""
    assert _run(authorization="Bearer from_header", token="from_query") == "from_header"


# ---- Neither form present ----

def test_missing_both_raises_401():
    with pytest.raises(HTTPException) as exc:
        _run()
    assert exc.value.status_code == 401
    assert "Missing" in exc.value.detail
