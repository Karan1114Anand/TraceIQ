# Autonomous Research Analyst
# ---------------------------------------------------------------------------
# Windows SSL cert store patch — MUST stay at top, before any langchain import.
# Fixes: ssl.SSLError: [ASN1] nested asn1 error
# Root cause: a malformed cert in the Windows store crashes ssl.create_default_context().
# Fix: wrap create_default_context to fall back to certifi if Windows store breaks.
# Also requires: pip install "aiohttp==3.8.6" (aiohttp>=3.9 runs this at import time)
# ---------------------------------------------------------------------------
import ssl as _ssl


def _safe_create_default_context(purpose=_ssl.Purpose.SERVER_AUTH, **kwargs):
    try:
        return _orig_create_default_context(purpose=purpose, **kwargs)
    except _ssl.SSLError:
        try:
            import certifi
            return _orig_create_default_context(purpose=purpose, cafile=certifi.where())
        except Exception:
            ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
            ctx.check_hostname = False
            ctx.verify_mode = _ssl.CERT_NONE
            return ctx


_orig_create_default_context = _ssl.create_default_context
_ssl.create_default_context = _safe_create_default_context

