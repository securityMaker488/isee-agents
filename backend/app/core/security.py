from datetime import datetime, timedelta, timezone
import re
from typing import Any, Dict, List, Optional, Tuple
import bcrypt
from jose import JWTError, jwt

from app.core.config import settings


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


def create_access_token(
    subject: str,
    role: str,
    expires_delta: Optional[timedelta] = None,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    now = datetime.now(timezone.utc)
    expire = now + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload: Dict[str, Any] = {
        "sub": subject,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError:
        return None


# Prompt Injection & System Prompt Leak Guardrails
PROMPT_INJECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "ignore_previous_instructions",
        re.compile(
            r"(ignore|disregard|esqueça|ignorar)\s+(all\s+|todas\s+as\s+)?(previous|prior|anteriores|acima)\s+(instructions|prompts|rules|instruções|regras)",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_exfiltration",
        re.compile(
            r"(reveal|show|print|repeat|exiba|mostre|revele)\s+(your\s+|the\s+|o\s+seu\s+)?(system\s+prompt|hidden\s+instructions|prompt\s+de\s+sistema|instruções\s+internas)",
            re.IGNORECASE,
        ),
    ),
    (
        "privilege_escalation_jailbreak",
        re.compile(
            r"(you\s+are\s+now\s+DAN|developer\s+mode\s+enabled|modo\s+desenvolvedor\s+irrestrito|bypass\s+all\s+security\s+constraints)",
            re.IGNORECASE,
        ),
    ),
    (
        "credential_harvesting",
        re.compile(
            r"(dump|expose|export|listar)\s+(all\s+)?(api\s+keys|passwords|secret_key|database_url|credenciais|senhas)",
            re.IGNORECASE,
        ),
    ),
]


def inspect_prompt_security(
    user_input: str, agent_constraints: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """Inspect user prompt for prompt injection or policy violations.
    Returns (is_safe, triggered_flags).
    """
    flags: List[str] = []
    for flag_name, pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(user_input):
            flags.append(flag_name)

    if agent_constraints:
        lowered = user_input.lower()
        for constraint in agent_constraints:
            if constraint.startswith("block_keyword:"):
                kw = constraint.split(":", 1)[1].strip().lower()
                if kw and kw in lowered:
                    flags.append(f"blocked_keyword:{kw}")

    return (len(flags) == 0, flags)


def sanitize_filename(filename: str) -> str:
    """Prevent path traversal and unsafe characters in uploaded filenames."""
    base = os.path.basename(filename or "uploaded_file")
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
    cleaned = cleaned.lstrip(".")
    return cleaned or "document.txt"


import os  # noqa: E402
