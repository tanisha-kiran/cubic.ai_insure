"""
Authentication for the offline demo -- standard library only (no external identity provider).

- Passwords: salted PBKDF2-HMAC-SHA256 (240k iterations), stored in data/users.json.
- Sessions: a signed token in an HttpOnly cookie ("quietly_session"), HMAC-SHA256 over
  {user, expiry} with a random key kept in data/secret.key (created on first run).
- Login throttling: 5 wrong passwords from one address locks it out for 60 seconds.

Manage users with:  python manage_users.py add|passwd|remove|list
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time

DATA_DIR = os.environ.get("QUIETLY_DATA", "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
KEY_FILE = os.path.join(DATA_DIR, "secret.key")
FIRST_RUN_FILE = os.path.join(DATA_DIR, "first_run_credentials.txt")

COOKIE_NAME = "quietly_session"
SESSION_SECONDS = 12 * 3600
PBKDF2_ROUNDS = 240_000
MAX_FAILS, LOCKOUT_SECONDS = 5, 60

_lock = threading.Lock()
_fails = {}   # client address and normalized account -> (count, locked_until)
_revoked_tokens = set()
_revoked_lock = threading.Lock()


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)


def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: dict):
    _ensure_dir()
    tmp = USERS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)
    os.replace(tmp, USERS_FILE)


def valid_username(name: str) -> bool:
    return bool(name) and len(name) <= 32 and all(c.isalnum() or c in "_-." for c in name)


def create_user(name: str, password: str, admin: bool = False):
    if not valid_username(name):
        raise ValueError("username: letters, digits, _ - . only (max 32)")
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    with _lock:
        users = _load_users()
        salt = secrets.token_bytes(16)
        users[name] = {"salt": salt.hex(), "hash": _hash(password, salt).hex(),
                       "admin": bool(admin), "created": int(time.time())}
        _save_users(users)


def remove_user(name: str) -> bool:
    with _lock:
        users = _load_users()
        found = users.pop(name, None) is not None
        _save_users(users)
    return found


def list_users() -> list:
    return [{"name": n, "admin": u.get("admin", False)} for n, u in sorted(_load_users().items())]


def is_admin(name: str) -> bool:
    return bool(_load_users().get(name, {}).get("admin"))


def ensure_first_user() -> str | None:
    """
    First run: create an admin account so the site is never open. Credentials come from
    QUIETLY_ADMIN_USER / QUIETLY_ADMIN_PASSWORD if set, otherwise a random password is generated and
    written to data/first_run_credentials.txt (delete that file after signing in).
    Returns a message to print, or None if users already exist.
    """
    if _load_users():
        return None
    user = os.environ.get("QUIETLY_ADMIN_USER", "admin")
    password = os.environ.get("QUIETLY_ADMIN_PASSWORD")
    generated = password is None
    password = password or secrets.token_urlsafe(9)
    create_user(user, password, admin=True)
    if generated:
        with open(FIRST_RUN_FILE, "w", encoding="utf-8") as f:
            f.write(f"username: {user}\npassword: {password}\n")
        return f"First run: created admin '{user}' with password '{password}' (also saved to {FIRST_RUN_FILE})"
    return f"First run: created admin '{user}' from QUIETLY_ADMIN_PASSWORD"


def check_login(name: str, password: str, client: str):
    """Rate limit by both client address and normalized account name."""
    now = time.time()
    name = (name or "").strip()
    keys = [client]
    if name:
        keys.append("account:" + name.casefold())
    with _lock:
        until = max((_fails.get(key, (0, 0))[1] for key in keys), default=0)
        if until > now:
            return False, f"Too many attempts. Try again in {int(until - now) + 1} seconds."
    user = _load_users().get(name)
    salt = bytes.fromhex(user["salt"]) if user else b"\x00" * 16
    candidate = _hash(password or "", salt)
    ok = bool(user) and hmac.compare_digest(candidate.hex(), user["hash"])
    with _lock:
        if ok:
            for key in keys:
                _fails.pop(key, None)
        else:
            for key in keys:
                count, _ = _fails.get(key, (0, 0))
                count += 1
                _fails[key] = (count, now + LOCKOUT_SECONDS if count >= MAX_FAILS else 0)
    return (True, "ok") if ok else (False, "Wrong username or password.")


def revoke_token(token: str | None) -> None:
    """Invalidate a token immediately for the lifetime of this server process."""
    if token:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with _revoked_lock:
            _revoked_tokens.add(digest)


def _secret() -> bytes:
    _ensure_dir()
    if not os.path.exists(KEY_FILE):
        with open(KEY_FILE, "wb") as f:
            f.write(secrets.token_bytes(32))
    with open(KEY_FILE, "rb") as f:
        return f.read()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_token(user: str, ttl: int = SESSION_SECONDS) -> str:
    payload = _b64(json.dumps({"u": user, "exp": int(time.time()) + ttl}).encode())
    sig = _b64(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def read_token(token: str | None) -> str | None:
    """The username in a valid, unexpired token, else None."""
    if not token or "." not in token:
        return None
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with _revoked_lock:
        if digest in _revoked_tokens:
            return None
    payload, sig = token.rsplit(".", 1)
    expected = _b64(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        data = json.loads(_unb64(payload))
    except (ValueError, json.JSONDecodeError):
        return None
    if data.get("exp", 0) < time.time() or data.get("u") not in _load_users():
        return None
    return data["u"]

