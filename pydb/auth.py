"""
Authentication
==============

Manages user accounts and password verification for the PyDB wire
protocol.  Passwords are hashed with PBKDF2-HMAC-SHA256 and a random
salt before storage.

Storage
-------
User records are persisted as ``users.json`` in the database directory,
kept separate from ``catalog.json`` so that authentication metadata and
schema metadata evolve independently.

On first startup (no ``users.json`` exists and no users are defined),
``ensure_default_admin()`` creates a default ``admin`` / ``admin``
account and prints a warning.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from pydb import AUTH_PBKDF2_ITERATIONS, AUTH_SALT_BYTES

def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """Hash a password with PBKDF2-HMAC-SHA256.

    Parameters
    ----------
    password : str
        The plaintext password.
    salt : bytes or None
        Random salt.  Generated automatically if not provided.

    Returns
    -------
    tuple[str, str]
        ``(hex_hash, hex_salt)`` for JSON-safe storage.
    """
    if salt is None:
        salt = os.urandom(AUTH_SALT_BYTES)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, AUTH_PBKDF2_ITERATIONS,
    )
    return dk.hex(), salt.hex()


def verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
    """Verify a password against a stored hash and salt.

    Uses ``hmac.compare_digest`` for timing-safe comparison.
    """
    salt = bytes.fromhex(stored_salt)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, AUTH_PBKDF2_ITERATIONS,
    )
    return hmac.compare_digest(dk.hex(), stored_hash)


@dataclass
class UserDef:
    """A stored user account."""
    username: str
    password_hash: str
    salt: str


class UserStore:
    """Thread-safe persistent store for user accounts.

    Parameters
    ----------
    path : Path
        File path for ``users.json``.
    """

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._users: dict[str, UserDef] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            data = json.loads(self._path.read_text())
            for name, rec in data.get("users", {}).items():
                self._users[name.lower()] = UserDef(
                    username=rec["username"],
                    password_hash=rec["password_hash"],
                    salt=rec["salt"],
                )

    def _save(self):
        data = {
            "users": {
                name: {
                    "username": u.username,
                    "password_hash": u.password_hash,
                    "salt": u.salt,
                }
                for name, u in self._users.items()
            }
        }
        self._path.write_text(json.dumps(data, indent=2))

    def create_user(self, username: str, password: str):
        """Create a new user.  Raises ``ValueError`` if already exists."""
        with self._lock:
            key = username.lower()
            if key in self._users:
                raise ValueError(f"User '{username}' already exists")
            pw_hash, salt = hash_password(password)
            self._users[key] = UserDef(username=username, password_hash=pw_hash, salt=salt)
            self._save()

    def drop_user(self, username: str):
        """Remove a user.  Raises ``KeyError`` if not found."""
        with self._lock:
            key = username.lower()
            if key not in self._users:
                raise KeyError(f"User '{username}' does not exist")
            del self._users[key]
            self._save()

    def alter_password(self, username: str, new_password: str):
        """Change a user's password.  Raises ``KeyError`` if not found."""
        with self._lock:
            key = username.lower()
            if key not in self._users:
                raise KeyError(f"User '{username}' does not exist")
            pw_hash, salt = hash_password(new_password)
            self._users[key].password_hash = pw_hash
            self._users[key].salt = salt
            self._save()

    def authenticate(self, username: str, password: str) -> bool:
        """Return ``True`` if the credentials are valid."""
        with self._lock:
            user = self._users.get(username.lower())
        if user is None:
            return False
        return verify_password(password, user.password_hash, user.salt)

    def user_exists(self, username: str) -> bool:
        with self._lock:
            return username.lower() in self._users

    def ensure_default_admin(self):
        """Create a default ``admin`` / ``admin`` account if no users exist."""
        with self._lock:
            if self._users:
                return
            pw_hash, salt = hash_password("admin")
            self._users["admin"] = UserDef(username="admin", password_hash=pw_hash, salt=salt)
            self._save()
        print("[auth] Created default user 'admin' with password 'admin' — change this!")
