# app/utils.py  –  Seguridad de contraseñas (PBKDF2-SHA256 con salt)
import hashlib
import os


def make_salt(nbytes: int = 16) -> str:
    return os.urandom(nbytes).hex()


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=260_000,
    ).hex()


def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    return hash_password(password, salt) == stored_hash
