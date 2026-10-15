"""Shared SSH config for the GPU server — no credentials in the repo.

Connection parameters come from environment variables (never hardcoded;
a previous credential leak made this mandatory):

  BITAHUB_HOST   default: xj-member.bitahub.com
  BITAHUB_PORT   default: 42220
  BITAHUB_USER   default: root
  BITAHUB_PW     password (or use BITAHUB_KEY_FILE for key auth)
  BITAHUB_KEY_FILE        path to a private key
  BITAHUB_KNOWN_HOSTS     path to known_hosts (default: ~/.ssh/known_hosts)

Host key policy: the server address changes its host key across re-provisions
of the same port. We therefore load known_hosts when present, and pin the
FIRST key seen per (host, port) in a local fingerprint cache file
(.ssh_known_fingerprints, untracked). A later mismatch aborts the
connection — that is a MITM or a re-provisioned host, and either way needs
an explicit human decision (delete the cached line to re-pin). This replaces
AutoAddPolicy, which trusted every unknown key forever.
"""
import hashlib
import os
from pathlib import Path

import paramiko

HOST = os.environ.get("BITAHUB_HOST", "xj-member.bitahub.com")
PORT = int(os.environ.get("BITAHUB_PORT", "42220"))
USER = os.environ.get("BITAHUB_USER", "root")
PW = os.environ.get("BITAHUB_PW")                     # optional (key auth otherwise)
KEY_FILE = os.environ.get("BITAHUB_KEY_FILE")         # optional
KNOWN_HOSTS = os.environ.get(
    "BITAHUB_KNOWN_HOSTS",
    os.path.join(os.path.expanduser("~"), ".ssh", "known_hosts"))

_PIN_FILE = Path(__file__).resolve().parent.parent / ".ssh_known_fingerprints"


def _pinned() -> dict:
    pins = {}
    if _PIN_FILE.exists():
        for line in _PIN_FILE.read_text().splitlines():
            if line.strip():
                pins[line.strip()] = True
    return pins


def _pin(key):
    h = hashlib.sha256(key.asbytes()).hexdigest()
    tag = f"{HOST}:{PORT}:{h}"
    _PIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_PIN_FILE, "a") as f:
        f.write(tag + "\n")
    return tag


class PinnedKeyPolicy(paramiko.MissingHostKeyPolicy):
    """Accept an unknown key only the first time; abort on any change."""

    def missing_host_key(self, client, hostname, key):
        h = hashlib.sha256(key.asbytes()).hexdigest()
        tag = f"{HOST}:{PORT}:{h}"
        if tag in _pinned():
            return  # already trusted
        if not _pinned():
            _pin(key)  # first contact: pin and continue
            return
        raise paramiko.SSHException(
            f"host key for {HOST}:{PORT} changed or is unknown "
            f"(pinned fingerprints exist); refusing to connect. "
            f"If the server was re-provisioned, remove the stale line "
            f"in {_PIN_FILE}.")


def connect(max_attempts=6):
    last = None
    for i in range(max_attempts):
        try:
            ssh = paramiko.SSHClient()
            if os.path.exists(KNOWN_HOSTS):
                ssh.load_system_host_keys(KNOWN_HOSTS)
            ssh.set_missing_host_key_policy(PinnedKeyPolicy())
            kwargs = dict(port=PORT, username=USER, timeout=25,
                          banner_timeout=45, auth_timeout=25)
            if KEY_FILE:
                kwargs["key_filename"] = KEY_FILE
            elif PW:
                kwargs["password"] = PW
            else:
                raise SystemExit(
                    "no credentials: set BITAHUB_PW or BITAHUB_KEY_FILE")
            ssh.connect(HOST, **kwargs)
            return ssh
        except SystemExit:
            raise
        except Exception as e:
            last = e
            time.sleep(4 + 3 * i)
    raise SystemExit(f"SSH failed after {max_attempts} attempts: {last}")


import time  # noqa: E402  (used in connect retry loop)
