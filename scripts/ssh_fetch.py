"""Robust SSH fetch/run helper for the BitHub server (banner errors are frequent)."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from scripts.ssh_config import connect  # env-based, host-key pinned


def run(ssh, cmd, timeout=300):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors='replace')
    err = stderr.read().decode(errors='replace')
    return out, err


if __name__ == '__main__':
    remote, local = sys.argv[1], sys.argv[2]
    ssh = connect()
    sftp = ssh.open_sftp()
    sftp.get(remote, local)
    sftp.close()
    ssh.close()
    print(f'fetched {remote} -> {local}')
