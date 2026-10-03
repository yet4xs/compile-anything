"""Robust SSH fetch/run helper for the BitHub server (banner errors are frequent)."""
import sys, time, paramiko

HOST, PORT, USER, PW = 'xj-member.bitahub.com', 42042, 'root', '^^x-CC71i9Ms6CW'


def connect(max_attempts=6):
    last = None
    for i in range(max_attempts):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(HOST, port=PORT, username=USER, password=PW,
                        timeout=25, banner_timeout=45, auth_timeout=25)
            return ssh
        except Exception as e:
            last = e
            time.sleep(4 + 3 * i)
    raise SystemExit(f'SSH failed after {max_attempts} attempts: {last}')


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
