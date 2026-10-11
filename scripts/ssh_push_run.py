"""Upload files to the BitHub server, fix CRLF, run a command.

Usage:
  python ssh_push_run.py --push local:remote [--push ...] -- cmd
"""
import sys, time, argparse, paramiko

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--push', action='append', default=[],
                    help='local:remote pairs')
    ap.add_argument('--timeout', type=int, default=600)
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    args = ap.parse_args()

    ssh = connect()
    if args.push:
        sftp = ssh.open_sftp()
        for pair in args.push:
            local, remote = pair.split(':', 1)
            sftp.put(local, remote)
            print(f'uploaded {local} -> {remote}')
        sftp.close()
        # CRLF fix on every uploaded text file
        targets = ' '.join(p.split(':', 1)[1] for p in args.push)
        _, out, err = ssh.exec_command(f"sed -i 's/\\r$//' {targets}")
        out.channel.recv_exit_status()
        print('CRLF fixed')

    cmd = ' '.join(args.cmd).lstrip('- ')
    if cmd:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=args.timeout)
        for line in iter(stdout.readline, ''):
            print(line, end='')
        err = stderr.read().decode(errors='replace')
        rc = stdout.channel.recv_exit_status()
        if err.strip():
            print(f'--- stderr ---\n{err}', file=sys.stderr)
        print(f'\n[exit {rc}]')
    ssh.close()


if __name__ == '__main__':
    main()
