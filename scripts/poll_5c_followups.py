"""Poll server until both Phase 5C follow-up evals finish, then fetch results."""
import sys, time, json

sys.path.insert(0, 'scripts')
from ssh_fetch import connect, run

REMOTE = '/ccfa2026/compile-anything/results/phase5c'
LOCAL = 'results/phase5c'


def fetch(ssh, name):
    sftp = ssh.open_sftp()
    sftp.chdir(REMOTE)
    with sftp.open(name, 'r') as f:
        data = f.read()
    sftp.close()
    open(f'{LOCAL}/{name}', 'wb').write(data)
    return json.loads(data)


deadline = time.time() + 3 * 3600
while time.time() < deadline:
    try:
        ssh = connect()
        out, _ = run(ssh, 'pgrep -f eval_5c_schema[.]py >/dev/null && echo S1=RUNNING || echo S1=DONE; '
                          'pgrep -f eval_5c_tau3[.]py >/dev/null && echo S2=RUNNING || echo S2=DONE; '
                          'pgrep -f eval_5c_bfcl_sem[.]py >/dev/null && echo S3=RUNNING || echo S3=DONE; '
                          'tail -2 /tmp/eval_5c_schema.log /tmp/eval_5c_tau3.log /tmp/eval_5c_bfcl_sem.log 2>/dev/null | grep -v Warning | grep -v ==$')
        status = dict(l.strip().split('=') for l in out.splitlines() if '=' in l and l.startswith('S'))
        print(out.strip(), flush=True)
        if all(status.get(f'S{i}') == 'DONE' for i in (1, 2, 3)):
            for name in ('internal_schema_eval.json', 'tau3_semantic.json', 'bfcl_semantic.json'):
                try:
                    d = fetch(ssh, name)
                    print(f'\n=== {name} ===')
                    print(json.dumps(d, indent=1, ensure_ascii=False))
                except Exception as e:
                    print(f'fetch {name} failed: {e}')
            ssh.close()
            print('\nALL FOLLOW-UPS COMPLETE')
            sys.exit(0)
        ssh.close()
    except SystemExit:
        raise
    except Exception as e:
        print(f'poll error: {e}', flush=True)
    time.sleep(300)

print('TIMEOUT after 3h')
sys.exit(1)
