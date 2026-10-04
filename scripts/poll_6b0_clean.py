"""Clean 6B-0 poller: exit only on the PHASE6B0-DONE marker, then fetch."""
import sys, time, json
sys.path.insert(0, 'scripts')
from ssh_fetch import connect, run

deadline = time.time() + 5 * 3600
while time.time() < deadline:
    try:
        ssh = connect()
        out, _ = run(ssh, 'grep -ac PHASE6B0-DONE /tmp/phase6b0.log 2>/dev/null; '
                          'tail -c 200 /tmp/phase6b0.log | tr "\\r" "\\n" | grep -v "^$" | tail -1')
        done = out.splitlines()[0].strip() if out else '0'
        print(f'done_count={done} | {out.splitlines()[-1][:120] if len(out.splitlines())>1 else ""}',
              flush=True)
        if done not in ('', '0'):
            sftp = ssh.open_sftp()
            sftp.chdir('/ccfa2026/compile-anything/results/phase6')
            with sftp.open('phase6b0_routing.json') as f:
                data = f.read()
            sftp.close()
            open('results/phase6/phase6b0_routing.json', 'wb').write(data)
            d = json.loads(data)
            print('\n=== PHASE 6B-0 FOUR-PROTOCOL TABLE ===')
            for k in ('A_bare', 'B_names', 'C_learned_ir', 'D_oracle_ir'):
                print(k, json.dumps(d[k]))
            print('\n6B0-COMPLETE')
            ssh.close()
            sys.exit(0)
        ssh.close()
    except SystemExit:
        raise
    except Exception as e:
        print(f'poll error: {e}', flush=True)
    time.sleep(420)
print('TIMEOUT')
sys.exit(1)
