"""Poll BFCL Track A completion, fetch results."""
import sys, time, json
sys.path.insert(0, 'scripts')
from ssh_fetch import connect, run

deadline = time.time() + 8 * 3600
while time.time() < deadline:
    try:
        ssh = connect()
        out, _ = run(ssh, 'grep -ac BFCL-TRACK-DONE /tmp/bfcl_track.log 2>/dev/null; '
                          'tail -c 200 /tmp/bfcl_track.log | tr "\\r" "\\n" | grep -av "warn\\|^$" | tail -1')
        lines = out.splitlines()
        done = lines[0].strip() if lines else '0'
        print(f'done={done} | {lines[-1][:110] if lines else ""}', flush=True)
        if done not in ('', '0'):
            sftp = ssh.open_sftp()
            sftp.chdir('/ccfa2026/compile-anything/results/bfcl_track')
            with sftp.open('train_sanity.json') as f:
                data = f.read()
            sftp.close()
            open('results/bfcl_track/train_sanity.json', 'wb').write(data)
            d = json.loads(data)
            print('\n=== BFCL TRACK A RESULTS ===')
            for cat, r in sorted(d.items()):
                print(f"  {cat:30s} {r['correct']:5d}/{r['n']:5d} = {r['accuracy']:6.2%}")
            print('\nTRACK-A-COMPLETE')
            ssh.close()
            sys.exit(0)
        ssh.close()
    except SystemExit:
        raise
    except Exception as e:
        print(f'poll error: {e}', flush=True)
    time.sleep(600)
print('TIMEOUT')
sys.exit(1)
