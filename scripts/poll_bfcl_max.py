"""BFCL max poller: check every 30min, only exit on BFCL-MAX-ALL-DONE + results file."""
import sys, time, json, os
sys.path.insert(0, 'scripts')
from ssh_fetch import connect, run

deadline = time.time() + 30 * 3600  # 30h max
while time.time() < deadline:
    try:
        ssh = connect()
        out, _ = run(ssh,
            'grep -ac BFCL-MAX-ALL-DONE /tmp/bfcl_max.log 2>/dev/null; '
            'ls /ccfa2026/compile-anything/results/bfcl_track/max_eval.json 2>/dev/null | wc -l; '
            'tail -c 200 /tmp/bfcl_max.log | tr "\\r" "\\n" | grep -av "warn\\|^$" | tail -1')
        lines = out.splitlines()
        done = lines[0].strip() if lines else '0'
        has_json = lines[1].strip() if len(lines) > 1 else '0'
        print(f'done={done} json={has_json} | {lines[-1][:100] if lines else ""}', flush=True)

        if done not in ('', '0') and has_json == '1':
            sftp = ssh.open_sftp()
            sftp.chdir('/ccfa2026/compile-anything/results/bfcl_track')
            with sftp.open('max_eval.json') as f:
                data = f.read()
            sftp.close()
            open('results/bfcl_track/max_eval.json', 'wb').write(data)
            d = json.loads(data)
            print('\n=== BFCL 7B MAX RESULTS ===')
            print(f"Held-out: {d['heldout']['OVERALL']['accuracy']:.2%}")
            print(f"Full train: {d['train_full']['OVERALL']['accuracy']:.2%}")
            for cat, r in sorted(d['heldout'].items()):
                if cat != 'OVERALL':
                    print(f"  {cat:30s} {r['accuracy']:6.2%}")
            print('\nMAX-COMPLETE')
            ssh.close()
            sys.exit(0)
        ssh.close()
    except SystemExit:
        raise
    except Exception as e:
        print(f'poll: {e}', flush=True)
    time.sleep(1800)  # 30 min
print('TIMEOUT')
sys.exit(1)
