"""Poll Phase 6A chain; fetch metrics when the whole chain completes."""
import sys, time, json

sys.path.insert(0, 'scripts')
from ssh_fetch import connect, run

deadline = time.time() + 16 * 3600
while time.time() < deadline:
    try:
        ssh = connect()
        out, _ = run(ssh, 'pgrep -f "run_phase6a_followups2[.]sh|eval_phase6a_external[.]py|eval_phase6a_followups[.]py" >/dev/null && echo CHAIN=RUNNING || echo CHAIN=DONE; '
                          'grep -c "FOLLOWUPS2-DONE" /tmp/phase6a_followups.log 2>/dev/null; '
                          'grep -E "=== train|=== eval|Segmen|Error" /tmp/phase6a_followups.log 2>/dev/null | tail -3; '
                          'tail -c 200 /tmp/phase6a_followups.log | tr "\\r" "\\n" | tail -1')
        status = dict(l.strip().split('=', 1) for l in out.splitlines() if '=' in l and l.startswith('CHAIN'))
        print(out.strip()[:400], flush=True)
        if status.get('CHAIN') == 'DONE' or 'FOLLOWUPS2-DONE' in out:
            # fetch metrics
            sftp = ssh.open_sftp()
            sftp.chdir('/ccfa2026/compile-anything/results/phase6')
            for name in ('phase6a_external.json', 'phase6a_followups.json'):
                try:
                    with sftp.open(name) as f:
                        data = f.read()
                    open(f'results/phase6/{name}', 'wb').write(data)
                    print('fetched', name)
                except Exception as e:
                    print('skip', name, e)
            sftp.close()
            with open('results/phase6/phase6a_metrics.json', 'rb') as f:
                data = f.read()
            d = json.loads(data)
            print('\n=== KEY RESULTS ===')
            for k in d:
                if any(x in k for x in ('boundary', 'counterfactual', 'probe_C', 'no_call')):
                    print(k, '->', json.dumps(d[k])[:220])
            print('\nALL PHASE6A INTERNAL COMPLETE')
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
