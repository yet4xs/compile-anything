"""SOTA pipeline poller: EF-SC → V-DPO → Final Eval."""
import sys, time, json
sys.path.insert(0, 'scripts')
from ssh_fetch import connect, run

deadline = time.time() + 20 * 3600
while time.time() < deadline:
    try:
        ssh = connect()
        out, _ = run(ssh, 'grep -ac SOTA-PIPELINE-DONE /tmp/sota.log 2>/dev/null; '
                          'grep -aE \"Phase|EF-SC-DONE|V-DPO-DONE|FINAL-EVAL-DONE|AST accuracy|FAILED\" /tmp/sota.log 2>/dev/null | tail -10')
        lines = out.splitlines()
        done = lines[0].strip() if lines else '0'
        print(f'done={done}', flush=True)
        for l in lines[1:]:
            if l.strip():
                print(f'  {l.strip()[:120]}', flush=True)

        if done not in ('', '0'):
            # Fetch results
            for fn in ('results/bfcl_track/ef_sc_results.json',
                       'results/bfcl_track/vdpo_stats.json',
                       'results/bfcl_track/final_comparison.json'):
                try:
                    sftp = ssh.open_sftp()
                    sftp.chdir('/ccfa2026/compile-anything')
                    with sftp.open(fn) as f:
                        data = f.read()
                    sftp.close()
                    import os
                    local = fn.replace('results/', 'results/')
                    os.makedirs(os.path.dirname(local), exist_ok=True)
                    open(local, 'wb').write(data)
                    d = json.loads(data)
                    print(f'\n{fn}:')
                    if 'OVERALL' in d:
                        print(f'  OVERALL: {d[\"OVERALL\"][\"accuracy\"]:.2%}')
                    if 'accuracy' in d:
                        print(f'  accuracy: {d[\"accuracy\"]:.2%}')
                    if isinstance(d, list):
                        for r in d:
                            print(f'  {r.get(\"label\",\"?\")}: {r.get(\"accuracy\",0):.2%}')
                except Exception as e:
                    print(f'  {fn}: {e}')
            print('\nSOTA-COMPLETE')
            ssh.close()
            sys.exit(0)
        ssh.close()
    except SystemExit:
        raise
    except Exception as e:
        print(f'poll: {e}', flush=True)
    time.sleep(900)
print('TIMEOUT')
sys.exit(1)
