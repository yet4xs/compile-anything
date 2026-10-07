"""6B-3 poller: exit on PHASE6B3-INTERNAL-DONE, fetch both seed JSONs."""
import sys, time, json
sys.path.insert(0, 'scripts')
from ssh_fetch import connect, run

deadline = time.time() + 10 * 3600
while time.time() < deadline:
    try:
        ssh = connect()
        out, _ = run(ssh, 'grep -ac PHASE6B3-INTERNAL-DONE /tmp/phase6b3.log 2>/dev/null; '
                          'grep -aE "^\\[I|resolver done" /tmp/phase6b3.log | tail -6')
        lines = out.splitlines()
        done = lines[0].strip() if lines else '0'
        print(f'done={done} | {lines[-1][:110] if lines else ""}', flush=True)
        if done not in ('', '0'):
            sftp = ssh.open_sftp()
            sftp.chdir('/ccfa2026/compile-anything/results/phase6b')
            for fn in ('integration_s42.json', 'integration_s44.json'):
                try:
                    with sftp.open(fn) as f:
                        data = f.read()
                    open(f'results/phase6b/{fn}', 'wb').write(data)
                    d = json.loads(data)
                    print(f'\n=== {fn} ===')
                    for k in ('I0_e1a_monolithic', 'I1_oracle_oracle', 'I2_learnedcanon_oracle',
                              'I3_oraclecanon_learnedres', 'I4_learned_learned', 'I4R1_learned_r1'):
                        if k in d['results']:
                            r = d['results'][k]
                            print(f"{k}: n={r['n']} valid={r.get('valid_pct')} "
                                  f"opseq={r.get('opseq_pct')} skF1={r.get('skill_f1_macro')} "
                                  f"EA-R={r.get('ea_recall_taskavg_pct')} "
                                  f"final={r.get('final_success_pct')}")
                except Exception as e:
                    print('skip', fn, e)
            sftp.close()
            print('\n6B3-INTERNAL-COMPLETE')
            ssh.close()
            sys.exit(0)
        ssh.close()
    except SystemExit:
        raise
    except Exception as e:
        print(f'poll error: {e}', flush=True)
    time.sleep(900)
print('TIMEOUT')
sys.exit(1)
