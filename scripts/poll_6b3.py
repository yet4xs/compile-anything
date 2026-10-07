"""6B-3 poller v2: requires DONE marker AND both integration JSONs."""
import sys, time, json, os
sys.path.insert(0, 'scripts')
from ssh_fetch import connect, run

deadline = time.time() + 10 * 3600
while time.time() < deadline:
    ok = False
    try:
        ssh = connect()
        out, _ = run(ssh, 'grep -ac PHASE6B3-INTERNAL-DONE /tmp/phase6b3.log 2>/dev/null; '
                          'ls /ccfa2026/compile-anything/results/phase6b/integration_s42.json '
                          '/ccfa2026/compile-anything/results/phase6b/integration_s44.json 2>/dev/null | wc -l; '
                          'grep -aE "^\[I" /tmp/phase6b3.log | tail -3')
        lines = out.splitlines()
        done = lines[0].strip() if lines else '0'
        jsons = lines[1].strip() if len(lines) > 1 else '0'
        print(f'done={done} jsons={jsons} | {lines[-1][:100] if lines else ""}', flush=True)
        if done not in ('', '0') and jsons == '2':
            sftp = ssh.open_sftp()
            sftp.chdir('/ccfa2026/compile-anything/results/phase6b')
            for fn in ('integration_s42.json', 'integration_s44.json'):
                with sftp.open(fn) as f:
                    data = f.read()
                open(f'results/phase6b/{fn}', 'wb').write(data)
            sftp.close()
            for fn in ('integration_s42.json', 'integration_s44.json'):
                d = json.loads(open(f'results/phase6b/{fn}', encoding='utf-8').read())
                print(f'\n=== {fn} ===')
                for k in ('I0_e1a_monolithic', 'I1_oracle_oracle', 'I2_learnedcanon_oracle',
                          'I3_oraclecanon_learnedres', 'I4_learned_learned', 'I4R1_learned_r1'):
                    if k in d['results']:
                        r = d['results'][k]
                        print(f"{k}: n={r['n']} valid={r.get('valid_pct')} "
                              f"opseq={r.get('opseq_pct')} skF1={r.get('skill_f1_macro')} "
                              f"EA-R={r.get('ea_recall_taskavg_pct')} final={r.get('final_success_pct')}")
            print('\n6B3-INTERNAL-COMPLETE')
            ok = True
        ssh.close()
        if ok:
            sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        print(f'poll error: {e}', flush=True)
    time.sleep(900)
print('TIMEOUT')
sys.exit(1)
