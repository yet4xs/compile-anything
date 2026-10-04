"""6B-1 poller: exit only on PHASE6B1-CHAIN-DONE marker, then fetch results."""
import sys, time, json
sys.path.insert(0, 'scripts')
from ssh_fetch import connect, run

deadline = time.time() + 14 * 3600
while time.time() < deadline:
    try:
        ssh = connect()
        out, _ = run(ssh, 'grep -ac PHASE6B1-CHAIN-DONE /tmp/phase6b1.log 2>/dev/null; '
                          'grep -aE "^=== |FAILED" /tmp/phase6b1.log | tail -2; '
                          'tail -c 200 /tmp/phase6b1.log | tr "\\r" "\\n" | grep -v "^$" | tail -1')
        lines = out.splitlines()
        done = lines[0].strip() if lines else '0'
        print(f'done={done} | {lines[-1][:130] if lines else ""}', flush=True)
        if done not in ('', '0'):
            sftp = ssh.open_sftp()
            sftp.chdir('/ccfa2026/compile-anything/results/phase6b')
            with sftp.open('composer_eval.json') as f:
                data = f.read()
            sftp.close()
            open('results/phase6b/composer_eval.json', 'wb').write(data)
            d = json.loads(data)
            print('\n=== 6B-1 COMPOSER RESULTS ===')
            for k, v in d.items():
                ca = v.get('condition_adherence', {})
                print(f"{k}: valid={v['valid_pct']} opseq={v['opseq_pct']} "
                      f"EA-R={v['exec_action_recall_taskavg_pct']} "
                      f"cond-recall={ca.get('provided_skill_recall_pct')}% "
                      f"halluc={ca.get('hallucinated_skill_rate_pct')}%")
            print('\n6B1-COMPLETE')
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
