"""6B-1 final poller: exit only on FINAL-EVAL-DONE, fetch composer_eval_v2.json."""
import sys, time, json
sys.path.insert(0, 'scripts')
from ssh_fetch import connect, run

deadline = time.time() + 6 * 3600
while time.time() < deadline:
    try:
        ssh = connect()
        out, _ = run(ssh, 'grep -ac FINAL-EVAL-DONE /tmp/phase6b1_final.log 2>/dev/null; '
                          'grep -aE "^\\[C|parse=" /tmp/phase6b1_final.log | tail -10; '
                          'tail -c 150 /tmp/phase6b1_c0t.log | tr "\\r" "\\n" | grep -av "warn\\|^$" | tail -1')
        lines = out.splitlines()
        done = lines[0].strip() if lines else '0'
        print(f'done={done} | {lines[-1][:130] if lines else ""}', flush=True)
        if done not in ('', '0'):
            sftp = ssh.open_sftp()
            sftp.chdir('/ccfa2026/compile-anything/results/phase6b')
            with sftp.open('composer_eval_v2.json') as f:
                data = f.read()
            sftp.close()
            open('results/phase6b/composer_eval_v2.json', 'wb').write(data)
            d = json.loads(data)
            print('\n=== 6B-1 FINAL TABLE (all fixes) ===')
            hdr = (f"{'Arm':<20s} {'V-E2E%':>7s} {'OpE2E%':>7s} {'SkF1':>7s} "
                   f"{'EA-Rµ%':>7s} {'CondR|V%':>9s} {'CondR-E2E%':>11s} {'offCond%':>9s} {'unsup%':>7s}")
            print(hdr)
            for k, v in d.items():
                if k.startswith('_'):
                    continue
                print(f"{k:<20s} {v['valid_e2e_pct']:>6.2f}% {v['opseq_e2e_pct']:>6.2f}% "
                      f"{v['skill_micro_f1']:>7.4f} {v['exec_action_recall_micro_pct']:>6.2f}% "
                      f"{v['provided_skill_recall_given_valid_pct']:>8.2f}% "
                      f"{v['provided_skill_recall_e2e_pct']:>10.2f}% "
                      f"{v['off_condition_skill_rate_pct']:>8.2f}% "
                      f"{v['unsupported_vs_reference_skill_rate_pct']:>6.2f}%")
            print('\nFINAL-COMPLETE')
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
