"""6B-1 fixed-eval poller: exit only on COMPOSER-EVAL-DONE marker."""
import sys, time, json
sys.path.insert(0, 'scripts')
from ssh_fetch import connect, run

deadline = time.time() + 4 * 3600
while time.time() < deadline:
    try:
        ssh = connect()
        out, _ = run(ssh, 'grep -ac COMPOSER-EVAL-DONE /tmp/phase6b1_eval2.log 2>/dev/null; '
                          'grep -aE "^\\[C|valid=" /tmp/phase6b1_eval2.log | tail -8')
        lines = out.splitlines()
        done = lines[0].strip() if lines else '0'
        print(f'done={done} | last: {lines[-1][:130] if lines else ""}', flush=True)
        if done not in ('', '0'):
            sftp = ssh.open_sftp()
            sftp.chdir('/ccfa2026/compile-anything/results/phase6b')
            with sftp.open('composer_eval.json') as f:
                data = f.read()
            sftp.close()
            open('results/phase6b/composer_eval.json', 'wb').write(data)
            d = json.loads(data)
            print('\n=== 6B-1 COMPOSER RESULTS (pairing fixed) ===')
            print(f"{'Arm':<10s} {'Valid%':>7s} {'OpSeq%':>7s} {'SkillF1':>8s} {'EA-R%':>7s} {'Cond-R%':>8s} {'Halluc%':>8s}")
            for k, v in d.items():
                ca = v['condition_adherence']
                print(f"{k:<10s} {v['valid_pct']:>6.2f}% {v['opseq_pct']:>6.2f}% "
                      f"{v['skill_f1_macro_over_tasks']:>8.4f} {v['exec_action_recall_taskavg_pct']:>6.2f}% "
                      f"{ca['provided_skill_recall_pct']:>7.2f}% {ca['hallucinated_skill_rate_pct']:>7.2f}%")
            print('\nEVAL2-COMPLETE')
            ssh.close()
            sys.exit(0)
        ssh.close()
    except SystemExit:
        raise
    except Exception as e:
        print(f'poll error: {e}', flush=True)
    time.sleep(480)
print('TIMEOUT')
sys.exit(1)
