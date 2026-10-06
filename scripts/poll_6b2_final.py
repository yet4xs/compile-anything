"""6B-2 final poller: exit only on RESOLVER-EVALV2-DONE."""
import sys, time, json
sys.path.insert(0, 'scripts')
from ssh_fetch import connect, run

deadline = time.time() + 4 * 3600
while time.time() < deadline:
    try:
        ssh = connect()
        out, _ = run(ssh, 'grep -ac RESOLVER-EVALV2-DONE /tmp/phase6b2_evalv2b.log 2>/dev/null; '
                          'grep -aE "^\\[R|setF1=|negtypes|R2 mean" /tmp/phase6b2_evalv2b.log | tail -12')
        lines = out.splitlines()
        done = lines[0].strip() if lines else '0'
        print(f'done={done} | {lines[-1][:120] if lines else ""}', flush=True)
        if done not in ('', '0'):
            sftp = ssh.open_sftp()
            sftp.chdir('/ccfa2026/compile-anything/results/phase6b')
            with sftp.open('resolver_eval_v2.json') as f:
                data = f.read()
            sftp.close()
            open('results/phase6b/resolver_eval_v2.json', 'wb').write(data)
            d = json.loads(data)
            print('\n=== 6B-2 FINAL TABLE ===')
            for k, v in d.items():
                if k.startswith('_'):
                    continue
                if k == 'R2_mean_std':
                    print('R2 mean±std:', json.dumps(v))
                    continue
                print(f"{k}: setF1={v['set_f1']} exact={v['exact_set']} "
                      f"P={v['set_precision']} R={v['set_recall']} "
                      f"top1={v['top1']} top3={v['top3']} "
                      f"noneF1={v['none_f1']} explicitNONE={v['explicit_none_accuracy']}")
                print(f"   negtypes: {json.dumps(v.get('negative_type_stats', {}))}")
            print('\n6B2-FINAL-COMPLETE')
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
