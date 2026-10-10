"""WorfBench comparison poller."""
import sys, time, json
sys.path.insert(0, 'scripts')
from ssh_fetch import connect, run

deadline = time.time() + 6 * 3600
while time.time() < deadline:
    try:
        ssh = connect()
        out, _ = run(ssh, 'grep -ac WORFBENCH-COMPARE-DONE /tmp/worfbench.log 2>/dev/null; '
                          'ls /ccfa2026/compile-anything/results/worfbench/comparison.json 2>/dev/null | wc -l')
        lines = out.splitlines()
        done = lines[0].strip() if lines else '0'
        has_json = lines[1].strip() if len(lines) > 1 else '0'
        print(f'done={done} json={has_json}', flush=True)

        if done not in ('', '0') and has_json == '1':
            sftp = ssh.open_sftp()
            sftp.chdir('/ccfa2026/compile-anything/results/worfbench')
            with sftp.open('comparison.json') as f:
                data = f.read()
            sftp.close()
            open('results/worfbench/comparison.json', 'wb').write(data)
            d = json.loads(data)
            print('\n=== WORFBENCH RESULTS ===')
            for key in sorted(d.keys()):
                r = d[key]
                print(f"  {key:35s} nodeF1={r['node_f1_avg']:.4f} "
                      f"edgeF1={r['edge_f1_avg']:.4f} graphF1={r['graph_f1_avg']:.4f}")
            print('\nCOMPLETE')
            ssh.close()
            sys.exit(0)
        ssh.close()
    except SystemExit:
        raise
    except Exception as e:
        print(f'poll: {e}', flush=True)
    time.sleep(600)
print('TIMEOUT')
sys.exit(1)
