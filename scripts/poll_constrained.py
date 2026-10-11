"""Poll constrained generation results."""
import sys, time, json
sys.path.insert(0, 'scripts')
from ssh_fetch import connect, run

deadline = time.time() + 4 * 3600
while time.time() < deadline:
    try:
        ssh = connect()
        out, _ = run(ssh, 'grep -ac CONSTRAINED-GEN-DONE /tmp/constrained_gen.log 2>/dev/null; '
                          'grep -aE \"Seed:|D0|D1|D2|D3|SUMMARY\" /tmp/constrained_gen.log 2>/dev/null | tail -15')
        lines = out.splitlines()
        done = lines[0].strip() if lines else '0'
        print(f'done={done}', flush=True)
        for l in lines[1:]:
            if l.strip():
                print(f'  {l.strip()[:110]}', flush=True)

        if done not in ('', '0'):
            sftp = ssh.open_sftp()
            sftp.chdir('/ccfa2026/compile-anything/results/phase7')
            with sftp.open('constrained_generation.json') as f:
                data = f.read()
            sftp.close()
            open('results/phase7/constrained_generation.json', 'wb').write(data)
            d = json.loads(data)
            print('\n=== CONSTRAINED GENERATION RESULTS ===')
            for seed in ('s42','s43','s44'):
                if seed not in d: continue
                for lv in ('D0_normal','D1_grammar','D2_typed_ssa','D3_control'):
                    r = d[seed][lv]
                    print(f\"  {seed}/{lv}: valid={r['valid_pct']}% opseq={r['opseq_pct']}% corr={r['corrections']} rej={r['rejections']}\")
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
