#!/usr/bin/env python3
import subprocess, json, concurrent.futures, time

KUBE_SERVER = "http://localhost:3131"  # 你的 kwok kube-apiserver
KUBECTL_BASE = [
    "kubectl",
    "--server", KUBE_SERVER,
    "--insecure-skip-tls-verify=true",
]

def run(cmd, stdin_str=None):
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if stdin_str is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate(stdin_str)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\nstdout:\n{out}\nstderr:\n{err}")
    return out

def create_node(idx: int, cpu="4", mem="8Gi", pods="110"):
    name = f"worker-{idx}"
    # 1) 创建 Node（metadata + labels）
    yaml_doc = f"""apiVersion: v1
kind: Node
metadata:
  name: {name}
  labels:
    kubernetes.io/role: worker
    node-type: standard
"""
    run(KUBECTL_BASE + ["apply", "-f", "-"], yaml_doc)

    # 2) 补丁 status（Ready + capacity/allocatable + addresses）
    payload = {
        "status": {
            "conditions": [{
                "type": "Ready",
                "status": "True",
                "reason": "KubeletReady",
                "message": "simulated ready status"
            }],
            "capacity": {
                "cpu": str(cpu),
                "memory": str(mem),
                "pods": str(pods),
            },
            "allocatable": {
                "cpu": str(cpu),
                "memory": str(mem),
                "pods": str(pods),
            },
            "addresses": [
                {"type": "InternalIP", "address": f"10.0.{idx//250}.{idx%250+1}"},
                {"type": "Hostname",   "address": name},
            ],
        }
    }
    run(KUBECTL_BASE + [
        "patch", "node", name,
        "--type=merge",
        "--subresource=status",
        "-p", json.dumps(payload)
    ])
    return name

def main(total=1000, workers=32):
    start = time.time()
    created = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(create_node, i) for i in range(1, total + 1)]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                f.result()
                created += 1
                if created % 50 == 0:
                    print(f"[{created}/{total}] nodes created...")
            except Exception as e:
                print("ERROR:", e)
    print(f"Done. Created {created}/{total} nodes in {time.time() - start:.1f}s")

if __name__ == "__main__":
    main(total=1000, workers=32)