#!/usr/bin/env python3
import argparse
import threading
import subprocess, json, concurrent.futures, time

KUBE_SERVER = "http://localhost:3131"  # 你的 kwok kube-apiserver
KUBECTL_BASE = [
    "kubectl",
    "--server", KUBE_SERVER,
    "--insecure-skip-tls-verify=true",
]
WORKDIR = "/root/simulating"

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

def create_node(idx: int, cpu="16", mem="32Gi", pods="8", storage="50Gi"):
    name = f"worker-{idx}"
    node_ip = f"10.0.{idx//250}.{idx%250+1}"

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
                "ephemeral-storage": str(storage),
            },
            "allocatable": {
                "cpu": str(cpu),
                "memory": str(mem),
                "pods": str(pods),
                "ephemeral-storage": str(storage),
            },
            "addresses": [
                {"type": "InternalIP", "address": node_ip},
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
    # create a directory to simulate node's filesystem
    #   workdir/InternalIP/PrefabService/File.json
    #   workdir/InternalIP/images.json
    run(["mkdir", "-p", f"{WORKDIR}/{node_ip}"])
    # create an empty images.json file
    run(["mkdir", "-p", f"{WORKDIR}/{node_ip}/PrefabService"])
    run(["bash", "-c", f"echo '{{}}' > {WORKDIR}/{node_ip}/images.json"])
    run(["bash", "-c", f"echo '{{}}' > {WORKDIR}/{node_ip}/PrefabService/File.json"])
    
    return name


def main(total=1000, workers=32):
    start = time.time()
    # kubectl delete nodes --all
    run(KUBECTL_BASE + ["delete", "nodes", "--all"])
    time.sleep(1)
    created = 0
    created_lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_node = {ex.submit(create_node, i): i for i in range(1, total + 1)}
        for future in concurrent.futures.as_completed(future_to_node):
            node_id = future_to_node[future]
            try:
                future.result()
                with created_lock:
                    created += 1
                    current_created = created
                
                cmd = ["kubectl", "taint", "nodes", f"worker-{node_id}", 
                       "node.kubernetes.io/not-ready:NoSchedule-"]
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error removing taint from worker-{node_id}: {e}")

                if current_created % 50 == 0:
                    print(f"[{current_created}/{total}] nodes created...")
                    
            except Exception as e:
                print(f"ERROR creating worker-{node_id}: {e}")
    
    print(f"Done. Created {created}/{total} nodes in {time.time() - start:.1f}s")

if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, help="")
    args = parser.parse_args()
    run(["kubectl", "delete", "pods", "--all", "--force", "--grace-period=0"])
    run(["kubectl", "delete", "nodes", "--all"])
    run(["rm", "-rf", WORKDIR])

    main(total=args.n, workers=32)