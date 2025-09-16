import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Set, Optional, Iterable, List, Tuple, Any
import json
import threading
import subprocess
import argparse
import time
import os

KUBE_SERVER = "http://localhost:3131"
NAMESPACE = "default"
KUBECTL_BASE = [
    "kubectl",
    "--server", KUBE_SERVER,
    "--insecure-skip-tls-verify=true",
    "--namespace", NAMESPACE,
]

# crio
BUNDLE_NAMES = [
    "clip",
    "lora",
    "sam2",
    "sb3",
    "stablediffusion",
    "transformers",
    "tts",
    "whisper",
    "yolo11"
]
JOBID_TO_BUNDLE: Dict[int, str] = {i: name for i, name in enumerate(BUNDLE_NAMES)}
MAX_POD_CONCURRENCY = 8
PULL_STRATEGY = 1
BANDWIDTH = 100
NODE_NUM = 1000
KB= 1024
MB = 1024 * KB
GB = 1024 * MB
MIN_THRESHOLD = 20 * KB  
MAX_CONTAINER_THRESHOLD = 2 * GB  
MIN_NODE_SCORE  = 0
MAX_NODE_SCORE  = 100
FACTOR = 2.0

@dataclass
class BundleMeta:
    prefabSizes: Dict[str, int] 
    allPrefabIDs: Set[str]
    totalSize: int
    id: str
    tag: str 
    realSize: float

def buildPrefabCatalog(appJSON: Dict, bundleJSON: Dict) -> Dict[str, BundleMeta]:
    # app.json ==>
    # { bundle_name -> {"taskc": [...], "prefabs": { prefabID: size, ... } }, ... }
    catalog: Dict[str, BundleMeta] = {}
    for bundle_name, entry in appJSON.items():
        prefabSizes: Dict[str, int] = {}
        allIDs: Set[str] = set()

        taskc = entry.get("taskc")
        if taskc:
            prefabSizes[taskc["prefabID"]] = int(taskc["prefabSize"])
            allIDs.add(taskc["prefabID"])

        for p in entry.get("prefabs", []):
            prefabSizes[p["prefabID"]] = int(p["prefabSize"])
            allIDs.add(p["prefabID"])
        
        totalSize = sum(prefabSizes.values())
        catalog[bundle_name] = BundleMeta(prefabSizes=prefabSizes, 
                                          allPrefabIDs=allIDs, 
                                          totalSize=totalSize,
                                          id=bundleJSON.get(bundle_name, {}).get("Id", ""),
                                          tag=bundleJSON.get(bundle_name, {}).get("Tag", ""),
                                          realSize=float(bundleJSON.get(bundle_name, {}).get("Size", 0.0)))

    return catalog

@dataclass
class NodeState:
    cache: Dict[str, int] = field(default_factory=dict)
    
@dataclass
class PodSpec:
    requirements: Dict[str, Optional[Set[str]]] 

class SimulatorState:
    def __init__(self, catalog: Dict[str, BundleMeta], nodeIDs: Iterable[str], networkBW: float = 100*MB):
        self.nodes: Dict[str, NodeState] = {nid: NodeState() for nid in nodeIDs}
        self.catalog = catalog
        # self.pods: Dict[str, PodSpec] = {}
        self.networkBW = networkBW

    def _prefab_sizes_for_ids(self, prefab: str, ids: Set[str]) -> int:
        meta = self.catalog.get(prefab)
        if not meta:
            return 0
        return sum(meta.prefabSizes.get(pid, 0) for pid in ids)
    
    def _needs_for_pod_on_node(self, node: NodeState, pod: PodSpec) -> Dict[str, Set[str]]:
        haved: Dict[str, Set[str]] = {}
        needs: Dict[str, Set[str]] = {}
        for prefab, req in pod.requirements.items():
            meta = self.catalog.get(prefab)
            if not meta:
                haved[prefab] = set()
                needs[prefab] = set()
                continue
            havedIDs = node.cache.get(prefab, set())
            needIDs = meta.allPrefabIDs if req is None else req

            haved[prefab] = havedIDs & needIDs
            needs[prefab] = needIDs - havedIDs
        return needs
    
def calculatePullTime(state: SimulatorState, pod: PodSpec, nodeID: str) -> float:
    nodeName = f"worker-{nodeID}"
    node = state.nodes.get(nodeName)
    if not node:
        print(f"[ERROR] Node {nodeName} not found, using 0 pull time")
        return 0.0
    
    needs = state._needs_for_pod_on_node(node, pod)
    newBytes = 0
    for prefab, ids in needs.items():
        newBytes += state._prefab_sizes_for_ids(prefab, ids)
    pullTime = speedPulling(newBytes, state.networkBW) if newBytes > 0 else 0.0

    return pullTime

def speedPulling(size: int, bandwidth: int) -> float:
    return (size /bandwidth) / FACTOR

def _run(cmd, stdin_str=None, timeout=15):
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if stdin_str is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = p.communicate(stdin_str, timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        raise RuntimeError(f"cmd timeout: {' '.join(cmd)}\nstdout:\n{out}\nstderr:\n{err}")
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\nstdout:\n{out}\nstderr:\n{err}")
    return out

def createPodConfig(
    podName: str,
    lifetimeSeconds: float,
    image: str = "pause",
    cpu: str = "2",
    mem: str = "512Mi",
    ephemeralStorage: str | None = None,
    labels: dict | None = None,
    annotations: dict | None = None,
):
    labels = labels or {}
    annotations = annotations or {}

    resources = {
        "requests": {"cpu": cpu, "memory": mem},
        "limits": {"cpu": cpu, "memory": mem},
    }
    if ephemeralStorage:
        resources["requests"]["ephemeral-storage"] = ephemeralStorage
        resources["limits"]["ephemeral-storage"] = ephemeralStorage

    imageTag = f"{image}:latest"
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": podName,
            "namespace": NAMESPACE,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": {
            "schedulerName": "bundle-scheduler",
            "restartPolicy": "Never",
            "activeDeadlineSeconds": max(int(lifetimeSeconds),1),  # Invalid value: 0: must be between 1 and 2147483647, inclusive
            "containers": [
                {
                    "name": image,
                    "image": imageTag,                    # 比如 busybox/nginx
                    "resources": resources,
                }
            ],
        },
    }

    payload = json.dumps(pod)
    _run(KUBECTL_BASE + ["apply", "-f", "-"], stdin_str=payload)
    # print(f"[INFO] Pod {podName} created with image {imageTag}")
        
def deletePod(podName: str, force: bool = False):
    args = ["delete", "pod", podName]
    if force:
        args += ["--grace-period=0", "--force"]
    else:
        args += ["--grace-period=5"]  # 给5秒的优雅关闭时间
    try:
        _run(KUBECTL_BASE + args, timeout=15)  # 增加超时时间
    except Exception as e:
        if "NotFound" not in str(e):
            print(f"[WARN] delete {podName} failed: {e}")

def createPodAndAutoDelete(
    state: SimulatorState,
    pod: PodSpec,
    podName: str,
    image: str = "busybox",
    cpu: str = "2",
    mem: str = "512Mi",
    ephemeralStorage: str | None = None,
    labels: dict | None = None,
    annotations: dict | None = None,
    lifetimeSeconds: float = 1.0,
    initTime: float = 0.0,
):
    createPodConfig(
        podName, lifetimeSeconds, image, cpu, mem, ephemeralStorage, labels, annotations
    )
    nodeID = waitForPodScheduled(initTime, podName, image, timeoutSeconds=30)
    if nodeID is None:
        print(f"[ERROR] Pod {podName} scheduling timeout")
        return
    pullingTime = calculatePullTime(state, pod, nodeID)
    t1 = threading.Timer(pullingTime, bundleInfoToStore, args=(nodeID, image, state, pod))
    t1.start()
    print(f"[INFO] Pod {podName}, pullingTime: {pullingTime:.2f}s, lifetime: {lifetimeSeconds:.2f}s, node: {nodeID}, pipelineNum: {getSpecificNodePodCount(nodeID)}")
    t2 = threading.Timer(pullingTime+lifetimeSeconds, deletePod, args=(podName,))
    t2.start()
    return nodeID, pullingTime, getSpecificNodePodCount(nodeID)

def getNodePodCount(nodeName: str = None) -> dict:
    """
    Returns: dict{node_name: pod_count}
    """
    try:
        if nodeName:
            output = _run(KUBECTL_BASE + [
                "get", "pods", "--all-namespaces", 
                f"--field-selector=spec.nodeName={nodeName}",
                "--no-headers"
            ])
            lines = [line.strip() for line in output.strip().split('\n') if line.strip()]
            return {nodeName: len(lines)}
        else:
            output = _run(KUBECTL_BASE + [
                "get", "pods", "-o", "wide", "--all-namespaces"
            ])
            
            lines = output.strip().split('\n')[1:]  
            node_counts = {}
            
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 7:  
                        node_name = parts[6]  
                        node_counts[node_name] = node_counts.get(node_name, 0) + 1
            return node_counts
    except Exception as e:
        print(f"[ERROR] Failed to get pod counts: {e}")
        return {}

def getSpecificNodePodCount(nodeID: str) -> int:
    nodeName = f"worker-{nodeID}"
    counts = getNodePodCount(nodeName)
    return counts.get(nodeName, 0)

def factorTime(time: float, factor: float):
    return round(time/factor,3)

def backtoRealTime(time: float, factor: float):
    return round(time*factor,3)

def load_simulation_events(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        events = json.load(f)
    events = [e for e in events if "jobid" in e and "start_time" in e]
    events.sort(key=lambda x: x["start_time"])
    return events

def bundleInfoToStore(nodeID: str, bundle: str, state: SimulatorState, pod: PodSpec) -> Optional[str]:
    try:
        nid = int(nodeID)
        dir = f"/root/simulating/10.0.{nid // 250}.{nid % 250 + 1}"
        bundleFilepath = f"{dir}/bundles.json"
        filejsonPath = f"{dir}/PrefabService/File.json"
        nodeName = f"worker-{nodeID}"
        
        if not os.path.exists(dir):
            return f"Directory not found: {dir}"
        if not os.path.exists(bundleFilepath):
            return f"Bundle file not found: {bundleFilepath}"
        if not os.path.exists(filejsonPath):
            return f"File.json not found: {filejsonPath}"
        
        node = state.nodes.get(nodeName)
        if node is None:
            return f"Node not found: {nodeName}"
            
        if bundle not in state.catalog:
            return f"Bundle not in catalog: {bundle}"
        
        # Update bundles.json
        with open(bundleFilepath, "r") as f:
            bundles_data = json.load(f)
            
        if bundle not in bundles_data:
            bundles_data[bundle] = {
                "Id": state.catalog[bundle].id,
                "Name": bundle,
                "Tag": state.catalog[bundle].tag,
                "Size": state.catalog[bundle].realSize
            }

        with open(bundleFilepath, "w") as f:
            json.dump(bundles_data, f, indent=4)

        # Update File.json
        with open(filejsonPath, "r") as f:
            file_data = json.load(f)
            
        needs = state._needs_for_pod_on_node(node, pod)
        
        for bundle_name, ids in needs.items():
            for pid in ids:
                if pid not in file_data:
                    filesize = state.catalog[bundle].prefabSizes.get(pid, 0) if bundle in state.catalog else 0
                    file_data[pid] = {
                        "filename": pid, 
                        "filetype": "test", 
                        "filesize": filesize
                    }
            # Update node cache
            node.cache.setdefault(bundle, set()).update(ids)
            
        with open(filejsonPath, "w") as f:
            json.dump(file_data, f, indent=4)

        return None
    except Exception as e:
        print(f"[ERROR] bundleInfoToStore failed on node {nodeID}, bundle {bundle}: {e}")
        return str(e)
    
def waitForPodScheduled(initTime: float, podID: str, bundle: str, timeoutSeconds: int = 10):
    start = time.monotonic()
    while time.monotonic() - start < timeoutSeconds:
        try:
            output = _run(KUBECTL_BASE + ["get", "pod", podID, "-o", "json"])
            podInfo = json.loads(output)
            output = _run(KUBECTL_BASE + ["get", "pod", podID, "-o", "json"])
            podInfo = json.loads(output)

            status = podInfo.get("status", {})
            phase = status.get("phase", "")
            spec = podInfo.get("spec", {})
            nodeName = spec.get("nodeName", None)
            if phase == "Failed":
                print(f"[ERROR] Pod {podID} scheduling failed")
                return
            if nodeName:
                nodeID = nodeName.split("-")[1]
                return nodeID
        except Exception as e:
            print(f"[WARN] Failed to get pod {podID} info: {e}")
            pass

        time.sleep(0.1)

    return

def loadJSON(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
    
class timeOpt:
    def __init__(self):
        self.initTime = time.monotonic()

    def getinitTime(self):
        return self.initTime

if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("--bw", type=int, help="")
    parser.add_argument("--test", type=str, help="")
    parser.add_argument("--factor", type=float, help="")
    args = parser.parse_args()
    if args.factor is not None:
        FACTOR = args.factor
        
    print(f"[INFO] Using time factor: {FACTOR}")
    f = open(f"bundle-{args.bw}.log",'w')
    events = load_simulation_events(args.test)
    scheduedCount = 0

    timeopt = timeOpt()
    initTime = timeopt.getinitTime()

    nodeIDs = [f"worker-{i}" for i in range(1, NODE_NUM+1)]
    catalog = buildPrefabCatalog(loadJSON("apps.json"), loadJSON("Bundles.json"))
    state = SimulatorState(catalog, nodeIDs, networkBW=(args.bw)*MB)

    f.write("No, podname, jobid, node, image, startAbs, pulledAbs, edABS\n")
    for idx, ev in enumerate(events):
        jobid = ev["jobid"]
        start_time = float(ev["start_time"])
        end_time = float(ev["end_time"])

        startTime = factorTime(start_time, FACTOR)
        endTime = factorTime(end_time, FACTOR)
        duration = endTime - startTime
        bundle = JOBID_TO_BUNDLE.get(jobid)
        if bundle is None:
            print(f"[WARN] Unknown jobid {jobid}, skipping")
            continue
        podID = f"job-{jobid}-{idx+1}"
        pod = PodSpec(requirements={bundle: None})
        podStart = initTime + startTime
        now = time.monotonic()
        if podStart > now:
            time.sleep(podStart - now)
        
        nid, pullingTime, ppNum = createPodAndAutoDelete(
                                            state,
                                            pod,
                                            podID,
                                            image=bundle,
                                            cpu="2",
                                            mem="512Mi",
                                            ephemeralStorage="1000Mi",
                                            lifetimeSeconds=duration,
                                            initTime=initTime
                                        )
        scheduedCount += 1

        realStart = start_time
        realPulled = realStart + backtoRealTime(pullingTime, FACTOR)
        realEnd = realPulled + (end_time - start_time)
        f.write(f"{scheduedCount}, {podID}, {jobid}, {nid}, {bundle}, {realStart:.2f}, {realPulled:.2f}, {realEnd:.2f}, {ppNum}\n")

    print(f"bundle-{args.bw}-Completed!")
    time.sleep(30)
    f.flush()
    print("store records to file!")
    time.sleep(10)
    _run(KUBECTL_BASE + ["delete", "pods", "--all", "--force", "--grace-period=0"], timeout=10)
    print("force to close!")
    print("--------------------------------")