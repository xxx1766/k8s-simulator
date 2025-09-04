from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Iterable, List, Tuple, Any
import math
import json
import threading
import time
import subprocess
from textwrap import dedent
import heapq
import argparse

KUBE_SERVER = "http://localhost:3131"
NAMESPACE = "default"
KUBECTL_BASE = [
    "kubectl",
    "--server", KUBE_SERVER,
    "--insecure-skip-tls-verify=true",
    "--namespace", NAMESPACE,
]
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
    nodeName: str,
    image: str = "pause",
    cpu: str = "2",
    mem: str = "512Mi",
    ephemeralStorage: str | None = None,
    labels: dict | None = None,
    annotations: dict | None = None,
):
    """
    以 JSON 形式构造 Pod，并通过 kubectl apply -f - 创建。
    关键：spec.nodeName=指定节点，实现“直绑”到该节点；重启策略 Never；gracePeriod=0 便于快速删除。
    """
    labels = labels or {}
    annotations = annotations or {}

    resources = {
        "requests": {"cpu": cpu, "memory": mem},
        "limits": {"cpu": cpu, "memory": mem},
    }
    if ephemeralStorage:
        resources["requests"]["ephemeral-storage"] = ephemeralStorage
        resources["limits"]["ephemeral-storage"] = ephemeralStorage

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
            "nodeName": nodeName,                      # 直接绑定到节点
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 0,
            "containers": [
                {
                    "name": image,
                    "image": image,                    # 比如 busybox/nginx
                    "resources": resources,
                }
            ],
        },
    }

    payload = json.dumps(pod)
    _run(KUBECTL_BASE + ["apply", "-f", "-"], stdin_str=payload)
        
def deletePod(podName: str, force: bool = True):
    args = ["delete", "pod", podName]
    if force:
        args += ["--grace-period=0", "--force"]
    try:
        _run(KUBECTL_BASE + args, timeout=10)
    except Exception as e:
        if "NotFound" not in str(e):
            print(f"[WARN] delete {podName} failed: {e}")

def createPodAndAutoDelete(
    podName: str,
    nodeName: str,
    image: str = "busybox",
    cpu: str = "2",
    mem: str = "512Mi",
    ephemeralStorage: str | None = None,
    labels: dict | None = None,
    annotations: dict | None = None,
    lifetimeSeconds: float = 1.0,
):
    createPodConfig(
        podName, nodeName, image, cpu, mem, ephemeralStorage, labels, annotations
    )
    t = threading.Timer(lifetimeSeconds, deletePod, args=(podName,))
    t.start()

# # crit
# BUNDLE_NAMES = [
#     "clip",
#     "lora",
#     "sam2",
#     "sb3",
#     "stablediffusion",
#     "transformers",
#     "tts",
#     "whisper",
#     "yolo11"
# ]

# crio
BUNDLE_NAMES = [
    "clip",
    "lora-gpu",
    "sam2",
    "sb3-gpu",
    "stablediffusion",
    "transformers-gpu",
    "tts",
    "whisper",
    "yolo11-gpu"
]
JOBID_TO_BUNDLE: Dict[int, str] = {i: name for i, name in enumerate(BUNDLE_NAMES)}
MAX_POD_CONCURRENCY = 8
PULL_STRATEGY = 1
BANDWIDTH = 100
NODE_NUM = 10
KB= 1024
MB = 1024 * KB
GB = 1024 * MB
MIN_THRESHOLD = 20 * KB  
MAX_CONTAINER_THRESHOLD = 2 * GB  
MIN_NODE_SCORE  = 0
MAX_NODE_SCORE  = 100

# only read info
@dataclass
class BundleMeta:
    prefabSizes: Dict[str, int] 
    allPrefabIDs: Set[str]
    totalSize: int

def buildBundleCatalog(appJSON: Dict) -> Dict[str, BundleMeta]:
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
                                          totalSize=totalSize)

    return catalog

def buildLayerCatalog(payloadJSON: Dict) -> Dict[str, BundleMeta]:
    catalog: Dict[str, BundleMeta] = {}
    for imageName, entry in payloadJSON.items():
        layerSizes: Dict[str, int] = {}
        allIDs:Set[str] = set()

        for p in entry.get("LayersData", []):
            layerSizes[p["Digest"]] = int(p["Size"])
            allIDs.add(p["Digest"])

        totalSize = sum(layerSizes.values())
        catalog[imageName] = BundleMeta(prefabSizes=layerSizes,
                                        allPrefabIDs=allIDs,
                                        totalSize=totalSize)
    return catalog
    
# when running
@dataclass
class PullTask:
    nodeID: str
    prefab: str
    prefabID: Set[str]           
    totalBytes: int
    startTime: float
    endTime: float
    podID: str


@dataclass
class NodeState:
    pods: Set[str] = field(default_factory=set)
    # bundleName -> set(prefabIDs) # pulled images
    cache: Dict[str, Set[str]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    # 拉取中（策略2评分不可见，策略1仅用于容量与ETA计算）：bundleName -> set(prefabIDs)  
    pullingCache: Dict[str, Set[str]] = field(default_factory=dict)  
    pullTasks: List[PullTask] = field(default_factory=list)  # 按时间排队的拉取任务  
    pullBusyUntil: float = 0.0                               # 节点串行拉取队列忙到的时间戳 

    diskCapacityBytes: int = 500 * GB
    bytesCache: int = 0  
    bytesPulling: int = 0

@dataclass
class PodSpec:
    # buddleName -> set(prefabIDs) | None  # None means all prefabs are needed in this bundle
    requirements: Dict[str, Optional[Set[str]]] 

# simulating
class SimulatorState:
    def __init__(self, bundleCatalog: Dict[str, BundleMeta], nodeIDs: Iterable[str], networkBW: float = 100*MB):
        self.catalog = bundleCatalog
        self.nodes: Dict[str, NodeState] = {nid: NodeState() for nid in nodeIDs}
        self.pods: Dict[str, PodSpec] = {}
        self.podsLock = threading.Lock()
        self.networkBW = networkBW
        self.pullStrategy = PULL_STRATEGY # 1=拉取前记录, 2=拉取完再记录
        
        # 按节点记录“执行忙碌到”的绝对时间戳（单节点串行/队列假设）
        self.nodeExecBusyUntil: Dict[str, float] = {nid: 0.0 for nid in nodeIDs}
        # 记录每个 job 的关键时间，用于后续调度参考/诊断
        # 结构：podID -> {node, job_start_at, job_end_at, compute_ready_at, data_ready_at, eta, ready_at}
        self.jobRecords: Dict[str, Dict[str, float]] = {}
        self.nodeExecHeaps: Dict[str, List[float]] = {nid: [] for nid in nodeIDs}
        self.nodePodLimit: int = MAX_POD_CONCURRENCY

    def _prefab_sizes_for_ids(self, prefab: str, ids: Set[str]) -> int:
        meta = self.catalog.get(prefab)
        if not meta:
            return 0
        return sum(meta.prefabSizes.get(pid, 0) for pid in ids)
    
    # TODO: need to fix!!!
    def _needs_sets_for_pod_on_node(self, node: NodeState, pod: PodSpec) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
        """  
        返回三类集合：  
        - have: 已在 cache 中的  
        - pulling: 正在拉取中的  
        - missing: 既不在 cache 也不在 pulling 中的（需新增）  
        """  
        have: Dict[str, Set[str]] = {}  
        pulling: Dict[str, Set[str]] = {}  
        missing: Dict[str, Set[str]] = {}  
        for prefab, req in pod.requirements.items():  
            meta = self.catalog.get(prefab)  
            if not meta:  
                have[prefab] = set()  
                pulling[prefab] = set()  
                missing[prefab] = set()  
                continue  

            needIDs = meta.allPrefabIDs if req is None else req  
            haveIDs = node.cache.get(prefab, set())  
            pullingIDs = node.pullingCache.get(prefab, set())  

            have[prefab] = needIDs & haveIDs  
            pulling[prefab] = needIDs & pullingIDs  
            missing[prefab] = needIDs - haveIDs - pullingIDs  
        return have, pulling, missing  
    
    def _cleanup_completed_pulls(self, node: NodeState, nowTime: str):
        """  
        把已到 endTime 的 PullTask 从 pullingCache 移到 cache，  
        同时更新 bytesPulling/bytesCache，清理 pullTasks。  
        """  
        now = nowTime
        completed: List[PullTask] = []  

        for task in list(node.pullTasks):
            if now >= task.endTime:
                 # 策略2：拉取完成后才对评分可见 => 放入 cache
                if self.pullStrategy == 2:
                    bucket = node.cache.setdefault(task.prefab, set())
                    bucket.update(task.prefabIDs)

               # 从拉取集中移除（两种策略都可以维护 pullingCache）
                pullingBucket = node.pullingCache.get(task.prefab)
                if pullingBucket:
                    pullingBucket.difference_update(task.prefabID)
                    if not pullingBucket:
                        node.pullingCache.pop(task.prefab, None)

                # 更新大小
                sizeDone = task.totalBytes
                node.bytesPulling = max(0, node.bytesPulling - sizeDone)
                node.bytesCache += sizeDone

                completed.append(task)
                # print(f"[Pull] {task.nodeID} completed {len(task.prefabIDs)} prefabs from {task.prefab}, +{sizeDone/MB:.1f}MB")
        for task in completed:
            node.pullTasks.remove(task)

        if not node.pullTasks and node.pullBusyUntil < now:
            node.pullBusyUntil = now

    def _cleanup_finished_exec_for_node(self, nodeID: str, nowTime: str):
        """
        清理 node 上已完成的执行任务（基于当前时间），
        并用堆中“最大结束时间”刷新 nodeExecBusyUntil。
        """
        now = nowTime
        heap = self.nodeExecHeaps.setdefault(nodeID, [])
        # 移除所有已经结束的任务
        while heap and heap[0] <= now:
            heapq.heappop(heap)
        # 将“节点忙碌到”的时间更新为堆中最大结束时间（无任务则回落到 now）
        self.nodeExecBusyUntil[nodeID] = (max(heap) if heap else now)

    def _unbind_pod_from_node(self, nodeID: str, nowTime: float):
        node = self.nodes[nodeID]
        pod_list = []
        with node.lock:
            for pid in node.pods:
                pod_list.append(pid)
            for id in pod_list:
                if (self.jobRecords[id]['ready_at'] + self.jobRecords[id]['job_duration']) <= nowTime:
                    try:
                        node.pods.remove(id)
                        with self.podsLock:
                            self.pods.pop(id, None)
                    except:
                        print(nowTime,id,self.pods)

    def recordJobSchedule(self, nodeID: str, podID: str, nowTime: float, job_start_abs: float, job_end_abs: float, pod: PodSpec):
        """
        - compute_ready_at: 节点可执行该 job 的最早时间（受并发限制与正在运行任务影响）
        - data_ready_at: 该 job 依赖数据拉取完成的时间（受拉取队列影响）
        - eta: max(compute_ready_at - now, data_ready_at - now)
        - ready_at: max(compute_ready_at, data_ready_at)
        """
        now = nowTime
        node = self.nodes[nodeID]
        with node.lock:
            self._cleanup_completed_pulls(node, now)
            _, _, missing = self._needs_sets_for_pod_on_node(node, pod)
            newBytes = 0
            for prefab, ids in missing.items():
                newBytes += self._prefab_sizes_for_ids(prefab, ids)

            pull_wait = max(0.0, node.pullBusyUntil - now)
            new_pull_time = speedPulling(newBytes, self.networkBW) if newBytes > 0 else 0.0
            data_ready_at = now + pull_wait + new_pull_time
       
        # 并发限制：清理已完成执行任务，计算计算就绪时间
        self._unbind_pod_from_node(nodeID, nowTime)
        self._cleanup_finished_exec_for_node(nodeID, nowTime)
        execHeap = self.nodeExecHeaps[nodeID]
        if len(execHeap) < self.nodePodLimit:
            compute_ready_at = data_ready_at
        else:
            compute_ready_at = execHeap[0]  # 最早空闲 slot 的时间

        ready_at = max(compute_ready_at, data_ready_at)
        # print(f"{job_start_abs:.2f}, {pull_wait:.2f}, {new_pull_time:.2f}, {data_ready_at:.2f}, {compute_ready_at:.2f}, {ready_at:.2f}")

         # 将当前 job 的结束时间放入该节点的执行堆（占用一个并发 slot）
        if job_end_abs is not None:
            job_duration = job_end_abs - job_start_abs
            heapq.heappush(execHeap, ready_at + job_duration)
            # 刷新“节点忙碌到”时间为当前所有任务最大结束时间
            self.nodeExecBusyUntil[nodeID] = max(execHeap) if execHeap else now

        # 记录本次任务的关键时间
        self.jobRecords[podID] = {
            "node": nodeID,
            "jon_start_at": job_start_abs, 
            "job_duration": job_duration,
            "compute_ready_at": compute_ready_at,
            "data_ready_at": data_ready_at,
            "ready_at": ready_at,
            "concurrency_limit": self.nodePodLimit,
            "running_slots_after_schedule": len(execHeap),
        }

    def calculatePriority(self, sumScores: int, numPods: int):
        maxThreshold = MAX_CONTAINER_THRESHOLD * numPods
        if sumScores < MIN_THRESHOLD:
            sumScores = MIN_THRESHOLD
        elif sumScores > maxThreshold:
            sumScores = maxThreshold
        # print(MAX_NODE_SCORE * (sumScores - MIN_THRESHOLD) / (maxThreshold - MIN_THRESHOLD))
        return MAX_NODE_SCORE * (sumScores - MIN_THRESHOLD) / (maxThreshold - MIN_THRESHOLD)

    def scoreNodeForPod(self, nodeID: str, pod: PodSpec,nowTime: float) -> int:
        if self.pullStrategy == 1:
            return self.scoreNodeForPod_Strategy1(nodeID, pod, nowTime)
        else:
            return self.scoreNodeForPod_Strategy2(nodeID, pod, nowTime)
        
    def bindPodToNode(self, nodeID: str, podID: str, pod: PodSpec, nowTime: float):
        if self.pullStrategy == 1:
            self.bindPodToNode_Strategy1(nodeID, podID, pod, nowTime)
        else:
            self.bindPodToNode_Strategy2(nodeID, podID, pod, nowTime)

    

    """策略1 拉取前就记录到缓存 评分时可见"""
    def scoreNodeForPod_Strategy1(self, nodeID: str, pod: PodSpec, nowTime: float) -> int:
        node = self.nodes[nodeID]
        
        with node.lock:
            self._cleanup_completed_pulls(node, nowTime)
            podsCount = len(node.pods)
            haveIDs = []
            for _, ids in node.cache.items():
                for id in ids:
                    haveIDs.append(id)  
            # print(haveIDs)

            hitBytes = 0
            for prefab, req in pod.requirements.items():
                meta =  self.catalog.get(prefab)
                if not meta:
                    continue
                needIDs = meta.allPrefabIDs if req is None else req
                # haveIDs = cacheSnapshot.get(prefab, set())
                
                for pid in needIDs:
                    if pid in haveIDs:
                        hitBytes += meta.prefabSizes.get(pid, 0)
            
            factor = math.sqrt(max(podsCount, 1))
            score = self.calculatePriority(int(hitBytes / factor), max(podsCount, 1))  
            # print(hitBytes, score)
            return score

    def bindPodToNode_Strategy1(self, nodeID: str, podID: str, pod: PodSpec, nowTime: float):
        with self.podsLock:
            self.pods[podID] = pod
        
        node = self.nodes[nodeID]
        now = nowTime
        with node.lock:
            self._cleanup_completed_pulls(node, now)
            if podID in node.pods:
                return
            node.pods.add(podID)

            _, _, missing = self._needs_sets_for_pod_on_node(node, pod)
            newBytes = 0
            for prefab, ids in missing.items():
                newBytes += self._prefab_sizes_for_ids(prefab, ids)

            if node.bytesCache + node.bytesPulling + newBytes > node.diskCapacityBytes:
                node.pods.remove(podID)
                with self.podsLock:
                    self.pods.pop(podID, None)
                raise RuntimeError(f"[ERROR] Node {nodeID} out of disk space for pod {podID}")
            # 策略1：评分可见 —— 将 missing 直接加入 cache（仅评分可见，不代表已完成）
            for prefab, ids in missing.items():
                if not ids:
                    continue
                node.cache.setdefault(prefab, set()).update(ids)
             # 构建拉取任务（串行排队），并预留 bytesPulling
            for prefab, ids in missing.items():
                if not ids:
                    continue
                sizeBytes = self._prefab_sizes_for_ids(prefab, ids)
                if sizeBytes <= 0:
                    continue

                # 可以选择是否把 ids 标到 pullingCache 做可视追踪
                node.pullingCache.setdefault(prefab, set()).update(ids)

                start = max(now, node.pullBusyUntil)
                end = start + speedPulling(sizeBytes, self.networkBW)
                task = PullTask(
                    nodeID=nodeID,
                    prefab=prefab,
                    prefabID=set(ids),
                    totalBytes=sizeBytes,
                    startTime=start,
                    endTime=end,
                    podID=podID
                )
                node.pullTasks.append(task)
                node.pullBusyUntil = end
                node.bytesPulling += sizeBytes
                # print(f"[Pull] {nodeID} started pulling {len(ids)} prefabs from {bundle}, f"size={sizeBytes/1024/1024:.1f}MB, est_time={end-start:.2f}s, , queue-until={node.pullBusyUntil:.2f}")
                
    """策略2 拉取完成后再记录到缓存 评分时不可见"""
    def scoreNodeForPod_Strategy2(self, nodeID: str, pod: PodSpec, nowTime: float) -> int:
        node = self.nodes[nodeID]

        with node.lock:
            self._cleanup_completed_pulls(node, nowTime)
            podsCount = len(node.pods)
            cacheSnapshot = {b: ids.copy() for b, ids in node.cache.items()}

        hitBytes = 0
        for bundle, req in pod.requirements.items():
            meta =  self.catalog.get(bundle)
            if not meta:
                continue
            needIDs = meta.allPrefabIDs if req is None else req
            haveIDs = cacheSnapshot.get(bundle, set())
            # prefab去重
            processedIDs = []
            for pid in needIDs:
                if pid in haveIDs and pid not in processedIDs:
                    hitBytes += meta.prefabSizes.get(pid, 0)
                    processedIDs.append(pid)

        factor = math.sqrt(max(podsCount, 1))
        return self.calculatePriority(int(hitBytes /factor), max(podsCount, 1))
    
    def bindPodToNode_Strategy2(self, nodeID: str, podID: str, pod: PodSpec, nowTime: float):
        with self.podsLock:
            self.pods[podID] = pod
        
        node = self.nodes[nodeID]
        now = nowTime
        with node.lock:
            self._cleanup_completed_pulls(node, now)
            if podID in node.pods:
                return
            node.pods.add(podID)

            _, _, missing = self._needs_sets_for_pod_on_node(node, pod)
            # 计算需要新增的字节（不含 pulling 中的）
            newBytes = 0
            for bundle, ids in missing.items():
                newBytes += self._prefab_sizes_for_ids(bundle, ids)

            # 容量校验
            if node.bytesCache + node.bytesPulling + newBytes > node.diskCapacityBytes:
                node.pods.remove(podID)
                with self.podsLock:
                    self.pods.pop(podID, None)
                raise RuntimeError(f"Node {nodeID} lacks capacity for {podID}: "
                                f"need {newBytes/GB:.2f}GB, used {(node.bytesCache+node.bytesPulling)/GB:.2f}GB / {node.diskCapacityBytes/GB:.2f}GB")

            # 策略2：拉取中对评分不可见 —— 写入 pullingCache，不写 cache
            for prefab, ids in missing.items():
                if not ids:
                    continue
                node.pullingCache.setdefault(prefab, set()).update(ids)

                sizeBytes = self._prefab_sizes_for_ids(prefab, ids)
                if sizeBytes <= 0:
                    continue

                start = max(now, node.pullBusyUntil)
                end = start + speedPulling(sizeBytes, self.networkBW)
                task = PullTask(
                    nodeID=nodeID,
                    prefab=prefab,
                    prefabIDs=set(ids),
                    totalBytes=sizeBytes,
                    startTime=start,
                    endTime=end,
                    podID=podID
                )
                node.pullTasks.append(task)
                node.pullBusyUntil = end
                node.bytesPulling += sizeBytes

@dataclass
class TimeOpt:
    initialTime: float = 0
    globalTime: float = 0
    
    def getInitialTime(self):
        return self.initialTime
    
    def getGlobalTime(self):
        return self.globalTime
    
    def setGlobalTime(self, newTime):
        self.globalTime = newTime

def loadAppJSON(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def load_simulation_events(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        events = json.load(f)
    # 基本校验 + 排序
    events = [e for e in events if "jobid" in e and "start_time" in e]
    events.sort(key=lambda x: x["start_time"])
    return events

def pickBestNode(
    state: SimulatorState,
    pod: PodSpec,
    nowTime: float, 
) -> Optional[str]:
    """
    根据容量与 ETA 选择节点：
    - dataWait = max(0, pullBusyUntil - now) + newPullTime
    - computeWait（并发限制）:
        * 清理已完成的执行任务
        * 若运行中的任务数 < nodePodLimit => computeWait = 0
        * 否则 computeWait = (最早结束的正在运行任务时间 - now)
    - ETA = max(computeWait, dataWait)
    """
    bestNid = None
    bestETA = float("inf")
    bestScore = -1
    bestLoad = float("inf")

    now = nowTime
    for nid, node in state.nodes.items():
        with node.lock:
            state._cleanup_completed_pulls(node, now)

            _, _, missing = state._needs_sets_for_pod_on_node(node, pod)
            newBytes = 0
            for prefab, ids in missing.items():
                newBytes += state._prefab_sizes_for_ids(prefab, ids)

            # 容量约束
            if node.bytesCache + node.bytesPulling + newBytes > node.diskCapacityBytes:
                continue

            pull_wait = max(0.0, node.pullBusyUntil - now)
            new_pull_time = speedPulling(newBytes, state.networkBW) if newBytes > 0 else 0.0
            dataWait = pull_wait + new_pull_time

        # 计算并发限制导致的计算等待
        state._unbind_pod_from_node(nid, nowTime)
        state._cleanup_finished_exec_for_node(nid, now)
        execHeap = state.nodeExecHeaps[nid]
        if len(execHeap) < state.nodePodLimit:
            computeWait = 0.0
        else:
            computeWait = max(0.0, execHeap[0] - now)  # 等到最早空闲的 slot

        eta = max(computeWait, dataWait)
        score = state.scoreNodeForPod(nid, pod, now)
        load = len(node.pods)

        if (load < state.nodePodLimit) and ((score > bestScore) or (score == bestScore and eta < bestETA) or (eta == bestETA and score == bestScore and load < bestLoad)):
            bestNid = nid
            bestETA = eta
            bestScore = score
            bestLoad = load
        # print(nid, eta, score, load)

    # print("Best: ", bestNid, bestETA, bestScore, bestLoad, "\n-------------------------")
    return bestNid, bestScore

def printCatalog():
    pass
    
def printState():
    pass

def speedPulling(size: int, bandwidth: int):
    return (size /bandwidth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="set bandwidth size")
    parser.add_argument("--size", type=int, help="")
    parser.add_argument("--test", type=str, help="")
    parser.add_argument("--node", type=int, help="")
    args = parser.parse_args()

    # for crit
    # f = open(f"crit-{args.size}.log",'w')
    # appJSON = loadAppJSON("apps.json")
    # catalog = buildBundleCatalog(appJSON)

    # for crio
    f = open(f"crio-{args.size}.log",'w')
    payloadJSON = loadAppJSON("payload.json")
    catalog = buildLayerCatalog(payloadJSON)

    if args.node:
        nodeIDs = [f"worker-{i}" for i in range(1, args.node+1)]
    else:
        nodeIDs = [f"worker-{i}" for i in range(1, NODE_NUM+1)]

    timeOpt = TimeOpt()
    state = SimulatorState(catalog, nodeIDs, networkBW=(args.size)*MB)
    state.pullStrategy = PULL_STRATEGY

    # events =load_simulation_events("2017-10-06-Simulation.json")
    events =load_simulation_events(args.test)
    # events =load_simulation_events(JOB_SEQ_FILE)
    scheduledCount = 0
    # print("No., job, node, running_pods, score, compute_ready_at, data_ready_at, start_time, end_time")
    f.write("No., job, node, running_pods, score, compute_ready_at, data_ready_at, start_time, end_time\n")

    for idx, ev in enumerate(events):
        jobid = int(ev["jobid"])
        start_time = float(ev["start_time"])  # seconds
        end_time  = float(ev["end_time"])

        bundle = JOBID_TO_BUNDLE.get(jobid)
        if bundle is None:
            print(f"[WARN] Unknown jobid={jobid} at t={start_time}s. Please add mapping. Skip.")
            continue

        job_start_abs = timeOpt.getInitialTime() + start_time
        job_end_abs = timeOpt.getInitialTime() + end_time if end_time > start_time else None
        if timeOpt.getGlobalTime() < job_start_abs:
            timeOpt.setGlobalTime(job_start_abs)

        podID = f"job-{jobid}-{idx+1}"
        pod = PodSpec(requirements={bundle: None})

        bestNode, bestScore = pickBestNode(state, pod, timeOpt.getGlobalTime())
        if bestNode is None:
            print(f"[ERROR] No available node for pod {podID} at t={start_time}s")
            continue
        
        state.recordJobSchedule(bestNode, podID, timeOpt.getGlobalTime(), job_start_abs, job_end_abs, pod)
        state.bindPodToNode(bestNode, podID, pod, timeOpt.getGlobalTime())

        # 只考虑调度，不考虑实际运行时间
        createPodAndAutoDelete(
            podID,
            bestNode,
            image=JOBID_TO_BUNDLE[jobid],
            cpu="2",
            mem="512Mi",
            ephemeralStorage="1000Mi",
            lifetimeSeconds=1.0,
            labels={"job": JOBID_TO_BUNDLE[jobid]},
            annotations={"simulator/start_time": str(start_time)},
        )
        # newScore = state.scoreNodeForPod(bestNode, pod, timeOpt.getGlobalTime())
        scheduledCount += 1

        # No., job, node, running_pods, score, compute_ready_at, data_ready_at, start_time, end_time,
        # print(
        #     f"{scheduledCount}, "
        #     f"{JOBID_TO_BUNDLE[jobid]}, "
        #     f"{bestNode}, "
        #     f"{state.jobRecords[podID]['running_slots_after_schedule']}, "
        #     f"{newScore:.2f}, "
        #     f"{state.jobRecords[podID]['compute_ready_at']:.2f}, "
        #     f"{state.jobRecords[podID]['data_ready_at']:.2f}, "
        #     f"{state.jobRecords[podID]['jon_start_at']:.2f}, "
        #     f"{(state.jobRecords[podID]['ready_at'] + state.jobRecords[podID]['job_duration']):.2f}"
        # )
        f.write(
            f"{scheduledCount}, "
            f"{JOBID_TO_BUNDLE[jobid]}, "
            f"{bestNode}, "
            f"{state.jobRecords[podID]['running_slots_after_schedule']}, "
            f"{bestScore:.20f}, "
            f"{state.jobRecords[podID]['compute_ready_at']:.2f}, "
            f"{state.jobRecords[podID]['data_ready_at']:.2f}, "
            f"{state.jobRecords[podID]['jon_start_at']:.2f}, "
            f"{(state.jobRecords[podID]['ready_at'] + state.jobRecords[podID]['job_duration']):.2f}\n"
        )

    print(f"crit-{args.size}-Completed!")