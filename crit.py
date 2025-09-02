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

mb = 1024 * 1024
gb = 1024 * mb
minThreshold = 20 * mb  # 20971520
maxContainerThreshold = 100 * mb  # 104857600

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

# when running
@dataclass
class PullTask:
    nodeID: str
    bundle: str
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

    diskCapacityBytes: int = 500 * gb
    bytesCache: int = 0  
    bytesPulling: int = 0

@dataclass
class PodSpec:
    # buddleName -> set(prefabIDs) | None  # None means all prefabs are needed in this bundle
    requirements: Dict[str, Optional[Set[str]]] 

# simulating
class SimulatorState:
    def __init__(self, bundleCatalog: Dict[str, BundleMeta], nodeIDs: Iterable[str], networkBW: float = 100*mb):
        self.catalog = bundleCatalog
        self.nodes: Dict[str, NodeState] = {nid: NodeState() for nid in nodeIDs}
        self.pods: Dict[str, PodSpec] = {}
        self.podsLock = threading.Lock()
        self.networkBW = networkBW
        self.pullStrategy = 1 # 1=拉取前记录, 2=拉取完再记录
        
        # 按节点记录“执行忙碌到”的绝对时间戳（单节点串行/队列假设）
        self.nodeExecBusyUntil: Dict[str, float] = {nid: 0.0 for nid in nodeIDs}
        # 记录每个 job 的关键时间，用于后续调度参考/诊断
        # 结构：podID -> {node, job_start_at, job_end_at, compute_ready_at, data_ready_at, eta, ready_at}
        self.jobRecords: Dict[str, Dict[str, float]] = {}
        self.nodeExecHeaps: Dict[str, List[float]] = {nid: [] for nid in nodeIDs}
        self.nodePodLimit: int = 8

    # calculate the score for a pod on a node
    # def scoreNodeForPod_old(self, nodeID: str, pod: PodSpec) -> int:
    #     node = self.nodes[nodeID]

    #     # 获锁后获取快照
    #     with node.lock:
    #         podsCount = len(node.pods)
    #         cacheSnapshot = {b: ids.copy() for b, ids in node.cache.items()}


    #     rawBytes = 0
    #     for bundle, req in pod.requirements.items():
    #         meta =  self.catalog.get(bundle)
    #         if not meta:
    #             continue
    #         needIDs = meta.allPrefabIDs if req is None else req
    #         haveIDs = cacheSnapshot.get(bundle, set())

    #         for pid in needIDs:
    #             if pid in haveIDs:
    #                 rawBytes += meta.prefabSizes.get(pid, 0)

    #     factor = math.sqrt(podsCount + 1)
    #     return int(rawBytes /factor)
    
    # def bindPodToNode_old(self, nodeID: str, podID: str, pod: PodSpec):
        # with self.podsLock:
        #     self.pods[podID] = pod
        
        # node = self.nodes[nodeID]
        # with node.lock:
        #     if podID in node.pods:
        #         return
        #     node.pods.add(podID)

        #     for bundle, req in pod.requirements.items():
        #         meta = self.catalog.get(bundle)
        #         if not meta:
        #             continue
        #         needIDs = meta.allPrefabIDs if req is None else req
        #         bucket = node.cache.setdefault(bundle, set())
        #         # 加入缓存（set 去重，不会重复计）
        #         bucket.update(pid for pid in needIDs if pid in meta.prefabSizes)

    def _bundle_sizes_for_ids(self, bundle: str, ids: Set[str]) -> int:
        meta = self.catalog.get(bundle)
        if not meta:
            return 0
        return sum(meta.prefabSizes.get(pid, 0) for pid in ids)
    
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
        for bundle, req in pod.requirements.items():  
            meta = self.catalog.get(bundle)  
            if not meta:  
                have[bundle] = set()  
                pulling[bundle] = set()  
                missing[bundle] = set()  
                continue  

            needIDs = meta.allPrefabIDs if req is None else req  
            haveIDs = node.cache.get(bundle, set())  
            pullingIDs = node.pullingCache.get(bundle, set())  

            have[bundle] = needIDs & haveIDs  
            pulling[bundle] = needIDs & pullingIDs  
            missing[bundle] = needIDs - haveIDs - pullingIDs  
        return have, pulling, missing  
    
    def _cleanup_completed_pulls(self, node: NodeState):
        """  
        把已到 endTime 的 PullTask 从 pullingCache 移到 cache，  
        同时更新 bytesPulling/bytesCache，清理 pullTasks。  
        """  
        now = time.monotonic()  
        completed: List[PullTask] = []  

        for task in list(node.pullTasks):
            if now >= task.endTime:
                 # 策略2：拉取完成后才对评分可见 => 放入 cache
                if self.pullStrategy == 2:
                    bucket = node.cache.setdefault(task.bundle, set())
                    bucket.update(task.prefabIDs)

               # 从拉取集中移除（两种策略都可以维护 pullingCache）
                pullingBucket = node.pullingCache.get(task.bundle)
                if pullingBucket:
                    pullingBucket.difference_update(task.prefabID)
                    if not pullingBucket:
                        node.pullingCache.pop(task.bundle, None)

                # 更新大小
                sizeDone = task.totalBytes
                node.bytesPulling = max(0, node.bytesPulling - sizeDone)
                node.bytesCache += sizeDone

                completed.append(task)
                # print(f"[Pull] {task.nodeID} completed {len(task.prefabIDs)} prefabs from {task.bundle}, +{sizeDone/mb:.1f}MB")
        for task in completed:
            node.pullTasks.remove(task)

         # 修正 pullBusyUntil（当没有任务时，允许回落到 now） 
        if not node.pullTasks and node.pullBusyUntil < now:
            node.pullBusyUntil = now

    def _cleanup_finished_exec_for_node(self, nodeID: str):
        """
        清理 node 上已完成的执行任务（基于当前时间），
        并用堆中“最大结束时间”刷新 nodeExecBusyUntil。
        """
        now = time.monotonic()
        heap = self.nodeExecHeaps.setdefault(nodeID, [])
        # 移除所有已经结束的任务
        while heap and heap[0] <= now:
            heapq.heappop(heap)
        # 将“节点忙碌到”的时间更新为堆中最大结束时间（无任务则回落到 now）
        self.nodeExecBusyUntil[nodeID] = (max(heap) if heap else now)

    def record_job_schedule(self, nodeID: str, podID: str, job_start_abs: float, job_end_abs: float, pod: PodSpec):
        """
        - compute_ready_at: 节点可执行该 job 的最早时间（受并发限制与正在运行任务影响）
        - data_ready_at: 该 job 依赖数据拉取完成的时间（受拉取队列影响）
        - eta: max(compute_ready_at - now, data_ready_at - now)
        - ready_at: max(compute_ready_at, data_ready_at)
        - 在 job_end_abs 将该任务的结束时间压入执行堆，以限制并发，并通过惰性清理回收
        """
        now = time.monotonic()
        node = self.nodes[nodeID]
        with node.lock:
            self._cleanup_completed_pulls(node)
            _, _, missing = self._needs_sets_for_pod_on_node(node, pod)
            newBytes = 0
            for bundle, ids in missing.items():
                newBytes += self._bundle_sizes_for_ids(bundle, ids)

            pull_wait = max(0.0, node.pullBusyUntil - now)
            new_pull_time = (newBytes / self.networkBW) if newBytes > 0 else 0.0
            data_ready_at = now + pull_wait + new_pull_time
       
        # 并发限制：清理已完成执行任务，计算计算就绪时间
        self._cleanup_finished_exec_for_node(nodeID)
        execHeap = self.nodeExecHeaps[nodeID]
        if len(execHeap) < self.nodePodLimit:
            compute_ready_at = now
        else:
            compute_ready_at = execHeap[0]  # 最早空闲 slot 的时间

        ready_at = max(compute_ready_at, data_ready_at)
        eta = ready_at - now

         # 将当前 job 的结束时间放入该节点的执行堆（占用一个并发 slot）
        if job_end_abs is not None:
            heapq.heappush(execHeap, job_end_abs)
            # 刷新“节点忙碌到”时间为当前所有任务最大结束时间
            self.nodeExecBusyUntil[nodeID] = max(execHeap) if execHeap else now

        # 记录本次任务的关键时间
        self.jobRecords[podID] = {
            "node": nodeID,
            "job_start_at": job_start_abs,
            "job_end_at": job_end_abs,
            "compute_ready_at": compute_ready_at,
            "data_ready_at": data_ready_at,
            "eta": eta,
            "ready_at": ready_at,
            "concurrency_limit": self.nodePodLimit,
            "running_slots_after_schedule": len(execHeap),
        }

    def scoreNodeForPod(self, nodeID: str, pod: PodSpec) -> int:
        if self.pullStrategy == 1:
            return self.scoreNodeForPod_Strategy1(nodeID, pod)
        else:
            return self.scoreNodeForPod_Strategy2(nodeID, pod)
        
    def bindPodToNode(self, nodeID: str, podID: str, pod: PodSpec):
        if self.pullStrategy == 1:
            self.bindPodToNode_Strategy1(nodeID, podID, pod)
        else:
            self.bindPodToNode_Strategy2(nodeID, podID, pod)

    def unbindPodFromNode(self, nodeID: str, podID: str, clearCache: bool = False):
        node = self.nodes[nodeID]
        with self.podsLock:
            pod = self.pods.get(podID)

        with node.lock:
            if podID in node.pods:
                node.pods.remove(podID)
            if clearCache and pod:
                for bundle, req in pod.requirements.items():
                    meta = self.catalog.get(bundle)
                    if not meta:
                        continue
                    needIDs = meta.allPrefabIDs if req is None else req
                    haveIDs = node.cache.get(bundle, set())
                    if haveIDs:
                        haveIDs.difference_update(needIDs)
                        if not haveIDs:
                            node.cache.pop(bundle, None)
        with self.podsLock:
            self.pods.pop(podID, None)

    """策略1 拉取前就记录到缓存 评分时可见"""
    def scoreNodeForPod_Strategy1(self, nodeID: str, pod: PodSpec) -> int:
        node = self.nodes[nodeID]

        with node.lock:
            self._cleanup_completed_pulls(node)
            podsCount = len(node.pods)
            cacheSnapshot = {b: ids.copy() for b, ids in node.cache.items()}
            
            hitBytes = 0
            for bundle, req in pod.requirements.items():
                meta =  self.catalog.get(bundle)
                if not meta:
                    continue
                needIDs = meta.allPrefabIDs if req is None else req
                haveIDs = cacheSnapshot.get(bundle, set())
                
                for pid in needIDs:
                    if pid in haveIDs:
                        hitBytes += meta.prefabSizes.get(pid, 0)

            factor = math.sqrt(podsCount + 1)
            return int(hitBytes /factor)          

    def bindPodToNode_Strategy1(self, nodeID: str, podID: str, pod: PodSpec):
        with self.podsLock:
            self.pods[podID] = pod
        
        node = self.nodes[nodeID]
        now = time.monotonic()
        with node.lock:
            self._cleanup_completed_pulls(node)
            if podID in node.pods:
                return
            node.pods.add(podID)

            _, _, missing = self._needs_sets_for_pod_on_node(node, pod)
            newBytes = 0
            for bundle, ids in missing.items():
                newBytes += self._bundle_sizes_for_ids(bundle, ids)

            # 容量校验
            if node.bytesCache + node.bytesPulling + newBytes > node.diskCapacityBytes:
                node.pods.remove(podID)
                with self.podsLock:
                    self.pods.pop(podID, None)
                raise RuntimeError(f"[ERROR] Node {nodeID} out of disk space for pod {podID}")
            # 策略1：评分可见 —— 将 missing 直接加入 cache（仅评分可见，不代表已完成）
            for bundle, ids in missing.items():
                if not ids:
                    continue
                node.cache.setdefault(bundle, set()).update(ids)
             # 构建拉取任务（串行排队），并预留 bytesPulling
            for bundle, ids in missing.items():
                if not ids:
                    continue
                sizeBytes = self._bundle_sizes_for_ids(bundle, ids)
                if sizeBytes <= 0:
                    continue

                # 可以选择是否把 ids 标到 pullingCache 做可视追踪
                node.pullingCache.setdefault(bundle, set()).update(ids)

                start = max(now, node.pullBusyUntil)
                end = start + sizeBytes / self.networkBW
                task = PullTask(
                    nodeID=nodeID,
                    bundle=bundle,
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
    def scoreNodeForPod_Strategy2(self, nodeID: str, pod: PodSpec) -> int:
        node = self.nodes[nodeID]

        with node.lock:
            self._cleanup_completed_pulls(node)
            podsCount = len(node.pods)
            cacheSnapshot = {b: ids.copy() for b, ids in node.cache.items()}

        hitBytes = 0
        for bundle, req in pod.requirements.items():
            meta =  self.catalog.get(bundle)
            if not meta:
                continue
            needIDs = meta.allPrefabIDs if req is None else req
            haveIDs = cacheSnapshot.get(bundle, set())
            
            for pid in needIDs:
                if pid in haveIDs:
                    hitBytes += meta.prefabSizes.get(pid, 0)
        factor = math.sqrt(podsCount + 1)
        return int(hitBytes /factor)
    
    def bindPodToNode_Strategy2(self, nodeID: str, podID: str, pod: PodSpec):
        with self.podsLock:
            self.pods[podID] = pod
        
        node = self.nodes[nodeID]
        now = time.monotonic()
        with node.lock:
            self._cleanup_completed_pulls(node)
            if podID in node.pods:
                return
            node.pods.add(podID)

            _, _, missing = self._needed_sets_for_pod_on_node(node, pod)
            # 计算需要新增的字节（不含 pulling 中的）
            newBytes = 0
            for bundle, ids in missing.items():
                newBytes += self._bundle_sizes_for_ids(bundle, ids)

            # 容量校验
            if node.bytesCache + node.bytesPulling + newBytes > node.diskCapacityBytes:
                node.pods.remove(podID)
                with self.podsLock:
                    self.pods.pop(podID, None)
                raise RuntimeError(f"Node {nodeID} lacks capacity for {podID}: "
                                f"need {newBytes/gb:.2f}GB, used {(node.bytesCache+node.bytesPulling)/gb:.2f}GB / {node.diskCapacityBytes/gb:.2f}GB")

            # 策略2：拉取中对评分不可见 —— 写入 pullingCache，不写 cache
            for bundle, ids in missing.items():
                if not ids:
                    continue
                node.pullingCache.setdefault(bundle, set()).update(ids)

                sizeBytes = self._bundle_sizes_for_ids(bundle, ids)
                if sizeBytes <= 0:
                    continue

                start = max(now, node.pullBusyUntil)
                end = start + (sizeBytes / self.networkBW)
                task = PullTask(
                    nodeID=nodeID,
                    bundle=bundle,
                    prefabIDs=set(ids),
                    totalBytes=sizeBytes,
                    startTime=start,
                    endTime=end,
                    podID=podID
                )
                node.pullTasks.append(task)
                node.pullBusyUntil = end
                node.bytesPulling += sizeBytes

 

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

def pick_best_node(
    state: SimulatorState,
    pod: PodSpec,
    job_end_abs: Optional[float] = None,  # 新增：本 job 的绝对结束时间（用于约束/优先选择）
) -> Optional[str]:
    """
    根据容量与 ETA 选择节点：
    - dataWait = max(0, pullBusyUntil - now) + newPullTime
    - computeWait（并发限制）:
        * 清理已完成的执行任务
        * 若运行中的任务数 < nodePodLimit => computeWait = 0
        * 否则 computeWait = (最早结束的正在运行任务时间 - now)
    - ETA = max(computeWait, dataWait)
    - 若提供 job_end_abs，则优先选择 now + ETA <= job_end_abs 的节点
    """
    bestNid = None
    bestETA = float("inf")
    bestScore = -1
    bestLoad = float("inf")
    bestFeasible = False  # 是否满足 job_end 约束

    now = time.monotonic()
    for nid, node in state.nodes.items():
        with node.lock:
            state._cleanup_completed_pulls(node)

            _, _, missing = state._needs_sets_for_pod_on_node(node, pod)
            newBytes = 0
            for bundle, ids in missing.items():
                newBytes += state._bundle_sizes_for_ids(bundle, ids)

            # 容量约束
            if node.bytesCache + node.bytesPulling + newBytes > node.diskCapacityBytes:
                continue

            pull_wait = max(0.0, node.pullBusyUntil - now)
            new_pull_time = (newBytes / state.networkBW) if newBytes > 0 else 0.0
            dataWait = pull_wait + new_pull_time

        # 计算并发限制导致的计算等待
        state._cleanup_finished_exec_for_node(nid)
        execHeap = state.nodeExecHeaps[nid]
        if len(execHeap) < state.nodePodLimit:
            computeWait = 0.0
        else:
            computeWait = max(0.0, execHeap[0] - now)  # 等到最早空闲的 slot

        eta = max(computeWait, dataWait)
        score = state.scoreNodeForPod(nid, pod)
        load = len(node.pods)

        feasible = (job_end_abs is None) or (now + eta <= job_end_abs)
        if (feasible and not bestFeasible) or \
           (feasible == bestFeasible and (
               (eta < bestETA) or
               (eta == bestETA and score > bestScore) or
               (eta == bestETA and score == bestScore and load < bestLoad)
           )):
            bestNid = nid
            bestETA = eta
            bestScore = score
            bestLoad = load
            bestFeasible = feasible

    return bestNid

if __name__ == "__main__":
    simulateFlag = 1  # 0=不模拟真实时间，1=模拟真实时间
    appJSON = loadAppJSON("apps.json")
    catalog = buildBundleCatalog(appJSON)

    # nodeIDs = [f"worker-{i}" for i in range(1, 1001)]
    nodeIDs = [f"worker-{i}" for i in range(1, 11)]
    state = SimulatorState(catalog, nodeIDs, networkBW=100*mb)
    state.pullStrategy = 1

    # events = load_simulation_events("2017-10-06-Simulation.json")
    events =load_simulation_events("test20.json")
    t0 = time.monotonic()
    scheduledCount = 0
    print("job, node, start_time, end_time, eta, running_pods")

    for idx, ev in enumerate(events):
        jobid = int(ev["jobid"])
        start_time = float(ev["start_time"])  # seconds
        end_time  = float(ev["end_time"])


        bundle = JOBID_TO_BUNDLE.get(jobid)
        if bundle is None:
            print(f"[WARN] Unknown jobid={jobid} at t={start_time}s. Please add mapping. Skip.")
            continue

        now = time.monotonic()
        job_start_abs = t0 + start_time
        job_end_abs = t0 + end_time if end_time > start_time else None
        if now < job_start_abs:
            time.sleep(job_start_abs - now)

        podID = f"job-{jobid}-{idx+1}"
        pod = PodSpec(requirements={bundle: None})

        bestNode = pick_best_node(state, pod, job_end_abs=job_end_abs)
        if bestNode is None:
            print(f"[ERROR] No available node for pod {podID} at t={start_time}s")
            continue
        
        state.record_job_schedule(bestNode, podID, job_start_abs, job_end_abs, pod)
        state.bindPodToNode(bestNode, podID, pod)

        if simulateFlag:
            ready_at = state.jobRecords[podID]["ready_at"]
            now2 = time.monotonic()
            delayToLaunch = max(0.1, ready_at - now2)
            lifetime = max(1.0, end_time - start_time)

            threading.Timer(delayToLaunch, 
                            createPodAndAutoDelete, 
                            args=(podID, bestNode),
                            kwargs={
                                "image": JOBID_TO_BUNDLE[jobid],
                                "cpu": "2",
                                "mem": "512Mi",
                                "ephemeralStorage": "1000Mi",
                                "lifetimeSeconds": lifetime,
                                "labels": {"job": JOBID_TO_BUNDLE[jobid]},
                                "annotations": {"simulator/start_time": str(start_time)},
                            }).start()

        else:
            # 只考虑调度，不考虑实际运行时间
            createPodAndAutoDelete(
                podID,
                bestNode,
                image=JOBID_TO_BUNDLE[jobid],
                cpu="2",
                mem="512Mi",
                ephemeralStorage="1000Mi",
                # lifetimeSeconds=end_time - start_time,
                lifetimeSeconds=1.0,
                labels={"job": JOBID_TO_BUNDLE[jobid]},
                annotations={"simulator/start_time": str(start_time)},
            )

        newScore = state.scoreNodeForPod(bestNode, pod)
        scheduledCount += 1

        # 只获取ETA
        print(f"{JOBID_TO_BUNDLE[jobid]}, {bestNode}, {end_time-start_time}, {(state.jobRecords[podID]['ready_at']-t0):.2f}, {len(state.nodes[bestNode].pods)}")


    
    print(f"Completed!")