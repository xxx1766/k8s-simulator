from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Iterable
import math
import json
import threading

testBundle = [
    "clip",
    "lora",
    "sam2",
    "sb3",
    "stablediffusion",
    "transformers",
    "whisper",
    "yolo11"
]
bundle2ID = {name: i for i, name in enumerate(testBundle, start=1)}
id2Bundle = [None] + testBundle
# print(bundle2ID["clip"])   # 1
# print(id2Bundle[3])

mb = 1024 * 1024
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
class NodeState:
    pods: Set[str] = field(default_factory=set)
    # bundleName -> set(prefabIDs) # pulled images
    cache: Dict[str, Set[str]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

@dataclass
class PodSpec:
    # buddleName -> set(prefabIDs) | None  # None means all prefabs are needed in this bundle
    requirements: Dict[str, Optional[Set[str]]] 

# simulating
class SimulatorState:
    def __init__(self, bundleCatalog: Dict[str, BundleMeta], nodeIDs: Iterable[str]):
        self.catalog = bundleCatalog
        self.nodes: Dict[str, NodeState] = {nid: NodeState() for nid in nodeIDs}
        self.pods: Dict[str, PodSpec] = {}
        self.podsLock = threading.Lock()

    # calculate the score for a pod on a node
    def scoreNodeForPod(self, nodeID: str, pod: PodSpec) -> int:
        node = self.nodes[nodeID]

        # 获锁后获取快照
        with node.lock:
            podsCount = len(node.pods)
            cacheSnapshot = {b: ids.copy() for b, ids in node.cache.items()}


        rawBytes = 0
        for bundle, req in pod.requirements.items():
            meta =  self.catalog.get(bundle)
            if not meta:
                continue
            needIDs = meta.allPrefabIDs if req is None else req
            haveIDs = cacheSnapshot.get(bundle, set())

            for pid in needIDs:
                if pid in haveIDs:
                    rawBytes += meta.prefabSizes.get(pid, 0)

        factor = math.sqrt(podsCount + 1)
        return int(rawBytes /factor)
    
    def bindPodToNode(self, nodeID: str, podID: str, pod: PodSpec):
        with self.podsLock:
            self.pods[podID] = pod
        
        node = self.nodes[nodeID]
        with node.lock:
            if podID in node.pods:
                return
            node.pods.add(podID)

            for bundle, req in pod.requirements.items():
                meta = self.catalog.get(bundle)
                if not meta:
                    continue
                needIDs = meta.allPrefabIDs if req is None else req
                bucket = node.cache.setdefault(bundle, set())
                # 加入缓存（set 去重，不会重复计）
                bucket.update(pid for pid in needIDs if pid in meta.prefabSizes)

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

def loadAppJSON(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    appJSON = loadAppJSON("apps.json")
    catalog = buildBundleCatalog(appJSON)

    nodeIDs = [f"worker-{i}" for i in range(1, 1001)]
    state = SimulatorState(catalog, nodeIDs)

    # podA = PodSpec(requirements={"clip": None})
    # scores = [(nid, state.scoreNodeForPod(nid, podA)) for nid in nodeIDs[:3]]
    # print("scores before bind:", scores)

    # # 绑定到 worker-1，视作缓存拉取完成
    # state.bindPodToNode("worker-1", "pod-A", podA)
    # # 再次给 worker-1 打分：此时应 > 0（已命中本地缓存）
    # print("worker-1 score now:", state.scoreNodeForPod("worker-1", podA))

    # get task seqs
    