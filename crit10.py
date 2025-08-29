from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Iterable
import math
import json

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

def updateLocalBundlesList(node):
    # need lock for each node
    pass

def getLocalBundlesList(node):
    # simulate get pulled images from node
    pass

def bundleHandler(node, bundle, action):
    # action: pull / remove
    pass

def bundleScore():
    # get & calculate score for each bundle
    # return max_score bundle id

    pass

def sumScore():

    pass
def calculatePriority():
    # calculate priority score for a bundle on a node
    # return score
    pass

# only read info
class BundleMeta:
    prefabSizes: Dict[str, int] 
    allPrefabIDs: Set[str]
    totalSize: int

def buildBundleCatalog(appJSON: Dict) -> Dict[str, BundleMeta]:
    # app.json ==>
    # { bundle_name -> {"taskc": [...], "prefabs": { prefabID: size, ... } }, ... }
    catalog: Dict[str, BundleMeta] = {}
    for bundle_name, entry in appJSON.items():
        prefabSizes = {}
        allIDs = set()

        if "taskc" in entry and entry["taskc"]:
            p = entry["taskc"]
            prefabSizes[p["prefabID"]] = int64(p["prefabSize"])
            allIDs.add(p["prefabID"])

        if "prefabs" in entry.get("prefabs", []):
            prefabSizes[p["prefabID"]] = int64(p["prefabSize"])
            allIDs.add(p["prefabID"])
        
        totalSize = sum(prefabSizes.values())
        catalog[bundle_name] = BundleMeta(prefabSizes=prefabSizes, 
                                          allPrefabIDs=allIDs, 
                                          totalSize=totalSize)

    return catalog

# when running
class NodeState:
    pods: Set[str] = field(default_factory=set)
    # bundleName -> set(prefabIDs) # pulled images
    cache: Dict[str, Set[str]] = field(default_factory=dict)

class PodSpec:
    # buddleName -> set(prefabIDs) | None  # None means all prefabs are needed in this bundle
    requirements: Dict[str, Optional[Set[str]]] 

# simulating
class SimulatorState:
    def __init__(self, bundleCatalog: Dict[str, BundleMeta], nodeIDs: Iterable[str]):
        self.catalog = bundleCatalog
        self.nodes: Dict[str, NodeState] = {nid: NodeState() for nid in nodeIDs}
        self.pods: Dict[str, PodSpec] = {}

    # calculate the score for a pod on a node
    def scoreNodeForPod(self, nodeID: str, pod: PodSpec) -> int:
        node = self.nodes[nodeID]
        rawBytes = 0

        for bundle, req in pod.requirements.items():
            meta = self.catalog[nodeID]
            if not meta:
                continue
            needIDs = meta.allPrefabIDs if req is None else req
            haveIDs = node.cache.get(bundle, set())

            for pid in needIDs:
                if pid not in haveIDs:
                    rawBytes += meta.prefabSizes.get(pid, 0)

        factor = math.sqrt(len(node.pods) + 1)
        return int(rawBytes /factor)
    
    def bindPodToNode(self, nodeID: str, podID: str, pod: PodSpec):
        self.pods[podID] = pod
        node = self.nodes[nodeID]
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
        pod = self.pods.get(podID)
        if podID in node:
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
        self.pods.pop(podID, None)

def loadAppJSON(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)
    
def int64(x) -> int:
    return int(x) & 0xFFFFFFFFFFFFFFFF

if __name__ == "__main__":
    appJSON = loadAppJSON("apps.json")
    catalog = buildBundleCatalog(appJSON)

    nodeIDs = [f"worker-{i}" for i in range(1, 1001)]
    state = SimulatorState(catalog, nodeIDs)

    podA = PodSpec(requirements={"clip": None})
    scores = [(nid, state.scoreNodeForPod(nid, podA)) for nid in nodeIDs[:3]]
    print("scores before bind:", scores)

     # 绑定到 node-0，视作缓存拉取完成
    state.bind_pod("node-0", "pod-A", podA)

     # 再次给 node-0 打分：此时应 > 0（已命中本地缓存）
    print("node-0 score now:", state.score_node_for_pod("node-0", podA))