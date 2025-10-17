VERSION=$1
BW=$2
JOBNUM=$4
NODENUM=$3
if [ -z "$VERSION" ]; then
  echo "Usage: $0 <version>"
  exit 1
fi

if [ "$VERSION" == "b" ]; then
  python3.11 compute.py "bundle-bw$BW-node$NODENUM-test$JOBNUM.log" --total-nodes "$NODENUM" > "bundle-bw$BW-node$NODENUM-test$JOBNUM-data.log"
  python3.11 statistics.py -f "bundle-bw$BW-node$NODENUM-test$JOBNUM.log" -o "bundle-bw$BW-node$NODENUM-test$JOBNUM-res.log"
  python3.11 analyzelog.py -f $(docker inspect --format='{{.LogPath}}' simulator-scheduler) -o "bundle-bw$BW-node$NODENUM-test$JOBNUM-score-all.log" --top10 "bundle-bw$BW-node$NODENUM-test$JOBNUM-score-top10.log"
elif [ "$VERSION" == "l" ]; then
  python3.11 compute.py "layer-bw$BW-node$NODENUM-test$JOBNUM.log" --total-nodes "$NODENUM" > "layer-bw$BW-node$NODENUM-test$JOBNUM-data.log"
  python3.11 statistics.py -f "layer-bw$BW-node$NODENUM-test$JOBNUM.log" -o "layer-bw$BW-node$NODENUM-test$JOBNUM-res.log"
  python3.11 analyzelog.py -f $(docker inspect --format='{{.LogPath}}' simulator-scheduler) -o "layer-bw$BW-node$NODENUM-test$JOBNUM-score-all.log" --top10 "layer-bw$BW-node$NODENUM-test$JOBNUM-score-top10.log"
elif [ "$VERSION" == "i" ]; then
  python3.11 compute.py "image-bw$BW-node$NODENUM-test$JOBNUM.log" --total-nodes "$NODENUM" > "image-bw$BW-node$NODENUM-test$JOBNUM-data.log"
  python3.11 statistics.py -f "image-bw$BW-node$NODENUM-test$JOBNUM.log" -o "image-bw$BW-node$NODENUM-test$JOBNUM-res.log"
  python3.11 analyzelog.py -f $(docker inspect --format='{{.LogPath}}' simulator-scheduler) -o "image-bw$BW-node$NODENUM-test$JOBNUM-score-all.log" --top10 "image-bw$BW-node$NODENUM-test$JOBNUM-score-top10.log"
else
  echo "Invalid version. Use 'b' for bundle or 'l' for layer."
  exit 1
fi