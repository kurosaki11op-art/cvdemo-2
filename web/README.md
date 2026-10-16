# Footfall Lab (browser build)

`footfall_lab.html` is the single-page browser port of the counting pipeline (tracker, tripwire, homography,
heatmap, identity filters) with a store simulator that has known ground truth. It was published as a Claude
Artifact. Video mode additionally needs the COCO-SSD "lite_mobilenet_v2" TensorFlow.js model files in `model/`
(not stored here, 17 MB): `https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/`.
The weight shards were published base64-encoded as `model/shardN.b64.txt` because the host serves text only.
This port is for experimenting; the production system is the Python `nexus` package.
