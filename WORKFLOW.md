+-----------------------------+
|   data-curation-pipeline    |
+-----------------------------+
    │
    └── Main Tab [ Image / video input]
                │
                |
                └──File type split
                    |    |
                    |    └── Video Stored
                    |         │                                                                                                                   
                    |         └── Video Analysis
                    |               |
                    |               └──> Stage 1 [OpenCV, InsightFace Thresholds:Low | Purpose: Maximize Recall | Speed: Fast]
                    |                       │
                    |                       └── Store data
                    |                               │
                    |                               └────> Stage 2 [CLIP, DeepFace, Thresholds:Medium | Purpose: Quality/attribute sort | Speed: Medium]
                    |                                        │
                    |                                        └── Store data
                    |                                               |
                    |                                               └────> Stage 3 [Ensemble, Thresholds:High | Purpose: Best for training | Speed: Slowest]
                    |                                                           |
                    |                                                           └───> Store best frames ──>──┐
                    |                                                                                        │
                    └── Image Stored  <───────────────────────────────────────────────────────────────────<──┘                     
                            |
                            └───┐
                                ├── Face Analysis
                                │   ├── detection
                                │   ├── emotion
                                │   ├── pose
                                │   ├── features
                                │   └── tattoos
                                │
                                ├── Clothing Analysis
                                ├── Pose/Body Analysis
                                ├── Camera Analysis
                                ├── NSFW Analysis
                                ├── Accessories Analysis


                                

                                └── Aggregation & Metadata
                                    └── Filtering & Export
                                        └── Output Folder




└──

──┐

──┘

┌──


──

├──


│

──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


Video Handler:
