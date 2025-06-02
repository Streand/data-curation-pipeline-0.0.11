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
    Stage 1
        Video Processing Parameters Explained
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

Sample Rate (frames to skip)
    What it does: Controls how many frames to skip between extractions from the video.
        - Sample rate = 1: Extract every single frame
        - Sample rate = 5: Extract 1 frame, skip 4 frames, and repeat
        - Sample rate = 15: Extract 1 frame, skip 14 frames, and repeat
    Impact: Lower values provide more thorough analysis but slower processing. Higher values are faster but might miss good frames.


    Presets:
        - TikTok/Instagram: 1 (extract every frame - ideal for short videos)
        - YouTube: 5 (skip frames - suitable for longer content)
        - Custom: 15 (skip many frames - for very long videos)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

Maximum Number of Frames to Extract
    What it does: Ensures selected frames aren't too similar by enforcing a minimum separation.
        How it works: After finding good frames, the system will only select frames that are at least this many frames apart from each other in the original video.
    Impact:
        - Smaller distance (e.g., 15): More similar frames might be included
        - Larger distance (e.g., 30): Greater visual diversity but might skip good frames
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

Try to detect scene changes
    What it does: Uses computer vision to identify major visual changes (scene transitions) in the video.
        - Prioritizes frames near scene changes
        - Better coverage of different scenes/settings in the video
        - Improves frame diversity
    How it works:
        - Uses the SceneDetect library to identify content changes
        - Gives a 20% score bonus to frames near scene transitions
        - Helps ensure you get at least one frame from each major scene
Note: This is slower, so it's optional and best used with longer videos that contain multiple scenes.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

