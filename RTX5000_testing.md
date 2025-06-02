──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

cheat sheet

RTX 5000 testing, Blackwell arcetectur
    folder -> GPU_testing

        cd .\GPU_testing\
        python -m venv venv
        venv\Scripts\activate

        pip install ******

        pip uninstall ********

        pip uninstall -y *********

testing file: 
    test_onnxruntime-gpu.py


──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


        
main requirements for insightface (especially for insightface>=0.6.0
    ✅  onnxruntime-gpu (for ONNX model inference on GPU)
    ✅  torch (for PyTorch model inference)
    ✅  numpy,
    ✅  scipy
    ✅  opencv-python
    ✅  requests
    ✅  tqdm

    Testing requirements
    ✅   CV2

    Optional/extra dependencies (for some features)
        mxnet (for legacy models, rarely needed in recent versions)
        Pillow (image processing)
        matplotlib (visualization)
        scikit-learn (metrics, clustering)
        scikit-image (image utilities)
        dlib (sometimes used for face alignment)
        onnx (for ONNX model export/conversion)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

    tested in venv
            pip install numpy<2.0.0
            pip install onnxruntime-gpu==1.16.3 #same resault
                └──pip install onnxruntime-gpu>=1.17.0 #same resault #currently installed

    for Torch on RTX 50000
        ├── pip install https://huggingface.co/w-e-w/torch-2.6.0-cu128.nv/resolve/main/torch-2.6.0+cu128.nv-cp310-cp310-win_amd64.whl
        └── pip install https://huggingface.co/w-e-w/torch-2.6.0-cu128.nv/resolve/main/torchvision-0.20.0a0+cu128.nv-cp310-cp310-win_amd64.whl

    rest of requirements for insightface
        ├── pip install scipy
        ├── pip install opencv-python
        ├── pip install requests
        └── pip install tqdm




──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

with current venv

To see your actual GPU model, you can use:
import torch
print(torch.cuda.get_device_name(0))
    └── NVIDIA GeForce RTX 5080

All main requirements for insightface are installed in venv
    testing file:
        test_insightface.py
            └──requierments: ok


