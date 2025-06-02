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

──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

#V1 Requierments for base insightface 0.7.3

    #for Torch on RTX 50000
        ├── pip install https://huggingface.co/w-e-w/torch-2.6.0-cu128.nv/resolve/main/torch-2.6.0+cu128.nv-cp310-cp310-win_amd64.whl
        └── pip install https://huggingface.co/w-e-w/torch-2.6.0-cu128.nv/resolve/main/torchvision-0.20.0a0+cu128.nv-cp310-cp310-win_amd64.whl

    
    #Main requirements for insightface
        ├── pip install insightface
        ├── pip install scipy
        ├── pip install opencv-python
        ├── pip install requests
        ├── pip install tqdm
        ├── pip install numpy<2.0.0
        └── pip install onnxruntime-gpu>=1.17.0


#V2
    #Optional/extra dependencies (for some features)
        ├── pip install Pillow          (image processing)
        ├── pip install matplotlib      (visualization)
        ├── pip install scikit-learn    (metrics, clustering)
        ├── pip install scikit-image    (image utilities)
        ├── pip install dlib            (sometimes used for face alignment)
        └── pip install onnx            (for ONNX model export/conversion)

Everthing was tested in test env
    filename: test_dependencies.py