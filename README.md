
<div align="center">
<img src="https://mmf.sh/img/logo.svg" width="50%"/>
</div>

#

<div align="center">
  <a href="https://mmf.sh/docs">
  <img alt="Documentation Status" src="https://readthedocs.org/projects/mmf/badge/?version=latest"/>
  </a>
  <a href="https://circleci.com/gh/facebookresearch/mmf">
  <img alt="CircleCI" src="https://circleci.com/gh/facebookresearch/mmf.svg?style=svg"/>
  </a>
</div>

---

## This is a fork of [Facebook's MMF framework](https://github.com/facebookresearch/mmf).

## Data
The data is located in the `/data/.` directory. 
The path to the labels is: `./data/datasets/propaganda/defaults/annotations/` and to the images: `./data/datasets/propaganda/defaults/images/`

## Running the code

#### !!! Please make sure that CUDA 10.2+ is installed on you machine.
#### !!! Linux OS is recommended, otherwise you would face installation problems.
#### !!! We are using the open source Multimodal Framework developed by FacebookResearch: https://mmf.sh/

### Follow these steps to get all the code ready and running:
1. Prerequisites - generating image caption features for VisualBERT and ViLBERT:
    1. Install MMF according to the instructions here: https://mmf.readthedocs.io/en/website/notes/installation.html
    2. Install the following packages: `pip install yacs, opencv-python, cython` (if using 'pip', any package manager works)
    3. Clone vqa-maskrcnn-benchmark repository: https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark
        1. Run `python setup.py build`
        2. Run `python setup.py develop`
        3. Run the feature extraction script from the following path: `mmf/tools/scripts/features/extract_features_vmb.py`
        4. After feature extraction is done convert the features to a .mdb file with the following script: `mmf/tools/scripts/features/extract_features_vmb.py`
        5. Rename the .mdb features file to `deceptron.lmdb` and move it to `/root/.cache/torch/mmf/data/datasets/propaganda/defaults/features/`
2. Running the models - open **'Propaganda_Detection.ipynb'** and **run** the code inside.


## License

MMF is licensed under BSD license available in [LICENSE](LICENSE) file
