## Dataset and code for the paper Detecting Propaganda Techniques in Memes

### Abstract
Propaganda can be defined as a form of communication that aims to influence the opinions or the actions of people towards a specific goal; this is achieved by means of well-defined rhetorical and psychological devices. Propaganda, in the form we know it today, can be dated back to the beginning of the 17th century. However, it is with the advent of the Internet and the social media that propaganda has started to spread on a much larger scale than before, thus becoming major societal and political issue. Nowadays, a large fraction of propaganda in social media is multimodal, mixing textual with visual content. With this in mind, here we propose a new multi-label multimodal task: detecting the type of propaganda techniques used in memes. We further create and release a new corpus of 950 memes, carefully annotated with 22 propaganda techniques, which can appear in the text, in the image, or in both. Our analysis of the corpus shows that understanding both modalities together is essential for detecting these techniques. This is further confirmed in our experiments with several state-of-the-art multimodal models. 

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
        5. Rename the `.mdb` features file to `deceptron.lmdb` and move it to `/root/.cache/torch/mmf/data/datasets/propaganda/defaults/features/`
2. Running the models - open **'Propaganda_Detection.ipynb'** and **run** the code inside.


## License

MMF is licensed under BSD license available in [LICENSE](LICENSE) file
