# Explainable-AI
## Comparing Grad-CAM and RISE: Localization and Faithfulness in CNN Explanations

Before running this project, make sure you have Grad-CAM installed. You can install it by running the following command:
'''
pip install grad-cam
'''

Since RISE runs best on CUDA due to its computational complexity, the code was transferred to Google Colab to enable GPU support, as my local computer does not have access to GPU processing.

The only change made to Main.py concerns how the other .py files are accessed. In Google Colab, to access the required files, you need to connect your Google Drive, where all project files should be stored.

You can mount your Google Drive by running the following code:
'''python
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/drive/MyDrive')
'''

After completing these steps, everything should be set up and you can proceed with running the rest of the project.


### References

@misc{jacobgilpytorchcam,
  title={PyTorch library for CAM methods},
  author={Jacob Gildenblat and contributors},
  year={2021},
  publisher={GitHub},
  howpublished={\url{https://github.com/jacobgil/pytorch-grad-cam}},
}

@inproceedings{Petsiuk2018rise,
  title = {RISE: Randomized Input Sampling for Explanation of Black-box Models},
  author = {Vitali Petsiuk and Abir Das and Kate Saenko},
  booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
  year = {2018}
}
