# SegmentEM

The project includes a graphical user interface (GUI) tool designed for segmenting and analyzing neurons of interest in the circadian clock from electron microscopy images.

---

## Prerequisites

1. **Python 3**  
   The present algorithm was tested on Python 3.9, 3.10, and 3.11. We recommend creating a virtual environment with one of these versions. For example, to create a virtual environment with Python 3.9, use the following command:  
   ```bash
   conda create -n env_name anaconda python=3.9
   ```
   If you already have Anaconda with one of the tested versions of Python, simply run:  
   ```bash
   conda create -n env_name
   ```
   To activate the created virtual environment, type:  
   ```bash
   conda activate env_name
   ```  
   *(Replace `env_name` with a name of your choice.)*

2. **Tensorflow (Version 2.17.0) and Torch (Version 2.22)**  
   The installation command depends on your operating system and compute platform. To install PyTorch and Tensorflow correctly, follow the instructions at [https://pytorch.org/](https://pytorch.org/) and [https://www.tensorflow.org/](https://www.tensorflow.org/install?hl=es).

3. **Additional Libraries**  
   Additional required libraries can be installed using the `requirements.txt` file included in this repository. The installation of these libraries should be done after cloning the repository (see the Installation section below).
---

## Installation

1. Clone or download this repository:
   ```bash
   git clone [https://github.com/Imbrosci/synaptic-vesicles-detection.git](https://github.com/labCeriani/SegmentEM.git)
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## Preliminary Step Before Starting the Analysis

Measure the pixel size of the images you want to analyze. The algorithm includes a step to rescale the images, ensuring consistency in vesicle detection across varying magnifications. The experimenter must provide the pixel size (in nm) of the images. Providing incorrect pixel sizes will compromise the results.

---

## Starting the Analysis

1. Run the GUI script:
   ```bash
   python GUI.py
   ```
2. To create a new project follow these steps in the graphical user interface:
   - Select `File > New`.
   - Provide the following:
     - An experiment name.
     - The pixel size (in nm) of the images.
     - Z-scale: The ratio between the slice thickness along the z-axis and the pixel size in the xy-plane.
     - The folder where the repository has been cloned and where the project will be created. A subfolder within this path will be created to store all the resulting volumes from the analysis, and the volumes to be analyzed will be copied from the original location.
     - The SBEM volume to analyze.
     - The neuron mask, if available.
   - The volume to analyse and mask (if available) should be visualized.
---
## Run Predictions

To predict neurons, mitochondria, or vesicles, follow these steps:

- **Predict Neurons**: Select 'Predict > Neurons'. The analysis will start immediately, and the resulting neuron mask will be applied.
- **Predict Mitochondria**: Select 'Predict > Mitochondria'. A neuron mask must be present for this step. The analysis will run immediately, and the resulting mitochondria mask will be displayed in red.
- **Predict Vesicles**: Select 'Predict > Vesicles'. A neuron mask must be present for this step. It is recommended to run the analysis at a higher resolution. If you choose this option, provide the path to the corresponding high-resolution volume (note that the volume will not be copied to the project folder). The result will display the area and center of the vesicles.

---
## Analyze results

- **Checkbox**: Use the checkbox to select which predicted structures to visualize.

- **To Label**: Select `Label > Mitochondria` or `Label > Vesicles`. The segmented structures will be numerically labeled and displayed in the right section.

- **To Add Mitochondria or Vesicles**: Select `Add > Mitochondria` and click on the mitochondria you want to add. A region growing algorithm will be applied within the selected area. The labeling algorithm must be run again to update the results.

- **To Delete Mitochondria or Vesicles**: Select `Delete > Mitochondria` and click on the mitochondria you want to remove. The labeling algorithm must be run again to update the results.

- **To Undo Changes**: Select 'Undo'.

The GUI also allows for calculating metrics over the predicted structures. For mitochondria, the metrics include: volume, surface area, and a mitochondrial complexity index. For vesicles, the metrics include area and intensity.

- Click the 'Metrics' button to run the analysis. An Excel file with the results will be downloaded to the project folder.

---


## Final Notes

1. A GPU-equipped computer is strongly recommended for faster analysis.
2. The files `train_neurons_mitochondria_unet_segmentation.py` are provided for re-training the multiclass segmentation model (neurons and mitochondria) or uniclass segmentation model (only neurons or mitochondria) with your own data.
3. The algorithm performs well on images with resolution of 25 nm neuron and mitochondria segmentation and 5 nm for vesicle detection.

---

## Reporting Issues

Report issues via the issue tracker of this repository.
