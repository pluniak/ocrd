# OCRD Project

## Description
This is a demo project for "Optical Character Recognition Digitization" of full text pages. It is designed for use as a Hugging Face Gradio app. 
<img src="./data/app_interface.jpg" alt="Input Image" width="300" style="margin: 10px;"/>

The underlying processing pipeline includes:
  1. Image binarization
  2. Text line segmentation
  3. Text line extraction, filtering, and deskewing
  4. OCR on text lines
  5. Printing recognized text on generated image for visualization

Please note:
- The app is optimized for **English**; other languages (e.g., German) may require OCR model fine-tuning.
- When running on CPUs, a pipeline run can take over 10 minutes depending on the input image.
- For lengthy waits or if the online app is down, look at the **pre-computed examples**: [https://github.com/pluniak/ocrd/tree/main/data/demo_data](https://github.com/pluniak/ocrd/tree/main/data/demo_data)
- The demo is just a **first prototype**! OCR performance and computation speed should be optimized.

Usage:
  1. **Test the the demo** online at https://huggingface.co/spaces/pluniak/ocrd
  2. or follow the steps below to install and run the app on your local computer.

## Installation
  1. Install Anaconda if you haven't done yet: https://docs.anaconda.com/free/anaconda/install
  2. Clone the repository, then set up and activate the virtual environment:
```bash
git clone https://github.com/pluniak/ocrd.git
cd ocrd
./create_conda_env_linux.sh # Linux
create_conda_env_windows.bat # Windows (using Conda terminal)
conda activate ocrd
```

## Run app locally
After activating the virtual environment, you can run the app locally as a Web Server or inside a Jupyter Notebook.
### Web Server
Execute this script from CLI:
```bash
python ./src/app.py
```
Then click on the generated local URL (usually: http://127.0.0.1:7860).
### Jupyter Notebook
Open and run this notebook:
```bash
./notebooks/app.ipynb
```

## OCRD Pipeline Example

### Input and Output Image Generated from Recognized Text

<div style="display: flex; justify-content: center; align-items: center;">
  <img src="./data/demo_data/act_image.jpg" alt="Input Image" width="300" style="margin: 10px;"/>
  <img src="./data/demo_data/act_genImg.jpg" alt="Output Image" width="300" style="margin: 10px;"/>
</div>

For more examples visit: [https://github.com/pluniak/ocrd/tree/main/data/demo_data](https://github.com/pluniak/ocrd/tree/main/data/demo_data)


## Acknowledgements and Attributions

This project makes use of significant components from the following open-source projects:

- **eynollah**: An automated layout analysis tool for historical documents, developed as part of the QURATOR project. The eynollah tool is instrumental in facilitating the preprocessing of document images in this project. For more details on eynollah, visit their GitHub repository: [qurator-spk/eynollah](https://github.com/qurator-spk/eynollah). The tool is used under the Apache License 2.0.

- **Microsoft trocr**: I utilize Microsoft's trocr models for optical character recognition tasks. The trocr models are highly effective in recognizing text from a variety of document types. For more information on trocr and its usage, please see [Microsoft's trocr repository](https://github.com/microsoft/unilm) under the MIT License.

I appreciate the efforts of the developers and the community in providing these high-quality open-source resources.
