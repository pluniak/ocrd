# OCRD Project

## Description
This is a demo project for Optical Character Recognition Digitization of full text pages. It is designed for use as a Hugging Face Gradio app. 

Pipeline steps include:
  1. Image binarization
  2. Text line segmentation
  3. Text line extraction, filtering, and deskewing
  4. OCR on text lines
  5. Printing recognized text on generated image for visualization

Please note:
- The app is optimized for **English**; other languages (e.g., German) may require OCR model fine-tuning.
- When running on CPUs, a pipeline run can take up to 10 minutes.
- For lengthy waits, look at the pre-computed examples: [https://github.com/pluniak/ocrd/tree/main/data/demo_data](https://github.com/pluniak/ocrd/tree/main/data/demo_data)
- The demo is just a **first prototype**! OCR performance and computation speed should be optimized.

Use the app:
  1. Try out the the demo online at https://huggingface.co/spaces/pluniak/ocrd
  2. or follow the steps below to run the app on your local computer.

## Installation
Install Anaconda if you haven't done yet: https://docs.anaconda.com/free/anaconda/install

Then clone the ocrd repository and set up and activate the virtual environment from the CLI:
```bash
git clone https://github.com/pluniak/ocrd.git
cd ocrd
./create_conda_env_linux.sh # Linux
create_conda_env_windows.bat # Windows (using Conda terminal)
conda activate ocrd
```

## Run app locally
### Web browser
Execute script from CLI
```bash
python ./src/app.py
```
### Inside Jupyter Notebook
Open notebook and execute cells
```bash
./notebooks/app.ipynb
```

## Acknowledgements and Attributions

This project makes use of significant components from the following open-source projects:

- **eynollah**: An automated layout analysis tool for historical documents, developed as part of the QURATOR project. The eynollah tool is instrumental in facilitating the preprocessing of document images in this project. For more details on eynollah, visit their GitHub repository: [qurator-spk/eynollah](https://github.com/qurator-spk/eynollah). The tool is used under the Apache License 2.0.

- **Microsoft trocr**: I utilize Microsoft's trocr models for optical character recognition tasks. The trocr models are highly effective in recognizing text from a variety of document types. For more information on trocr and its usage, please see [Microsoft's trocr repository](https://github.com/microsoft/unilm) under the MIT License.

I appreciate the efforts of the developers and the community in providing these high-quality open-source resources.
