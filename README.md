# OCRD Project

## Description
This is a demo project for Optical Character Recognition Digitization of full text pages. It is designed for use as a Hugging Face Gradio app. The demo makes use of freely available software components only. Learn more and try out the demo here: https://huggingface.co/spaces/pluniak/ocrd

## Installation
Steps to install the project, for example:
```bash
git clone https://github.com/yourusername/ocrd.git
cd ocrd
# Linux (using bash)
./create_conda_env_linux.sh 
# Windows (using Conda terminal)
create_conda_env_windows.bat
conda activate ocrd-env
```

## Run the app locally from the CLI
```bash
python ./src/app.py
```

## Acknowledgements and Attributions

This project makes use of significant components from the following open-source projects:

- **eynollah**: An automated layout analysis tool for historical documents, developed as part of the QURATOR project. The eynollah tool is instrumental in facilitating the preprocessing of document images in this project. For more details on eynollah, visit their GitHub repository: [qurator-spk/eynollah](https://github.com/qurator-spk/eynollah). The tool is used under the Apache License 2.0.

- **Microsoft trocr**: I utilize Microsoft's trocr models for optical character recognition tasks. The trocr models are highly effective in recognizing text from a variety of document types. For more information on trocr and its usage, please see [Microsoft's trocr repository](https://github.com/microsoft/unilm) under the MIT License.

I appreciate the efforts of the developers and the community in providing these high-quality open-source resources.
