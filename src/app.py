import gradio as gr
# import tensorflow as tf
# import numpy as np
# import cv2
# from PIL import Image
# from huggingface_hub import from_pretrained_keras
import sys
sys.path.append('.')
from utils.helpers import OCRD


def run_ocrd_pipeline(img_path, binarize_mode='detailed', min_pixel_sum='default', median_bounds='default', font_size='default'):
    """
    Executes the OCRD pipeline on an image from file loading to text overlay creation. This function orchestrates
    the calling of various OCRD class methods to process the image, extract and recognize text, and then overlay
    this text on the original image.

    Parameters:
        img_path (str): Path to the image file.
        binarize_mode (str): Mode to be used for image binarization. Can be 'detailed', 'fast', or 'no'.
        min_pixel_sum (int, optional): Minimum sum of pixels to consider a text line segmentation for extraction. 
            If 'default', default values are applied. Check function definition for details.
        median_bounds (tuple, optional): Bounds to filter text line segmentations based on size relative to the median. 
            If 'default', default values are applied. Check function definition for details.
        font_size (int, optional): Font size to be used in text overlay. If 'default', a default size or scaling logic is applied.
            Check function definition for details.

    Returns:
        Image: An image with overlay text, where text is extracted and recognized from the original image.

    This function handles:
    - Image binarization.
    - Text line segmentation.
    - Text line extraction and deskewing.
    - Optical character recognition on text lines.
    - Creating an image overlay with recognized text.
    """
    
    # prepare kwargs
    efadt_kwargs = {}
    if min_pixel_sum != 'default':
        efadt_kwargs['min_pixel_sum'] = min_pixel_sum
    if median_bounds != 'default':
        efadt_kwargs['median_bounds'] = median_bounds

    ctoi_kwargs = {}
    if font_size != 'default':
        ctoi_kwargs['font_size'] = font_size

    # run pipeline
    ocrd = OCRD(img_path)
    print('\nBinarizing image...\n')
    binarized = ocrd.binarize_image(ocrd.image, binarize_mode)
    print('\nSegmenting textlines...\n')
    textline_segments = ocrd.segment_textlines(binarized)
    print('\nExtracting, filtering and de-skewing textlines...\n')
    image_scaled = ocrd.scale_image(ocrd.image)  # textline_segments were predicted on rescaled image
    textline_images, _ = ocrd.extract_filter_and_deskew_textlines(image_scaled, textline_segments[...,0], **efadt_kwargs)
    print('\nOCR on textlines...\n')
    textline_preds = ocrd.ocr_on_textlines(textline_images)
    print('\nCreating output overlay image...')
    img_gen = ocrd.create_text_overlay_image(textline_images, textline_preds, (image_scaled.shape[0], image_scaled.shape[1]), **ctoi_kwargs)
    print('\nJOB COMPLETED\n')

    return img_gen





title = "Welcome to the Eynollah Demo page! 👁️"
description = """
 <div class="row" style="display: flex">
  <div class="column" style="flex: 50%; font-size: 17px">
        This Space demonstrates the functionality of various Eynollah models developed at <a rel="nofollow" href="https://huggingface.co/SBB">SBB</a>.
        <br><br>
        The Eynollah suite introduces an <u>end-to-end pipeline</u> to extract layout, text lines and reading order for historic documents, where the output can be used as an input for OCR engines.
        Please keep in mind that with this demo you can just use <u>one of the 13 sub-modules</u> of the whole Eynollah system <u>at a time</u>.
  </div>
  <div class="column" style="flex: 5%; font-size: 17px"></div>
  <div class="column" style="flex: 45%; font-size: 17px">
    <strong style="font-size: 19px">Resources for more information:</strong>
        <ul>
            <li>The GitHub Repo can be found <a rel="nofollow" href="https://github.com/qurator-spk/eynollah">here</a></li>
            <li>Associated Paper: <a rel="nofollow" href="https://doi.org/10.1145/3604951.3605513">Document Layout Analysis with Deep Learning and Heuristics</a></li>
            <li>The full Eynollah pipeline can be viewed <a rel="nofollow" href="https://huggingface.co/spaces/SBB/eynollah-demo-test/blob/main/eynollah-flow.png">here</a></li>
        </ul>
    </li>
  </div>
</div> 
"""
iface = gr.Interface(
            title=title,
            description=description,
            fn=do_prediction, 
            inputs=[
                gr.Dropdown([
                    "SBB/eynollah-binarization", 
                    "SBB/eynollah-enhancement",
                    "SBB/eynollah-page-extraction", 
                    "SBB/eynollah-column-classifier",
                    "SBB/eynollah-tables",
                    "SBB/eynollah-textline",
                    "SBB/eynollah-textline_light",
                    "SBB/eynollah-main-regions",
                    "SBB/eynollah-main-regions-aug-rotation",
                    "SBB/eynollah-main-regions-aug-scaling",
                    "SBB/eynollah-main-regions-ensembled",
                    "SBB/eynollah-full-regions-1column",
                    "SBB/eynollah-full-regions-3pluscolumn"
                ], label="Select one model of the Eynollah suite 👇", info=""),
                gr.Image()
            ], 
            outputs=[
              gr.Textbox(label="Output of model (numerical or bitmap) ⬇️"),
              gr.Image()
            ],
            #examples=[['example-1.jpg']]
        )
iface.launch()