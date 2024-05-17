### Purpose of file: Putting app.ipynb into a script as needed for the Gradio app ### 


import gradio as gr
import sys
sys.path.append('../src/')
from utils.helpers import OCRD



def run_ocrd_pipeline(img_path, font_size=30, binarize_mode='detailed', min_pixel_sum=30, median_bounds=(None, None), status=gr.Progress()):
    """
    Executes the OCRD pipeline on an image from file loading to text overlay creation. This function orchestrates
    the calling of various OCRD class methods to process the image, extract and recognize text, and then overlay
    this text on the original image.

    Parameters:
        img_path (str): Path to the image file.
        font_size (int, optional): Font size to be used in text overlay. If 'default', a default size or scaling logic is applied.
        binarize_mode (str): Mode to be used for image binarization. Can be 'detailed', 'fast', or 'no'.
        min_pixel_sum (int, optional): Minimum sum of pixels to consider a text line segmentation for extraction. 
            If 'default', default values are applied.
        median_bounds (tuple, optional): Bounds to filter text line segmentations based on size relative to the median. 
            If 'default', default values are applied.

    Returns:
        Image: An image with overlay text, where text is extracted and recognized from the original image.

    This function handles:
    - Image binarization.
    - Text line segmentation.
    - Text line extraction and deskewing.
    - Optical character recognition on text lines.
    - Creating an image overlay with recognized text.
    """
    
    # convert gradio app dropdown options
    if font_size == 'small':
        font_size = 30
    if font_size == 'medium':
        font_size = 50
    if font_size == 'large':
        font_size = 70
    elif font_size == 'adjusted':
        font_size = -1

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
    status(0, desc='\nStep 1/5: Binarizing image...\n')
    binarized = ocrd.binarize_image(ocrd.image, binarize_mode)
    status(0, desc='\nStep 2/5: Segmenting textlines...\n')
    textline_segments = ocrd.segment_textlines(binarized)
    status(0, desc='\nStep 3/5: Extracting, filtering and de-skewing textlines...\n')
    image_scaled = ocrd.scale_image(ocrd.image)  # textline_segments were predicted on rescaled image
    textline_images, _ = ocrd.extract_filter_and_deskew_textlines(image_scaled, textline_segments[...,0], **efadt_kwargs)
    status(0, desc='\nStep 4/5: OCR on textlines...\n')
    textline_preds = ocrd.ocr_on_textlines(textline_images)
    status(0, desc='\nStep 5/5: Creating output overlay image...')
    img_gen = ocrd.create_text_overlay_image(textline_images, textline_preds, (image_scaled.shape[0], image_scaled.shape[1]), **ctoi_kwargs)
    status(1, desc='\nJOB COMPLETED\n')

    return img_gen


demo_data = [
    ['../src/demo_data/act_image.jpg', None],
    ['../src/demo_data/newjersey2_image.jpg', None],
    ['../src/demo_data/washington_image.jpg', None]
]

description = """<ul>
                    <li>This interactive demo showcases an 'Optical Character Recognition Digitization' pipeline that processes images to recognize text.</li>
                    <li>Steps include:
                        <ol>
                            <li>Image binarization</li>
                            <li>Text line segmentation</li>
                            <li>Text line extraction, filtering and deskewing</li>
                            <li>OCR on textlines</li>
                            <li>Printing recognized text on generated image for visualization</li>
                        </ol>
                    </li>
                    <li>Optimized for <b>English</b>; other languages (e.g., German) may require OCR model fine-tuning.</li>
                    <li>Uses free CPU-based compute, which is rather <b>slow</b>. A pipeline run will take up to 10 minutes.</li> 
                    <li>For lengthy waits, click on the pre-computed examples below or look at example results at: <a href='https://github.com/pluniak/ocrd/tree/main/src/demo_data'>https://github.com/pluniak/ocrd/tree/main/src/demo_data</a></li>
                    <li>The demo is based on code from my GitHub repository: <a href='https://github.com/pluniak/ocrd'>https://github.com/pluniak/ocrd</a></li>
                    <li>Note: The demo is just a <b>first prototype</b>! OCR performance and computation speed should be optimized.</li>
                </ul>"""

iface = gr.Interface(fn = run_ocrd_pipeline,
                     title="OCRD Pipeline",
                     description=description,
                     inputs=[
                         gr.Image(type='filepath', label='Input image'), 
                         gr.Dropdown(choices=['small', 'medium', 'large', 'adjusted'], label='Output image font size', value='small', 
                                     info='"adjusted" will try to mimic font sizes from the input image')
                     ],
                     outputs=gr.Image(label='Output image: overlay with recognized text', type='pil', format='jpeg'),
                     examples=demo_data)
iface.launch()