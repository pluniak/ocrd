import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
from huggingface_hub import from_pretrained_keras


def resize_image(img_in,input_height,input_width):
    return cv2.resize( img_in, ( input_width,input_height) ,interpolation=cv2.INTER_NEAREST)

def write_dict_to_json(dictionary, save_path, indent=4):
    with open(save_path, "w") as outfile:  
        json.dump(dictionary, outfile, indent=indent) 

def load_json_to_dict(load_path):
    with open(load_path) as json_file:
        return json.load(json_file)


class OCRD:
    """
    Optical Character Recognition and Document processing class that provides functionalities
    to preprocess images, detect text lines, perform OCR, and visualize the results.

    The class utilizes deep learning models for various tasks such as binarization and text
    line segmentation. It provides comprehensive methods to handle image scaling, prediction,
    text extraction, and overlaying recognized text on images.

    Attributes:
        image (ndarray): The image loaded into memory from the specified path. This image
                         is used across various methods within the class.

    Methods:
        __init__(img_path: str):
            Initializes the OCRD class by loading an image from the specified file path.

        scale_image(img: ndarray) -> ndarray:
            Scales an image while maintaining its aspect ratio based on predefined width thresholds.

        predict(model, img: ndarray) -> ndarray:
            Uses a specified model to make predictions on the image. This function handles
            image resizing and segmenting for model input.

        binarize_image(img: ndarray, binarize_mode: str) -> ndarray:
            Applies binarization to the image based on the specified mode ('detailed', 'fast', or 'no').

        segment_textlines(img: ndarray) -> ndarray:
            Segments text lines from the binarized image using a pretrained model.

        extract_filter_and_deskew_textlines(img: ndarray, textline_mask: ndarray, min_pixel_sum: int, median_bounds: tuple) -> (dict, ndarray):
            Processes an image to extract and correct orientation of text lines based on the provided mask.

        ocr_on_textlines(textline_images: dict) -> dict:
            Performs OCR on the extracted text lines and returns the recognized text.

        create_text_overlay_image(textline_images: dict, textline_preds: dict, img_shape: tuple, font_size: int) -> Image:
            Creates an image overlay with the recognized text annotations.

        visualize_model_output(prediction: ndarray, img: ndarray) -> ndarray:
            Visualizes the model's prediction by overlaying it onto the original image with distinct colors.
    """
    
    def __init__(self, img_path):
        self.image = np.array(Image.open(img_path))

    def scale_image(self, img):
        """
        Scales an image to have dimensions suitable for neural network inference. Scaling is based on the 
        input width parameter. The new width and height of the image are calculated to maintain the aspect 
        ratio of the original image.

        Parameters:
        - img (ndarray): The image to be scaled, expected to be in the form of a numpy array where 
                        img.shape[0] is the height and img.shape[1] is the width.

        Behavior:
        - If image width is less than 1100, the new width is set to 2000 pixels. The height is adjusted 
        to maintain the aspect ratio.
        - If image width is between 1100 (inclusive) and 2500 (exclusive), the width remains unchanged 
        and the height is adjusted to maintain the aspect ratio.
        - If image width is 2500 or more, the width is set to 2000 pixels and the height is similarly 
        adjusted to maintain the aspect ratio.

        Returns:
        - img_new (ndarray): A new image array that has been resized according to the specified rules. 
                            The aspect ratio of the original image is preserved.

        Note:
        - This function assumes that a function `resize_image(img, height, width)` is available and is 
        used to resize the image where `img` is the original image array, `height` is the new height,
        and `width` is the new width.
        """

        width_early = img.shape[1]

        if width_early < 1100:
            img_w_new = 2000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 2000)
        elif width_early >= 1100 and width_early < 2500:
            img_w_new = width_early
            img_h_new = int(img.shape[0] / float(img.shape[1]) * width_early)
        else:
            img_w_new = 2000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 2000)

        img_new = resize_image(img, img_h_new, img_w_new)

        return img_new

    def predict(self, model, img):
        """
        Processes an image to predict segmentation outputs using a given model. The function handles image resizing 
        to match the model's input dimensions and ensures that the entire image is processed by segmenting it into patches 
        that the model can handle. The prediction from these patches is then reassembled into a single output image.

        Parameters:
        - model (keras.Model): The neural network model used for predicting the image segmentation. The model should have 
                            predefined input dimensions (height and width).
        - img (ndarray): The image to be processed, represented as a numpy array.

        Returns:
        - prediction_true (ndarray): An image of the same size as the input image, containing the segmentation prediction 
                                    with each pixel labeled according to the model's output.

        Details:
        - The function first scales the input image according to the model's required input dimensions. If the scaled image 
        is smaller than the model's height or width, it is resized to match exactly.
        - The function processes the image in overlapping patches to ensure smooth transitions between the segments. These 
        patches are then processed individually through the model.
        - Predictions from these patches are then stitched together to form a complete output image, ensuring that edge 
        artifacts are minimized by carefully blending the overlapping areas.
        - This method assumes the availability of `resize_image` function for scaling and resizing 
        operations, respectively.
        - The output is converted to an 8-bit image before returning, suitable for display or further processing.
        """

        # bitmap output
        img_height_model=model.layers[len(model.layers)-1].output_shape[1]
        img_width_model=model.layers[len(model.layers)-1].output_shape[2]

        img = self.scale_image(img)

        if img.shape[0] < img_height_model:
            img = resize_image(img, img_height_model, img.shape[1])

        if img.shape[1] < img_width_model:
            img = resize_image(img, img.shape[0], img_width_model)

        marginal_of_patch_percent = 0.1
        margin = int(marginal_of_patch_percent * img_height_model)
        width_mid = img_width_model - 2 * margin
        height_mid = img_height_model - 2 * margin
        img = img / float(255.0)
        img = img.astype(np.float16)
        img_h = img.shape[0]
        img_w = img.shape[1]
        prediction_true = np.zeros((img_h, img_w, 3))
        nxf = img_w / float(width_mid)
        nyf = img_h / float(height_mid)
        nxf = int(nxf) + 1 if nxf > int(nxf) else int(nxf)
        nyf = int(nyf) + 1 if nyf > int(nyf) else int(nyf)

        for i in range(nxf):
            for j in range(nyf):
                if i == 0:
                    index_x_d = i * width_mid
                    index_x_u = index_x_d + img_width_model
                else:
                    index_x_d = i * width_mid
                    index_x_u = index_x_d + img_width_model
                if j == 0:
                    index_y_d = j * height_mid
                    index_y_u = index_y_d + img_height_model
                else:
                    index_y_d = j * height_mid
                    index_y_u = index_y_d + img_height_model
                if index_x_u > img_w:
                    index_x_u = img_w
                    index_x_d = img_w - img_width_model
                if index_y_u > img_h:
                    index_y_u = img_h
                    index_y_d = img_h - img_height_model

                img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
                label_p_pred = model.predict(img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]),
                                                verbose=0)

                seg = np.argmax(label_p_pred, axis=3)[0]
                seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)

                if i == 0 and j == 0:
                    seg_color = seg_color[0 : seg_color.shape[0] - margin, 0 : seg_color.shape[1] - margin, :]
                    prediction_true[index_y_d + 0 : index_y_u - margin, index_x_d + 0 : index_x_u - margin, :] = seg_color
                elif i == nxf - 1 and j == nyf - 1:
                    seg_color = seg_color[margin : seg_color.shape[0] - 0, margin : seg_color.shape[1] - 0, :]
                    prediction_true[index_y_d + margin : index_y_u - 0, index_x_d + margin : index_x_u - 0, :] = seg_color
                elif i == 0 and j == nyf - 1:
                    seg_color = seg_color[margin : seg_color.shape[0] - 0, 0 : seg_color.shape[1] - margin, :]
                    prediction_true[index_y_d + margin : index_y_u - 0, index_x_d + 0 : index_x_u - margin, :] = seg_color
                elif i == nxf - 1 and j == 0:
                    seg_color = seg_color[0 : seg_color.shape[0] - margin, margin : seg_color.shape[1] - 0, :]
                    prediction_true[index_y_d + 0 : index_y_u - margin, index_x_d + margin : index_x_u - 0, :] = seg_color
                elif i == 0 and j != 0 and j != nyf - 1:
                    seg_color = seg_color[margin : seg_color.shape[0] - margin, 0 : seg_color.shape[1] - margin, :]
                    prediction_true[index_y_d + margin : index_y_u - margin, index_x_d + 0 : index_x_u - margin, :] = seg_color
                elif i == nxf - 1 and j != 0 and j != nyf - 1:
                    seg_color = seg_color[margin : seg_color.shape[0] - margin, margin : seg_color.shape[1] - 0, :]
                    prediction_true[index_y_d + margin : index_y_u - margin, index_x_d + margin : index_x_u - 0, :] = seg_color
                elif i != 0 and i != nxf - 1 and j == 0:
                    seg_color = seg_color[0 : seg_color.shape[0] - margin, margin : seg_color.shape[1] - margin, :]
                    prediction_true[index_y_d + 0 : index_y_u - margin, index_x_d + margin : index_x_u - margin, :] = seg_color
                elif i != 0 and i != nxf - 1 and j == nyf - 1:
                    seg_color = seg_color[margin : seg_color.shape[0] - 0, margin : seg_color.shape[1] - margin, :]
                    prediction_true[index_y_d + margin : index_y_u - 0, index_x_d + margin : index_x_u - margin, :] = seg_color
                else:
                    seg_color = seg_color[margin : seg_color.shape[0] - margin, margin : seg_color.shape[1] - margin, :]
                    prediction_true[index_y_d + margin : index_y_u - margin, index_x_d + margin : index_x_u - margin, :] = seg_color

        prediction_true = prediction_true.astype(np.uint8)

        return prediction_true

    def binarize_image(self, img, binarize_mode='detailed'):
        """
        Binarizes an image according to the specified mode.

        Parameters:
        - img (ndarray): The input image to be binarized.
        - binarize_mode (str): The mode of binarization. Can be 'detailed', 'fast', or 'no'.
        - 'detailed': Uses a pre-trained deep learning model for binarization.
        - 'fast': Uses OpenCV for a quicker, threshold-based binarization.
        - 'no': Returns a copy of the original image.

        Returns:
        - ndarray: The binarized image.

        Raises:
        - ValueError: If an invalid binarize_mode is provided.

        Description:
        Depending on the 'binarize_mode', the function processes the image differently:
        - For 'detailed' mode, it loads a specific model and performs prediction to binarize the image.
        - For 'fast' mode, it quickly converts the image to grayscale and applies a threshold.
        - For 'no' mode, it simply returns the original image unchanged.
        If an unsupported mode is provided, the function raises a ValueError.

        Note:
        - The 'detailed' mode requires a pre-trained model from huggingface_hub.
        - This function depends on OpenCV (cv2) for image processing in 'fast' mode.
        """

        if binarize_mode == 'detailed':
            model_name = "SBB/eynollah-binarization"
            model = from_pretrained_keras(model_name)
            binarized = self.predict(model, img)

            # Convert from mask to image (letters black)
            binarized = binarized.astype(np.int8)
            binarized = -binarized + 1
            binarized = (binarized * 255).astype(np.uint8)

        elif binarize_mode == 'fast':
            binarized = self.scale_image(img, self.image)
            binarized = cv2.cvtColor(binarized, cv2.COLOR_BGR2GRAY)
            _, binarized = cv2.threshold(binarized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            binarized = np.repeat(binarized[:, :, np.newaxis], 3, axis=2)

        elif binarize_mode == 'no':
            binarized = img.copy()

        else:
            accepted_values = ['detailed', 'fast', 'no']
            raise ValueError(f"Invalid value provided: {binarize_mode}. Accepted values are: {accepted_values}")

        binarized = binarized.astype(np.uint8)

        return binarized
    

    def segment_textlines(self, img):
        '''
        ADD DOCUMENTATION!
        '''
        model_name = "SBB/eynollah-textline"
        model = from_pretrained_keras(model_name)
        textline_segments = self.predict(model, img)

        return textline_segments
    

    def extract_filter_and_deskew_textlines(self, img, textline_mask, min_pixel_sum=30, median_bounds=(.5, 20)):

        """
        Extracts and deskews text lines from an image based on a provided textline mask. This function identifies
        text lines, filters out those that do not meet size criteria, calculates their minimum area rectangles,
        performs perspective transformations to deskew each text line, and handles potential rotations to ensure
        text lines are presented horizontally.

        Parameters:
        - img (numpy.ndarray): The original image from which to extract and deskew text lines. It should be a 3D array.
        - textline_mask (numpy.ndarray): A binary mask where text lines have been segmented. It should be a 2D array.
        - min_pixel_sum (int, optional): The minimum number of pixels (area) a connected component must have to be considered
        a valid text line. If None, no filtering is applied.
        - median_bounds (tuple, optional): A tuple representing the lower and upper bounds as multipliers for filtering
        text lines based on the median size of identified text lines. If None, no filtering is applied.

        Returns:
        - tuple: 
            - dict: A dictionary containing lists of the extracted and deskewed text line images along with their
            metadata (center, left side, height, width, and rotation angle of the bounding box).
            - numpy.ndarray: An image visualization of the filtered text line mask for debugging or analysis.

        Description:
        The function first uses connected components to identify potential text lines from the mask. It filters these
        based on absolute size (min_pixel_sum) and relative size (median_bounds). For each valid text line, it computes
        a minimum area rectangle, extracts and deskews the bounded region. This includes rotating the text line if it
        is detected as vertical (taller than wide). Finally, it aggregates the results and provides an image for
        visualization of the text lines retained after filtering.

        Notes:
        - This function assumes the textline_mask is properly segmented and binary (0s for background, 255 for text lines).
        - Errors in perspective transformation due to incorrect contour extraction or bounding box calculations are handled
        gracefully, reporting the error but continuing with other text lines.
        """
        
        num_labels, labels_im = cv2.connectedComponents(textline_mask)

        # Thresholds for filtering
        MIN_PIXEL_SUM = min_pixel_sum # absolute filtering
        MEDIAN_LOWER_BOUND = median_bounds[0] # relative filtering
        MEDIAN_UPPER_BOUND = median_bounds[1] # relative filtering

        # Gather masks and their sizes
        cc_sizes = []
        masks = []
        labels_im_filtered = labels_im > 0 # for visualizing filtering result
        for label in range(1, num_labels): # ignore background class
            mask = np.where(labels_im == label, True, False)
            if MIN_PIXEL_SUM is None:
                is_above_min_pixel_sum = True
            else:
                is_above_min_pixel_sum = mask.sum() > MIN_PIXEL_SUM
            if is_above_min_pixel_sum: # dismiss mini segmentations to avoid skewing of median
                cc_sizes.append(mask.sum())
                masks.append(mask)

        # filter masks by size in relation to median; then calculate contours and min area bounding box for remaining ones
        rectangles = []
        median = np.median(cc_sizes)
        for mask in masks:
            mask_sum = mask.sum()
            if MEDIAN_LOWER_BOUND is None:
                is_above_lower_media_bound = True
            else:
                is_above_lower_media_bound = mask_sum > median*MEDIAN_LOWER_BOUND
            if MEDIAN_UPPER_BOUND is None:
                is_below_upper_median_bound = True
            else:
                is_below_upper_median_bound = mask_sum < median*MEDIAN_UPPER_BOUND
            if is_above_lower_media_bound and is_below_upper_median_bound:
                labels_im_filtered[mask > 0] = False
                mask = (mask*255).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rect = cv2.minAreaRect(contours[0])
                if np.prod(rect[1]) > 0: # filter out if height or width = 0
                    rectangles.append(rect)

        # Transform (rotated) bounding boxes to horizontal; store together with rotation angle for downstream process re-transform
        if rectangles:
            # Filter rectangles and de-skew images
            textline_images = []
            for rect in rectangles:
                width, height = rect[1]
                rotation_angle = rect[2] # clarify how to interpret and use rotation angle!
                
                # Convert dimensions to integer and ensure they are > 0
                width = int(width)
                height = int(height)

                # get source and destination points for image transform
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                src_pts = box.astype("float32")
                dst_pts = np.array([[0, height-1],
                                    [0, 0],
                                    [width-1, 0],
                                    [width-1, height-1]], dtype="float32")
                
                try:
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(img, M, (width, height))
                    # Check and rotate if the text line is taller than wide
                    if height > width:
                        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                        temp = height
                        height = width
                        width = temp
                        rotation_angle = 90-rotation_angle
                    center = rect[0]
                    left = center[0] - width//2
                    textline_images.append((warped, center, left, height, width, rotation_angle))
                except cv2.error as e:
                    print(f"Error with warpPerspective: {e}")

            # cast to dict
            keys = ['array', 'center', 'left', 'height', 'width', 'rotation_angle']
            textline_images = {key: [tup[i] for tup in textline_images] for i, key in enumerate(keys)}
            num_labels_filtered = len(textline_images['array'])
            labels_im_filtered = np.repeat(labels_im_filtered[:, :, np.newaxis], 3, axis=2).astype(np.uint8) # 3 color channels for plotting
            print(f'Kept {num_labels_filtered} of {num_labels} text segments after filtering.')
            print(f'All segments deleted smaller than {MIN_PIXEL_SUM} pixels (absolute min size).')
            if MEDIAN_LOWER_BOUND is not None:
                print(f'All segments deleted smaller than {median*MEDIAN_LOWER_BOUND} pixels (lower median bound).')
            if MEDIAN_UPPER_BOUND is not None:
                print(f'All segments deleted bigger than {median*MEDIAN_UPPER_BOUND} pixels (upper median bound).')
            if MEDIAN_LOWER_BOUND is not None or MEDIAN_UPPER_BOUND is not None:
                print(f'Median segment size (pixel sum) used for filtering: {int(median)}.')

        return textline_images, labels_im_filtered


    def ocr_on_textlines(self, textline_images, model_name="microsoft/trocr-base-handwritten"):
        """
        Processes a list of image arrays using a pre-trained OCR model to extract text.

        Parameters:
        - textline_images (dict): A dictionary with a key 'array' that contains a list of image arrays. 
        Each image array represents a line of text that will be processed by the OCR model.
        - model_name (str): A huggingface model trained for OCR on single text lines

        Returns:
        - dict: A dictionary containing a list of extracted text under the key 'preds'.

        Description:
        The function initializes the OCR model 'microsoft/trocr-base-handwritten' using Hugging Face's 
        `pipeline` API for image-to-text conversion. Each image in the input list is converted from an 
        array format to a PIL Image, processed by the model, and the text prediction is collected.
        The progress of image processing is printed every 10 images. The final result is a dictionary 
        with the key 'preds' that holds all text predictions as a list.

        Note:
        - This function requires the `transformers` library from Hugging Face and PIL library to run.
        - Ensure that the model 'microsoft/trocr-base-handwritten' is correctly loaded and the 
        `transformers` library is updated to use the pipeline.
        """
        
        pipe = pipeline("image-to-text", model=model_name)

        # Model inference
        textline_preds = []
        len_array = len(textline_images['array'])
        for i, textline in enumerate(textline_images['array'][:]):
            if i % 10 == 1:
                print(f'Processing textline no. {i} of {len_array}')
            textline = Image.fromarray(textline)
            textline_preds.append(pipe(textline))

        # Convert to dict
        preds = [pred[0]['generated_text'] for pred in textline_preds]
        textline_preds_dict = {'preds': preds}

        return textline_preds_dict


    def adjust_font_size(self, draw, text, box_width):
        """
        Adjusts the font size to ensure the text fits within a specified width.

        Parameters:
        - draw (ImageDraw.Draw): An instance of ImageDraw.Draw used to render the text.
        - text (str): The text string to be rendered.
        - box_width (int): The maximum width in pixels that the text should occupy.

        Returns:
        - ImageFont: A font object with a size adjusted to fit the text within the specified width.
        """

        for font_size in range(1, 200):  # Adjust the range as needed
            font = ImageFont.load_default(font_size)
            text_width = draw.textlength(text, font=font)
            if text_width > box_width:
                font_size = max(5, int(font_size - 10)) # min font size of 5
                return ImageFont.load_default(font_size)  # Return the last fitting size
        return font  # Return max size if none exceeded the box


    def create_text_overlay_image(self, textline_images, textline_preds, img_shape, font_size=-1):
        """
        Creates an image overlay with text annotations based on provided bounding box information and predictions.

        Parameters:
        - textline_images (dict): A dictionary containing the bounding box data for each text segment. 
        It should have keys 'left', 'center', 'width', and optionally 'height'. Each key should have 
        a list of values corresponding to each text segment's properties.
        - textline_preds (dict): A dictionary containing the predicted text segments. It should have 
        a key 'preds' which holds a list of text predictions corresponding to the bounding boxes in 
        textline_images.
        - img_shape (tuple): A tuple representing the shape of the image where the text is to be drawn. 
        The format should be (height, width).
        - font_size (int, optional): Specifies the font size for the text. If set to -1 (default), the font size 
        is dynamically adjusted to fit the text within its bounding box width using the `adjust_font_size` 
        function. If a specific integer is provided, it uses that size for all text segments.

        Returns:
        - Image: An image object with text drawn over a blank white background.

        Raises:
        - AssertionError: If the lengths of the lists in `textline_images` and `textline_preds['preds']` 
        do not correspond, indicating a mismatch in the number of bounding boxes and text predictions.
        """

        for key in textline_images.keys():
            assert len(textline_images[key]) == len(textline_preds['preds']), f'Length of {key} and preds doesnt correspond'

        # Create a blank white image
        img_gen = Image.new('RGB', (img_shape[1], img_shape[0]), color=(255, 255, 255))
        draw = ImageDraw.Draw(img_gen)

        # Draw each text segment within its bounding box
        for i in range(len(textline_preds['preds'])):
            left_x = textline_images['left'][i]
            center_y = textline_images['center'][i][1]
            #height = textline_images['height'][i]
            width = textline_images['width'][i]
            text = textline_preds['preds'][i]
            
            # dynamic or static text size
            if font_size==-1:
                font = self.adjust_font_size(draw, text, width)
            else:
                font = ImageFont.load_default(font_size)
            draw.text((left_x, center_y), text, fill=(0, 0, 0), font=font, align='left')

        return img_gen


    def visualize_model_output(self, prediction, img):
        """
        Visualizes the output of a model prediction by overlaying predicted classes with distinct colors onto the original image.

        Parameters:
        - prediction (ndarray): A 3D array where the first channel holds the class predictions.
        - img (ndarray): The original image to overlay predictions onto. This should be in the same dimensions or resized accordingly.

        Returns:
        - ndarray: An image where the model's predictions are overlaid on the original image using a predefined color map.

        Description:
        The function first identifies unique classes present in the prediction's first channel. Each class is assigned a specific color from a predefined dictionary `rgb_colors`. The function then creates an output image where each pixel's color corresponds to the class predicted at that location.

        The function resizes the original image to match the dimensions of the prediction if necessary. It then blends the original image and the colored prediction output using OpenCV's `addWeighted` method to produce a final image that highlights the model's predictions with transparency.

        Note:
        - This function relies on `numpy` for array manipulations and `cv2` for image processing.
        - Ensure the `rgb_colors` dictionary contains enough colors for all classes your model can predict.
        - The function assumes `prediction` array's shape is compatible with `img`.
        """

        unique_classes = np.unique(prediction[:,:,0])
        rgb_colors = {'0' : [255, 255, 255],
                        '1' : [255, 0, 0],
                        '2' : [255, 125, 0],
                        '3' : [255, 0, 125],
                        '4' : [125, 125, 125],
                        '5' : [125, 125, 0],
                        '6' : [0, 125, 255],
                        '7' : [0, 125, 0],
                        '8' : [125, 125, 125],
                        '9' : [0, 125, 255],
                        '10' : [125, 0, 125],
                        '11' : [0, 255, 0],
                        '12' : [0, 0, 255],
                        '13' : [0, 255, 255],
                        '14' : [255, 125, 125],
                        '15' : [255, 0, 255]}

        output = np.zeros(prediction.shape)

        for unq_class in unique_classes:
            rgb_class_unique = rgb_colors[str(int(unq_class))]
            output[:,:,0][prediction[:,:,0]==unq_class] = rgb_class_unique[0]
            output[:,:,1][prediction[:,:,0]==unq_class] = rgb_class_unique[1]
            output[:,:,2][prediction[:,:,0]==unq_class] = rgb_class_unique[2]

        img = resize_image(img, output.shape[0], output.shape[1])

        output = output.astype(np.int32)
        img = img.astype(np.int32)
        
        #added_image = cv2.addWeighted(img,0.5,output,0.1,0) # orig by eynollah (gives dark image output)
        added_image = cv2.addWeighted(img,0.8,output,0.2,10)
            
        return added_image