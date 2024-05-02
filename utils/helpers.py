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


def return_scaled_image(img, num_col, width_early):
    if num_col == 1 and width_early < 1100:
        img_w_new = 2000
        img_h_new = int(img.shape[0] / float(img.shape[1]) * 2000)
    elif num_col == 1 and width_early >= 2500:
        img_w_new = 2000
        img_h_new = int(img.shape[0] / float(img.shape[1]) * 2000)
    elif num_col == 1 and width_early >= 1100 and width_early < 2500:
        img_w_new = width_early
        img_h_new = int(img.shape[0] / float(img.shape[1]) * width_early)
    img_new = resize_image(img, img_h_new, img_w_new)

    return img_new


def visualize_model_output(prediction, img, model_name):
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
    
    added_image = cv2.addWeighted(img,0.5,output,0.1,0)
        
    return added_image


def binarize_image(img, binarize_mode):
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

    Example usage:
    ```python
    img = cv2.imread('path_to_image.jpg')
    binarized_img = binarize_image(img, 'fast')
    plt.imshow(binarized_img)
    ```

    Note:
    - The 'detailed' mode requires a pre-trained model from huggingface_hub.
    - This function depends on OpenCV (cv2) for image processing in 'fast' mode.
    """

    if binarize_mode == 'detailed':
        model_name = "SBB/eynollah-binarization"
        model = from_pretrained_keras(model_name)
        binarized = do_prediction(model, img)

        # Convert from mask to image (letters black)
        binarized = binarized.astype(np.int8)
        binarized = -binarized + 1
        binarized = (binarized * 255).astype(np.uint8)

    elif binarize_mode == 'fast':
        binarized = return_scaled_image(img, 1, img.shape[1])
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


def do_prediction(model, img):
    #img_org = np.copy(img)       

    # bitmap output
    img_height_model=model.layers[len(model.layers)-1].output_shape[1]
    img_width_model=model.layers[len(model.layers)-1].output_shape[2]
    #n_classes=model.layers[len(model.layers)-1].output_shape[3]

    #img_org = np.copy(img)
    #img_height_h = img_org.shape[0]
    #img_width_h = img_org.shape[1]

    num_col_classifier = 1 # only one column text pages for demo
    width_early = img.shape[1]

    img = return_scaled_image(img, num_col_classifier, width_early)

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
    #mask_true = np.zeros((img_h, img_w))
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
            

# def extract_textlines(img, textline_mask):
#     """
#     Extracts textline segments from the image using a mask, with filters to remove unlikely text boxes.

#     Args:
#     img: cv2.Image - Original image from which textlines are to be extracted.
#     textline_mask: array-like - Binary image with textline segments labeled.

#     Returns:
#     textline_images: List of tuples, each containing an image of an extracted textline and its position.
#     """
#     num_labels, labels_im = cv2.connectedComponents(textline_mask)
#     bounding_boxes = []

#     for label in range(1, num_labels):
#         mask = np.where(labels_im == label, 255, 0).astype('uint8')
#         x, y, w, h = cv2.boundingRect(mask)
#         bounding_boxes.append((x, y, w, h))

#     # Calculate median width and height
#     if bounding_boxes:
#         median_width = np.median([w for _, _, w, _ in bounding_boxes])
#         median_height = np.median([h for _, _, _, h in bounding_boxes])

#         # Filter boxes that are too small or improperly shaped
#         textline_images = []
#         for x, y, w, h in bounding_boxes:
#             if w > 0.5 * median_width and h > 0.5 * median_height and w > h:
#                 cropped_image = img[y:y+h, x:x+w]
#                 textline_images.append((cropped_image, x, y, w, h))

#     return textline_images


def extract_and_deskew_textlines(img, textline_mask, size_filter=True):

    """
    Extracts and deskews textlines from an image based on the provided textline mask. It calculates
    the minimum area rectangle for contours, performs perspective transformations to deskew the text,
    and handles potential rotations to ensure text lines are horizontal. 

    Args:
    img (3D np.array): The original image from which to extract textlines.
    textline_mask (2D np.array): A binary mask where textlines have been segmented.
    filter

    Returns:
    list: A list of tuples for each textline, containing:
          - de-skewed image (np.array) as well as center, height and width of the original bounding box (tuple)
    """
    
    num_labels, labels_im = cv2.connectedComponents(textline_mask)

    # Thresholds for filtering
    MIN_PIXEL_SUM = 30 # absolute filtering
    MEDIAN_LOWER_BOUND = .5 # relative filtering
    MEDIAN_HIGHER_BOUND = 5 # relative filtering

    # Gather masks and their sizes
    cc_sizes = []
    masks = []
    for label in range(1, num_labels):
        mask = np.where(labels_im == label, 1, 0)
        if mask.sum() > MIN_PIXEL_SUM: # dismiss mini segmentations to avoid skewing of median
            cc_sizes.append(mask.sum())
            masks.append(mask)

    # filter masks by size in relation to median; then calculate contours and min area bounding box for remaining ones
    rectangles = []
    median = np.median(cc_sizes)
    for mask in masks:
        if mask.sum() > median*MEDIAN_LOWER_BOUND and mask.sum() < median*MEDIAN_HIGHER_BOUND:
            mask = (mask*255).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rect = cv2.minAreaRect(contours[0])
            rectangles.append(rect)

    # Transform (rotated) bounding boxes to horizontal; store together with rotation angle for downstream process re-transform
    if rectangles:
        # Filter rectangles and de-skew images
        textline_images = []
        for rect in rectangles:
            width, height = rect[1]
            rotation_angle = rect[2] # unclear how to interpret and use rotation angle!!!
            
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

    return textline_images #, cc_sizes


def perform_ocr_on_textlines(textline_images):
    """
    Processes a list of image arrays using a pre-trained OCR model to extract text.

    Parameters:
    - textline_images (dict): A dictionary with a key 'array' that contains a list of image arrays. 
      Each image array represents a line of text that will be processed by the OCR model.

    Returns:
    - dict: A dictionary containing a list of extracted text under the key 'preds'.

    Description:
    The function initializes the OCR model 'microsoft/trocr-base-handwritten' using Hugging Face's 
    `pipeline` API for image-to-text conversion. Each image in the input list is converted from an 
    array format to a PIL Image, processed by the model, and the text prediction is collected.
    The progress of image processing is printed every 10 images. The final result is a dictionary 
    with the key 'preds' that holds all text predictions as a list.

    Example usage:
    ```python
    textline_images = {'array': [array_of_image1, array_of_image2, ...]}
    extracted_texts = perform_ocr_on_textlines(textline_images)
    print(extracted_texts['preds'])
    ```

    Note:
    - This function requires the `transformers` library from Hugging Face and PIL library to run.
    - Ensure that the model 'microsoft/trocr-base-handwritten' is correctly loaded and the 
      `transformers` library is updated to use the pipeline.
    """
    
    model_name = "microsoft/trocr-base-handwritten"
    pipe = pipeline("image-to-text", model=model_name)

    # Model inference
    textline_preds = []
    len_array = len(textline_images['array'])
    for i, textline in enumerate(textline_images['array'][:]):
        if i % 10 == 1:
            print(f'Processing image no. {i} of {len_array}')
        textline = Image.fromarray(textline)
        textline_preds.append(pipe(textline))

    # Convert to dict
    preds = [pred[0]['generated_text'] for pred in textline_preds]
    textline_preds_dict = {'preds': preds}

    return textline_preds_dict


def adjust_font_size(draw, text, box_width):
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
            font_size = int(font_size - 10)
            return ImageFont.load_default(font_size)  # Return the last fitting size
    return font  # Return max size if none exceeded the box


def create_text_overlay_image(textline_images, textline_preds, img_shape, font_size=-1):
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

    Example usage:
    ```python
    img_overlay = create_text_overlay_image(textline_images, textline_preds, (500, 800))
    img_overlay.show()
    ```
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
            font = adjust_font_size(draw, text, width)
        else:
            font = ImageFont.load_default(font_size)
        draw.text((left_x, center_y), text, fill=(0, 0, 0), font=font, align='left')

    return img_gen