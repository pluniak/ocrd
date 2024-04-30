import cv2
import numpy as np
import json


def resize_image(img_in,input_height,input_width):
    return cv2.resize( img_in, ( input_width,input_height) ,interpolation=cv2.INTER_NEAREST)


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


def write_dict_to_json(dictionary, save_path, indent=4):
    with open(save_path, "w") as outfile:  
        json.dump(dictionary, outfile, indent=indent) 

def load_json_to_dict(load_path):
    with open(load_path) as json_file:
        return json.load(json_file)