import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
from huggingface_hub import from_pretrained_keras
import imageio


def resize_image(img_in,input_height,input_width):
    return cv2.resize( img_in, ( input_width,input_height) ,interpolation=cv2.INTER_NEAREST)

def write_dict_to_json(dictionary, save_path, indent=4):
    with open(save_path, "w") as outfile:  
        json.dump(dictionary, outfile, indent=indent) 

def load_json_to_dict(load_path):
    with open(load_path) as json_file:
        return json.load(json_file)


class OCRD:
    def __init__(self, img_path):
        self.image = imageio.imread(img_path)

    def do_prediction(self, model, img):
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
            binarized = self.do_prediction(model, img)

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