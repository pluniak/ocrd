import cv2
import numpy as np


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
            

def do_line_segmentation(model_name, img):

    img_org = np.copy(img)
    #model = from_pretrained_keras(model_name)

    # bitmap output
    img_height_model=model.layers[len(model.layers)-1].output_shape[1]
    img_width_model=model.layers[len(model.layers)-1].output_shape[2]
    #n_classes=model.layers[len(model.layers)-1].output_shape[3]

    #img_org = np.copy(img)
    #img_height_h = img_org.shape[0]
    #img_width_h = img_org.shape[1]

    num_col_classifier = 1 # return_num_columns(img)
    width_early = img.shape[1]

    img = return_scaled_image(img, num_col_classifier, width_early, model_name)

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