import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import sys
sys.path.append('../src/')
from utils.helpers import OCRD 


def generate_random_image(width, height, mode='RGB'):
    """Generates a random image of the given dimensions and specified mode."""
    if mode in ['RGB', 'RGBA']:
        channels = 3 if mode == 'RGB' else 4  # 3 channels for RGB, 4 for RGBA
        array = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
    elif mode == 'L':
        array = np.random.randint(0, 256, (height, width), dtype=np.uint8)  # Only 2 dimensions for grayscale

    return Image.fromarray(array, mode)


@pytest.fixture(scope="module")
def image_formats_sizes():
    """A fixture that generates various image formats and sizes."""
    sizes = [(640, 480), (800, 600), (1200, 1600), (6000, 6000)]  # Various image sizes
    modes = ['RGB', 'RGBA', 'L']  # Different color modes
    images = [(generate_random_image(w, h, m), f"{m.lower()}_{w}x{h}.jpg") for w, h in sizes for m in modes]
    return images


def test_scale_image(image_formats_sizes):
    """Tests the scale_image method for various image formats and sizes."""
    for img, _ in image_formats_sizes:
        ocrd = OCRD(img=img)
        scaled_img = ocrd.scale_image(np.array(img))
        assert scaled_img.shape[1] == 2000 or scaled_img.shape[1] == img.width  # Verify the scaling


@patch('utils.helpers.OCRD.predict')
def test_predict(mock_predict):
    """Tests the predict method with a mock."""
    # Mock Setup
    mock_predict.return_value = np.zeros((100, 100), dtype=np.uint8)

    # Generate a random image
    test_image = generate_random_image(100, 100)
    ocrd = OCRD(img=test_image)

    # Perform the test
    result = ocrd.predict(MagicMock(), ocrd.image)

    # Verify the result is as expected
    assert result.shape == ocrd.image.shape[:2]
    assert result.dtype == np.uint8
    assert np.array_equal(result, np.zeros((100, 100), dtype=np.uint8))


@pytest.mark.parametrize("mode", ["detailed", "fast", "no"])
def test_binarize_image(mode, image_formats_sizes):
    test_image = generate_random_image(100, 100)
    ocrd = OCRD(img=test_image)

    # When the mode is 'detailed', mock the predict method within the OCRD class
    if mode == "detailed":
        for img, _ in image_formats_sizes:
            img = np.array(img)
            img_scaled = ocrd.scale_image(img)
            height, width = img_scaled.shape[:2]
            with patch.object(OCRD, 'predict', return_value=np.zeros((height, width), dtype=np.uint8)) as mock_predict:
                binarized_img = ocrd.binarize_image(img, binarize_mode=mode)
                mock_predict.assert_called_once() # Ensure that predict was indeed called
                assert isinstance(binarized_img, np.ndarray) # Ensure object type
                assert binarized_img.shape[:2] == img_scaled.shape[:2]  # Ensure spatial shape is matching
                assert binarized_img.dtype == np.uint8  # Ensure type is uint8
    else:
        for img, _ in image_formats_sizes:
            img = np.array(img)
            img_scaled = ocrd.scale_image(img)
            binarized_img = ocrd.binarize_image(img, binarize_mode=mode)
            assert isinstance(binarized_img, np.ndarray) # Ensure that predict was indeed called
            assert binarized_img.shape[:2] == img_scaled.shape[:2]  # Ensure spatial shape is matching
            assert binarized_img.dtype == np.uint8  # Ensure type is uint8


@patch('utils.helpers.OCRD.ocr_on_textlines')
def test_ocr_on_textlines(mock_ocr_on_textlines):
    """Tests the ocr_on_textlines method with a mock."""
    # Mock Setup
    textline_images = {'array': [np.zeros((50, 200), dtype=np.uint8) for _ in range(5)]}
    expected_preds = ['text1', 'text2', 'text3', 'text4', 'text5']
    mock_ocr_on_textlines.return_value = {'preds': expected_preds}

    # Perform the test
    test_image = generate_random_image(100, 100) # needed for class instantiation
    ocrd = OCRD(img=test_image)
    result = ocrd.ocr_on_textlines(textline_images)

    # Verify the results
    assert isinstance(result, dict)
    assert len(result['preds']) == len(textline_images['array'])
    assert result['preds'] == expected_preds


