#by Andrei Erofeev
import torch
import torchvision.transforms.functional as TF

device = ('cuda' if torch.cuda.is_available else 'cpu')

def resize(image, boxes, dims, return_percent_coords=False, box_mode = 'wh'):
    """
    Resize image.
    """
    # Resize image
    new_image = TF.resize(image, dims)

    # Resize bounding boxes
    if len(boxes) > 0:
        new_boxes = torch.tensor(boxes, dtype=torch.float32)
        x_scale = dims[0] / image.width
        y_scale = dims[1] / image.height
        if box_mode == 'wh':
            new_boxes[:, 2] += new_boxes[:, 0]
            new_boxes[:, 3] += new_boxes[:, 1]
        new_boxes[:, 0] = new_boxes[:, 0] * x_scale
        new_boxes[:, 2] = new_boxes[:, 2] * x_scale

        new_boxes[:, 1] = new_boxes[:, 1] * y_scale
        new_boxes[:, 3] = new_boxes[:, 3] * y_scale
    else:
        new_boxes = boxes
    #old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    #new_boxes = torch.tensor(boxes, dtype=torch.float32) / old_dims  # percent coordinates

    #if not return_percent_coords:
    #    new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
    #    new_boxes = new_boxes * new_dims

    return new_image, new_boxes

def transform(image, boxes, box_mode = 'wh'):
    """
    Apply the transformations above.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :return: transformed image, transformed bounding box coordinates
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    # Skip the following operations for evaluation/testing

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes, dims=(320, 320), box_mode=box_mode)

    # Convert PIL image to Torch tensor
    new_image = TF.to_tensor(new_image)
    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = TF.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes