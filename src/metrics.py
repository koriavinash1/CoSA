import torch
import os
import numpy as np
import torchvision
from shutil import rmtree
from tqdm import tqdm
from pytorch_fid import fid_score
from sklearn.metrics import accuracy_score


@torch.no_grad()
def calculate_fid(loader, model, batch_size=16, num_batches=100, fid_dir='./tmp/CLEVR/FID' ):
    torch.cuda.empty_cache()

    real_path = os.path.join(fid_dir, 'real')
    fake_path = os.path.join(fid_dir, 'fake')

    # remove any existing files used for fid calculation and recreate directories

    if not os.path.exists(real_path):
        rmtree(real_path, ignore_errors=True)
        os.makedirs(real_path)

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
            samples = next(iter(loader))
            real_batch = samples['images']
            for k, image in enumerate(real_batch.cpu()):
                filename = str(k + batch_num * batch_size)
                torchvision.utils.save_image(image, os.path.join(real_path, f'{filename}.png'))

    # generate a bunch of fake images in results / name / fid_fake

    rmtree(fake_path, ignore_errors=True)
    os.makedirs(fake_path)

    model.eval()
   
    for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
        samples = next(iter(loader))

        image = samples['image'].to(model.device)
        recon_combined, *_ = model(image, 
                                epoch=0, 
                                batch=batch_num)
       
        for j, image in enumerate(recon_combined.cpu()):
            torchvision.utils.save_image(image, os.path.join(fake_path, f'{str(j + batch_num * batch_size)}.png'))

    return fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, model.device, 2048)





# set prediction evaluation metrics
@torch.no_grad()
def average_precision_clevr(loader, model, num_batches, distance_threshold):
    """
    Base Code from: https://github.com/google-research/google-research/blob/c3bef5045a2d4777a80a1fb333d31e03474222fb/slot_attention/utils.py#L60

    Computes the average precision for CLEVR.
    This function computes the average precision of the predictions specifically
    for the CLEVR dataset. First, we sort the predictions of the model by
    confidence (highest confidence first). Then, for each prediction we check
    whether there was a corresponding object in the input image. A prediction is
    considered a true positive if the discrete features are predicted correctly
    and the predicted position is within a certain distance from the ground truth
    object.
    Args:
    pred: np.ndarray of shape [batch_size, num_elements, dimension] containing
        predictions. The last dimension is expected to be the confidence of the
        prediction.
    attributes: np.ndarray of shape [batch_size, num_elements, dimension] containing
        ground-truth object properties.
    distance_threshold: Threshold to accept match. -1 indicates no threshold.
    Returns:
    Average precision of the predictions.
    """

    model.eval()

    pred = []; attributes = []
    for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
        samples = next(iter(loader))

        image = samples['image'].to(model.device)
        properties = samples['properties'].to(model.device)

        predictions, *_ = model(image, 
                                epoch=0, 
                                batch=batch_num)

        attributes.append(properties)
        pred.append(predictions)

    attributes = torch.cat(attributes, 0).cpu().numpy()
    pred = torch.cat(pred, 0).cpu().numpy()

    # =============================================

    [batch_size, _, element_size] = attributes.shape
    [_, predicted_elements, _] = pred.shape

    def unsorted_id_to_image(detection_id, predicted_elements):
        """Find the index of the image from the unsorted detection index."""
        return int(detection_id // predicted_elements)

    flat_size = batch_size * predicted_elements
    flat_pred = np.reshape(pred, [flat_size, element_size])
    sort_idx = np.argsort(flat_pred[:, -1], axis=0)[::-1]  # Reverse order.

    sorted_predictions = np.take_along_axis(
        flat_pred, np.expand_dims(sort_idx, axis=1), axis=0)
    idx_sorted_to_unsorted = np.take_along_axis(
        np.arange(flat_size), sort_idx, axis=0)

    def process_targets(target):
        """Unpacks the target into the CLEVR properties."""
        object_size = np.argmax(target[:3])
        shape = np.argmax(target[3:5])
        material = np.argmax(target[5:7])
        color = np.argmax(target[7:15])
        real_obj = target[15]
        coords = target[16:]
        return coords, object_size, material, shape, color, real_obj

    true_positives = np.zeros(sorted_predictions.shape[0])
    false_positives = np.zeros(sorted_predictions.shape[0])

    detection_set = set()

    for detection_id in range(sorted_predictions.shape[0]):
        # Extract the current prediction.
        current_pred = sorted_predictions[detection_id, :]
        # Find which image the prediction belongs to. Get the unsorted index from
        # the sorted one and then apply to unsorted_id_to_image function that undoes
        # the reshape.
        original_image_idx = unsorted_id_to_image(
            idx_sorted_to_unsorted[detection_id], predicted_elements)
        # Get the ground truth image.
        gt_image = attributes[original_image_idx, :, :]

        # Initialize the maximum distance and the id of the groud-truth object that
        # was found.
        best_distance = 10000
        best_id = None

        # Unpack the prediction by taking the argmax on the discrete attributes.
        (pred_coords, pred_object_size, pred_material, pred_shape, pred_color,
            _) = process_targets(current_pred)

        # Loop through all objects in the ground-truth image to check for hits.
        for target_object_id in range(gt_image.shape[0]):
            target_object = gt_image[target_object_id, :]
            # Unpack the targets taking the argmax on the discrete attributes.
            (target_coords, target_object_size, target_material, target_shape,
            target_color, target_real_obj) = process_targets(target_object)
            # Only consider real objects as matches.
            if target_real_obj:
            # For the match to be valid all attributes need to be correctly
                # predicted.
                pred_attr = [pred_object_size, pred_material, pred_shape, pred_color]
                target_attr = [
                    target_object_size, target_material, target_shape, target_color]
                match = pred_attr == target_attr
                if match:
                    # If a match was found, we check if the distance is below the
                    # specified threshold. Recall that we have rescaled the coordinates
                    # in the dataset from [-3, 3] to [0, 1], both for `target_coords` and
                    # `pred_coords`. To compare in the original scale, we thus need to
                    # multiply the distance values by 6 before applying the norm.
                    distance = np.linalg.norm((target_coords - pred_coords) * 6.)

                    # If this is the best match we've found so far we remember it.
                    if distance < best_distance:
                        best_distance = distance
                        best_id = target_object_id

        if best_distance < distance_threshold or distance_threshold == -1:
            # We have detected an object correctly within the distance confidence.
            # If this object was not detected before it's a true positive.
            if best_id is not None:
                if (original_image_idx, best_id) not in detection_set:
                    true_positives[detection_id] = 1
                    detection_set.add((original_image_idx, best_id))
                else:
                    false_positives[detection_id] = 1
            else:
                false_positives[detection_id] = 1
        else:
            false_positives[detection_id] = 1
    accumulated_fp = np.cumsum(false_positives)
    accumulated_tp = np.cumsum(true_positives)
    recall_array = accumulated_tp / np.sum(attributes[:, :, -1])
    precision_array = np.divide(accumulated_tp, (accumulated_fp + accumulated_tp))

    return compute_average_precision(
        np.array(precision_array, dtype=np.float32),
        np.array(recall_array, dtype=np.float32))



def compute_average_precision(precision, recall):
    """
    Base Code from: https://github.com/google-research/google-research/blob/c3bef5045a2d4777a80a1fb333d31e03474222fb/slot_attention/utils.py#L183
    Computation of the average precision from precision and recall arrays.
    """
    recall = recall.tolist()
    precision = precision.tolist()
    recall = [0] + recall + [1]
    precision = [0] + precision + [0]

    for i in range(len(precision) - 1, -0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices_recall = [
        i for i in range(len(recall) - 1) if recall[1:][i] != recall[:-1][i]
    ]

    average_precision = 0.
    for i in indices_recall:
        average_precision += precision[i + 1] * (recall[i + 1] - recall[i])
    return average_precision


@torch.no_grad()
def accuracy(loader, model, num_batches):
    model.eval()

    labels = []; predictions = []
    for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
        samples = next(iter(loader))

        image = samples['image'].to(model.device)
        labels = samples['label'].to(model.device)

        _, _, logits, *_ = model(image, 
                                epoch=0, 
                                batch=batch_num)

        predictions.append(torch.argmax(logits, 1))
        labels.append(labels)

    predictions = torch.cat(predictions, 0).cpu().numpy()
    labels = torch.cat(labels, 0).cpu().numpy()

    return accuracy_score(labels, predictions)
