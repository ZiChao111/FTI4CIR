import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FashionIQDataset, CIRRDataset, CIRCODataset

from utils import extract_index_features, collate_fn, device, targetpad_transform


# Fashion_iq
@torch.no_grad()
def compute_fiq_val_metrics(relative_val_dataset: FashionIQDataset, clip_model: CLIP, img2text,
                            index_features: torch.tensor,
                            index_names: List[str]):
    # Generate predictions
    compose_features, target_names = generate_fiq_val_predictions(clip_model,
                                                                  img2text,
                                                                  relative_val_dataset
                                                                  )

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Move the features to the device
    index_features = index_features.to(device)
    compose_features = compose_features.to(device)

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - compose_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


@torch.no_grad()
def generate_fiq_val_predictions(clip_model: CLIP, img2text, relative_val_dataset: FashionIQDataset):
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=8, pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    predicted_features_list = []
    target_names_list = []
    img2text = img2text.cuda()
    # Compute features
    for batch in tqdm(relative_val_loader):
        reference_image = batch['reference_image']
        target_names = batch['target_name']
        relative_captions = batch['relative_captions']

        # Concatenate the captions in a deterministic way
        flattened_captions: list = np.array(relative_captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]

        input_captions_reversed = [
            f"{flattened_captions[i + 1].strip('.?, ')} and {flattened_captions[i].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]

        reference_image = reference_image.cuda()
        # Compute the predicted features
        with torch.no_grad():
            text_img_feature = img2text.img_to_text(reference_image, clip_model, input_captions)
            text_img_feature_reversed = img2text.img_to_text(reference_image, clip_model, input_captions_reversed)

        predicted_features = F.normalize((F.normalize(text_img_feature) + F.normalize(text_img_feature_reversed)) / 2)
        predicted_features_list.append(predicted_features)
        target_names_list.extend(target_names)
    predicted_features = torch.vstack(predicted_features_list)
    return predicted_features, target_names_list


@torch.no_grad()
def fashioniq_val_retrieval(datapath, clip_model, dress_type, img2text, preprocess: callable):
    # Load the model
    clip_model = clip_model.float().eval().requires_grad_(False)
    img2text.eval()
    # Define the validation datasets and extract the index features
    classic_val_dataset = FashionIQDataset(datapath, 'val', [dress_type], 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
    relative_val_dataset = FashionIQDataset(datapath, 'val', [dress_type], 'relative', preprocess)

    return compute_fiq_val_metrics(relative_val_dataset, clip_model, img2text, index_features, index_names)


# CIRR
@torch.no_grad()
def cirr_compute_val_metrics(relative_val_dataset, clip_model, img2text, index_features, index_names):
    """
    Compute the retrieval metrics on the CIRR validation set given the dataset, pseudo tokens and the reference names
    """

    # Generate the predicted features
    predicted_features, reference_names, target_names, group_members = cirr_generate_val_predictions(clip_model,
                                                                                                     img2text,
                                                                                                     relative_val_dataset)

    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return {
        'cirr_recall_at1': recall_at1,
        'cirr_recall_at5': recall_at5,
        'cirr_recall_at10': recall_at10,
        'cirr_recall_at50': recall_at50,
        'cirr_group_recall_at1': group_recall_at1,
        'cirr_group_recall_at2': group_recall_at2,
        'cirr_group_recall_at3': group_recall_at3,
    }


@torch.no_grad()
def cirr_generate_val_predictions(clip_model, img2text, relative_val_dataset):
    # Define the dataloader
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=8,
                                     pin_memory=False, collate_fn=collate_fn)
    predicted_features_list = []
    target_names_list = []
    group_members_list = []
    reference_names_list = []

    for batch in tqdm(relative_val_loader):
        reference_names = batch['reference_name']
        target_names = batch['target_name']
        relative_captions = batch['relative_caption']
        group_members = batch['group_members']
        reference_image = batch['reference_image'].cuda()

        group_members = np.array(group_members).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            text_img_feature = img2text.img_to_text(reference_image, clip_model, relative_captions)

        predicted_features = F.normalize(text_img_feature)

        predicted_features_list.append(predicted_features)
        target_names_list.extend(target_names)
        group_members_list.extend(group_members)
        reference_names_list.extend(reference_names)

    predicted_features = torch.vstack(predicted_features_list)

    return predicted_features, reference_names_list, target_names_list, group_members_list


@torch.no_grad()
def cirr_val_retrieval(datapath, clip_model, img2text, preprocess):
    """
    Compute the retrieval metrics on the CIRR validation set given the pseudo tokens and the reference names
    """
    # Load the model
    clip_model = clip_model.float().eval().requires_grad_(False)
    img2text.eval()
    # Extract the index features
    classic_val_dataset = CIRRDataset(datapath, 'val', 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)

    # Define the relative validation dataset
    relative_val_dataset = CIRRDataset(datapath, 'val', 'relative', preprocess)

    return cirr_compute_val_metrics(relative_val_dataset, clip_model, img2text, index_features, index_names)


# CIRCO
@torch.no_grad()
def circo_compute_val_metrics(relative_val_dataset, clip_model, img2text, index_features, index_names):
    """
    Compute the retrieval metrics on the CIRCO validation set given the dataset, pseudo tokens and the reference names
    """

    # Generate the predicted features
    predicted_features, target_names, gts_img_ids = circo_generate_val_predictions(clip_model, img2text,
                                                                                   relative_val_dataset)
    ap_at5 = []
    ap_at10 = []
    ap_at25 = []
    ap_at50 = []

    recall_at5 = []
    recall_at10 = []
    recall_at25 = []
    recall_at50 = []

    # Move the features to the device
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Normalize the features
    index_features = F.normalize(index_features.float())

    for predicted_feature, target_name, gt_img_ids in tqdm(zip(predicted_features, target_names, gts_img_ids)):
        gt_img_ids = np.array(gt_img_ids)[
            np.array(gt_img_ids) != '']  # remove trailing empty strings added for collate_fn
        similarity = predicted_feature @ index_features.T
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        map_labels = torch.tensor(np.isin(sorted_index_names, gt_img_ids), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
        precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position

        ap_at5.append(float(torch.sum(precisions[:5]) / min(len(gt_img_ids), 5)))
        ap_at10.append(float(torch.sum(precisions[:10]) / min(len(gt_img_ids), 10)))
        ap_at25.append(float(torch.sum(precisions[:25]) / min(len(gt_img_ids), 25)))
        ap_at50.append(float(torch.sum(precisions[:50]) / min(len(gt_img_ids), 50)))

        assert target_name == gt_img_ids[0], f"Target name not in GTs {target_name} {gt_img_ids}"
        single_gt_labels = torch.tensor(sorted_index_names == target_name)
        recall_at5.append(float(torch.sum(single_gt_labels[:5])))
        recall_at10.append(float(torch.sum(single_gt_labels[:10])))
        recall_at25.append(float(torch.sum(single_gt_labels[:25])))
        recall_at50.append(float(torch.sum(single_gt_labels[:50])))

    map_at5 = np.mean(ap_at5) * 100
    map_at10 = np.mean(ap_at10) * 100
    map_at25 = np.mean(ap_at25) * 100
    map_at50 = np.mean(ap_at50) * 100
    recall_at5 = np.mean(recall_at5) * 100
    recall_at10 = np.mean(recall_at10) * 100
    recall_at25 = np.mean(recall_at25) * 100
    recall_at50 = np.mean(recall_at50) * 100

    return {
        'circo_map_at5': map_at5,
        'circo_map_at10': map_at10,
        'circo_map_at25': map_at25,
        'circo_map_at50': map_at50,
        'circo_recall_at5': recall_at5,
        'circo_recall_at10': recall_at10,
        'circo_recall_at25': recall_at25,
        'circo_recall_at50': recall_at50,
    }


@torch.no_grad()
def circo_generate_val_predictions(clip_model, img2text, relative_val_dataset):
    """
    Generates features predictions for the validation set of CIRCO
    """
    # Create the data loader
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=8,
                                     pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    target_names_list = []
    gts_img_ids_list = []

    # Compute the features
    for batch in tqdm(relative_val_loader):
        reference_names = batch['reference_name']
        target_names = batch['target_name']
        relative_captions = batch['relative_caption']
        gt_img_ids = batch['gt_img_ids']
        reference_image = batch['reference_image'].cuda()

        gt_img_ids = np.array(gt_img_ids).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            text_img_feature = img2text.img_to_text(reference_image, clip_model, relative_captions)

        predicted_feature = F.normalize(text_img_feature)

        predicted_features_list.append(predicted_feature)
        target_names_list.extend(target_names)
        gts_img_ids_list.extend(gt_img_ids)

    predicted_features = torch.vstack(predicted_features_list)

    return predicted_features, target_names_list, gts_img_ids_list


@torch.no_grad()
def circo_val_retrieval(datapath, clip_model, img2text, preprocess):
    """
    Compute the retrieval metrics on the CIRCO validation set given the pseudo tokens and the reference names
    """
    # Load the model
    clip_model = clip_model.float().eval()
    img2text.eval()

    # Extract the index features
    classic_val_dataset = CIRCODataset(datapath, 'val', 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)

    # Define the relative validation dataset
    relative_val_dataset = CIRCODataset(datapath, 'val', 'relative', preprocess)

    return circo_compute_val_metrics(relative_val_dataset, clip_model, img2text, index_features, index_names)


def main():
    global preprocess
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['cirr', 'fashioniq', 'circo'], default="fashioniq",
                        help="Dataset to use")
    # test dataset path
    parser.add_argument('--Fashion_IQ_path', type=str, default="")
    parser.add_argument('--CIRR_path', type=str, default="")
    parser.add_argument('--CIRCO_path', type=str, default="")

    parser.add_argument('--preprocess_type', type=str, default="targetpad")
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument('--clip_model_name', type=str, default='ViT-L/14')

    args = parser.parse_args()

    # load clip and preprocess
    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
    clip_model = clip_model.eval().float()

    # Define the preprocess function
    if args.preprocess_type == "clip":
        preprocess = clip_preprocess
        print('CLIP default preprocess pipeline is used for training')
    elif args.preprocess_type == "targetpad":
        preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
        print(f'Target pad  preprocess pipeline is used for training')

    # load img2text model
    img2text = torch.load(args.model_path)
    img2text.eval()

    if args.dataset.lower() == 'fashioniq':
        average_recall10_list = []
        average_recall50_list = []

        shirt_recallat10, shirt_recallat50 = fashioniq_val_retrieval(args.Fashion_IQ_path, clip_model,
                                                                     'shirt', img2text, preprocess)
        average_recall10_list.append(shirt_recallat10)
        average_recall50_list.append(shirt_recallat50)

        dress_recallat10, dress_recallat50 = fashioniq_val_retrieval(args.Fashion_IQ_path, clip_model,
                                                                     'dress', img2text, preprocess)
        average_recall10_list.append(dress_recallat10)
        average_recall50_list.append(dress_recallat50)

        toptee_recallat10, toptee_recallat50 = fashioniq_val_retrieval(args.Fashion_IQ_path, clip_model,
                                                                       'toptee', img2text, preprocess)
        average_recall10_list.append(toptee_recallat10)
        average_recall50_list.append(toptee_recallat50)

        metrics = {"dress_10": dress_recallat10, "dress_50": dress_recallat50,
                   "shirt_10": shirt_recallat10, "shirt_50": shirt_recallat50,
                   "toptee_10": toptee_recallat10, "toptee_50": toptee_recallat50,
                   "average_10": mean(average_recall10_list), "average_50": mean(average_recall50_list)}
        print(metrics)
    elif args.dataset.lower() == 'cirr':
        metrics = cirr_val_retrieval(args.CIRR_path, clip_model, img2text, preprocess)
        print(metrics)
    elif args.dataset.lower() == 'circo':
        metrics = circo_val_retrieval(args.CIRCO_path, clip_model, img2text, preprocess)
        print(metrics)
    else:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ' or CIRCO'")


if __name__ == '__main__':
    main()
