import json
from pathlib import Path
import clip
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


from utils import collate_fn, device


@torch.no_grad()
def cirr_generate_test_submission_file(img2text, clip_model, index_features, index_names, relative_test_dataset,
                                       submission_name, save_dir):
    """
    Generate the test submission file for the CIRR dataset given the pseudo tokens
    """

    # Load the model
    clip_model = clip_model.float().eval().requires_grad_(False)
    img2text.eval()

    # Get the predictions dicts
    pairid_to_retrieved_images, pairid_to_group_retrieved_images = \
        cirr_generate_test_dicts(relative_test_dataset, clip_model, img2text, index_features, index_names)

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_retrieved_images)
    group_submission.update(pairid_to_group_retrieved_images)

    submissions_folder_path = Path(save_dir + "/cirr")
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    with open(submissions_folder_path / f"{submission_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"subset_{submission_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def cirr_generate_test_dicts(relative_test_dataset, clip_model, img2text, index_features, index_names):
    """
    Generate the test submission dicts for the CIRR dataset given the pseudo tokens
    """

    # Get the predicted features
    predicted_features, reference_names, pairs_id, group_members = \
        cirr_generate_test_predictions(clip_model, img2text, relative_test_dataset)

    print(f"Compute CIRR prediction dicts")

    # Normalize the index features
    index_features = index_features.to(device)
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_retrieved_images = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                                  zip(pairs_id, sorted_index_names)}
    pairid_to_group_retrieved_images = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                        zip(pairs_id, sorted_group_names)}

    return pairid_to_retrieved_images, pairid_to_group_retrieved_images


def cirr_generate_test_predictions(clip_model, img2text, relative_test_dataset):
    """
    Generate the test prediction features for the CIRR dataset given the pseudo tokens
    """

    # Create the test dataloader
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=10,
                                      pin_memory=False)

    predicted_features_list = []
    reference_names_list = []
    pair_id_list = []
    group_members_list = []

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        reference_names = batch['reference_name']
        pairs_id = batch['pair_id']
        relative_captions = batch['relative_caption']
        group_members = batch['group_members']
        reference_image = batch['reference_image'].cuda()

        group_members = np.array(group_members).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            text_img_feature = img2text.img_to_text(reference_image, clip_model, relative_captions)

        predicted_feature = F.normalize(text_img_feature)

        predicted_features_list.append(predicted_feature)
        reference_names_list.extend(reference_names)
        pair_id_list.extend(pairs_id)
        group_members_list.extend(group_members)

    predicted_features = torch.vstack(predicted_features_list)

    return predicted_features, reference_names_list, pair_id_list, group_members_list


@torch.no_grad()
def circo_generate_test_submission_file(img2text, clip_model, index_features, index_names, relative_test_dataset,
                                       submission_name, save_dir):
    """
    Generate the test submission file for the CIRCO dataset given the pseudo tokens
    """

    # Load the model
    clip_model = clip_model.float().eval().requires_grad_(False)
    img2text.eval()

    # Get the predictions dict
    queryid_to_retrieved_images = circo_generate_test_dict(relative_test_dataset, clip_model, img2text, index_features, index_names)
    submissions_folder_path = Path(save_dir + "/circo")
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    with open(submissions_folder_path / f"{submission_name}.json", 'w+') as file:
        json.dump(queryid_to_retrieved_images, file, sort_keys=True)


def circo_generate_test_predictions(clip_model, img2text, relative_test_dataset):
    """
    Generate the test prediction features for the CIRCO dataset given the pseudo tokens
    """

    # Create the test dataloader
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=10,
                                      pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    query_ids_list = []

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        reference_names = batch['reference_name']
        relative_captions = batch['relative_caption']
        query_ids = batch['query_id']
        reference_image = batch['reference_image'].cuda()

        # Compute the predicted features
        with torch.no_grad():
            text_img_feature = img2text.img_to_text(reference_image, clip_model, relative_captions)

        predicted_feature = F.normalize(text_img_feature)

        predicted_features_list.append(predicted_feature)
        query_ids_list.extend(query_ids)

    predicted_features = torch.vstack(predicted_features_list)
    return predicted_features, query_ids_list


def circo_generate_test_dict(relative_test_dataset, clip_model, img2text, index_features, index_names):
    """
    Generate the test submission dicts for the CIRCO dataset given the pseudo tokens
    """

    # Get the predicted features
    predicted_features, query_ids = circo_generate_test_predictions(clip_model, img2text, relative_test_dataset)

    # Normalize the features
    index_features = index_features.float().to(device)
    index_features = F.normalize(index_features, dim=-1)

    # Compute the similarity
    similarity = predicted_features @ index_features.T
    sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Generate prediction dicts
    queryid_to_retrieved_images = {query_id: query_sorted_names[:50].tolist() for
                                   (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)}

    return queryid_to_retrieved_images


