import os
from argparse import ArgumentParser
import clip
import torch
import generate_test_json
import pandas as pd
from dataset import FashionIQDataset, CIRRDataset, CIRCODataset

from utils import extract_index_features, device, targetpad_transform, save_dict_to_json
import evaluate


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['cirr', 'fashioniq', 'circo'], default="fashioniq",
                        help="Dataset to use")
    # test dataset path
    parser.add_argument('--Fashion_IQ_path', type=str, default="/data/FashionIQ/")
    parser.add_argument('--CIRR_path', type=str, default="/data/CIRR/")
    parser.add_argument('--CIRCO_path', type=str, default="/data/CIRCO/")

    parser.add_argument('--preprocess_type', type=str, default="targetpad")
    parser.add_argument("--save_path", type=str, default="./save_model")
    parser.add_argument("--model_path", type=str, default="./save_model/FTI4CIR_model.pth")
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

    # load dataset
    if args.dataset.lower() == 'fashioniq':
        # Extract the index features
        classic_val_dress_dataset = FashionIQDataset(args.Fashion_IQ_path, 'val', ["dress"], 'classic', preprocess)
        index_dress_features, index_dress_names = extract_index_features(classic_val_dress_dataset, clip_model)
        # Define the relative validation dataset
        relative_val_dress_dataset = FashionIQDataset(args.Fashion_IQ_path, 'val', ["dress"], 'relative', preprocess)

        # Extract the index features
        classic_val_shirt_dataset = FashionIQDataset(args.Fashion_IQ_path, 'val', ["shirt"], 'classic', preprocess)
        index_shirt_features, index_shirt_names = extract_index_features(classic_val_shirt_dataset, clip_model)
        # Define the relative validation dataset
        relative_val_shirt_dataset = FashionIQDataset(args.Fashion_IQ_path, 'val', ["shirt"], 'relative', preprocess)

        # Extract the index features
        classic_val_toptee_dataset = FashionIQDataset(args.Fashion_IQ_path, 'val', ["toptee"], 'classic', preprocess)
        index_toptee_features, index_toptee_names = extract_index_features(classic_val_toptee_dataset, clip_model)
        # Define the relative validation dataset
        relative_val_toptee_dataset = FashionIQDataset(args.Fashion_IQ_path, 'val', ["toptee"], 'relative', preprocess)

    elif args.dataset.lower() == 'cirr':
        # Extract the index features
        classic_test_dataset = CIRRDataset(args.CIRR_path, 'test', 'classic', preprocess)
        index_features, index_names = extract_index_features(classic_test_dataset, clip_model)
        relative_test_dataset = CIRRDataset(args.CIRR_path, 'test', 'relative', preprocess)
    elif args.dataset.lower() == 'circo':
        # Extract the index features
        classic_test_dataset = CIRCODataset(args.CIRCO_path, 'test', 'classic', preprocess)
        index_features, index_names = extract_index_features(classic_test_dataset, clip_model)
        # Define the relative validation dataset
        relative_test_dataset = CIRCODataset(args.CIRCO_path, 'test', 'relative', preprocess)
    else:
        raise ValueError("Dataset not supported")

    # load the model
    img2text = torch.load(args.model_path)
    img2text.eval()

    # load dataset
    if args.dataset.lower() == 'fashioniq':

        dress_10, dress_50 = evaluate.compute_fiq_val_metrics(relative_val_dress_dataset, clip_model, img2text,
                                                              index_dress_features, index_dress_names)
        shirt_10, shirt_50 = evaluate.compute_fiq_val_metrics(relative_val_shirt_dataset, clip_model, img2text,
                                                              index_shirt_features, index_shirt_names)
        toptee_10, toptee_50 = evaluate.compute_fiq_val_metrics(relative_val_toptee_dataset, clip_model, img2text,
                                                                index_toptee_features, index_toptee_names)
        metrics = {"dress_10": dress_10, "dress_50": dress_50,
                   "shirt_10": shirt_10, "shirt_50": shirt_50,
                   "toptee_10": toptee_10, "toptee_50": toptee_50}
        # Validation CSV logging
        fiq_json_path = os.path.join(args.save_path, "metrics_fashioniq.json")
        save_dict_to_json(metrics, fiq_json_path)

    elif args.dataset.lower() == 'cirr':
        # generate the test results
        generate_test_json.cirr_generate_test_submission_file(img2text, clip_model, index_features, index_names,
                                                              relative_test_dataset,
                                                              f"cirr", args.save_path)

    elif args.dataset.lower() == 'circo':
        # generate the test results
        generate_test_json.circo_generate_test_submission_file(img2text, clip_model, index_features,
                                                               index_names,
                                                               relative_test_dataset,
                                                               f"circo", args.save_path)


if __name__ == '__main__':
    main()
