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
    parser.add_argument("--exp_name", type=str, help="Experiment to evaluate")
    parser.add_argument("--dataset", type=str, choices=['cirr', 'fashioniq', 'circo'], default="fashioniq",
                        help="Dataset to use")
    # test dataset path
    parser.add_argument('--Fashion_IQ_path', type=str, default="")
    parser.add_argument('--CIRR_path', type=str, default="")
    parser.add_argument('--CIRCO_path', type=str, default="")

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--preprocess_type', type=str, default="targetpad")
    parser.add_argument("--save_path", type=str, default="")

    parser.add_argument('--epoch_num', type=int, default=80)
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

        classic_val_dataset = CIRRDataset(args.CIRR_path, 'val', 'classic', preprocess)
        index_features_val, index_names_val = extract_index_features(classic_val_dataset, clip_model)

        # Define the relative validation dataset
        relative_val_dataset = CIRRDataset(args.CIRR_path, 'val', 'relative', preprocess)

    elif args.dataset.lower() == 'circo':
        # Extract the index features
        classic_test_dataset = CIRCODataset(args.CIRCO_path, 'test', 'classic', preprocess)
        index_features, index_names = extract_index_features(classic_test_dataset, clip_model)
        # Define the relative validation dataset
        relative_test_dataset = CIRCODataset(args.CIRCO_path, 'test', 'relative', preprocess)

        # Extract the index features
        classic_val_dataset = CIRCODataset(args.CIRCO_path, 'val', 'classic', preprocess)
        index_features_val, index_names_val = extract_index_features(classic_val_dataset, clip_model)
        # Define the relative validation dataset
        relative_val_dataset = CIRCODataset(args.CIRCO_path, 'val', 'relative', preprocess)

    else:
        raise ValueError("Dataset not supported")

    best_score_fashioniq = float('-inf')
    best_score_cirr = float('-inf')
    best_score_circo = float('-inf')
    validation_log_frame_fashioniq = pd.DataFrame()
    validation_log_frame_cirr = pd.DataFrame()
    validation_log_frame_circo = pd.DataFrame()

    for epoch in range(30, args.epoch_num):
        model_path = args.save_path + f"save_model/img2text_model_{epoch}.pth"
        # 加载参数
        img2text = torch.load(model_path)
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
            val_log_dict = {'epoch': epoch}
            val_log_dict.update(metrics)
            validation_log_frame_fashioniq = pd.concat(
                [validation_log_frame_fashioniq, pd.DataFrame(data=val_log_dict, index=[0])])
            validation_log_frame_fashioniq.to_csv(str(args.save_path + '/validation_metrics_fashioniq.csv'),
                                                  index=False)
            current_score = 0
            for val in metrics.values():
                current_score = current_score + val

            is_best = current_score >= best_score_fashioniq
            if is_best:
                best_score_fashioniq = current_score
                best_json_path_combine = os.path.join(args.save_path, "metrics_best_fashioniq.json")
                save_dict_to_json(metrics, best_json_path_combine)

        elif args.dataset.lower() == 'cirr':
            args.test_dataset = "cirr"
            metrics = evaluate.cirr_compute_val_metrics(relative_val_dataset, clip_model, img2text,
                                                        index_features_val, index_names_val)

            # Validation CSV logging
            val_log_dict = {'epoch': epoch}
            val_log_dict.update(metrics)
            validation_log_frame_cirr = pd.concat(
                [validation_log_frame_cirr, pd.DataFrame(data=val_log_dict, index=[0])])
            validation_log_frame_cirr.to_csv(str(args.save_path + '/validation_metrics_cirr.csv'), index=False)

            current_score = metrics["cirr_recall_at1"] + metrics["cirr_recall_at5"] + metrics["cirr_recall_at10"] + \
                            metrics["cirr_recall_at50"]
            is_best = current_score >= best_score_cirr
            if is_best:
                best_score_cirr = current_score
                best_json_path_combine = os.path.join(args.save_path, "metrics_best_validation_cirr.json")
                save_dict_to_json(metrics, best_json_path_combine)
            # generate the test results
            generate_test_json.cirr_generate_test_submission_file(img2text, clip_model, index_features, index_names,
                                                                  relative_test_dataset,
                                                                  f"cirr_{epoch}", args.save_path)

        elif args.dataset.lower() == 'circo':
            metrics = evaluate.circo_compute_val_metrics(relative_val_dataset, clip_model, img2text,
                                                         index_features_val, index_names_val)

            # Validation CSV logging
            val_log_dict = {'epoch': epoch}
            val_log_dict.update(metrics)
            validation_log_frame_circo = pd.concat(
                [validation_log_frame_circo, pd.DataFrame(data=val_log_dict, index=[0])])
            validation_log_frame_circo.to_csv(str(args.save_path + '/validation_metrics_circo.csv'), index=False)

            current_score = metrics["circo_recall_at5"] + metrics["circo_recall_at10"] + metrics["circo_recall_at25"] + \
                            metrics[
                                "circo_recall_at50"]
            is_best = current_score >= best_score_circo
            if is_best:
                best_score_circo = current_score
                best_json_path_combine = os.path.join(args.save_path, "metrics_best_circo.json")
                save_dict_to_json(metrics, best_json_path_combine)
            # generate the test results
            generate_test_json.circo_generate_test_submission_file(img2text, clip_model, index_features,
                                                                   index_names,
                                                                   relative_test_dataset,
                                                                   f"circo_{epoch}", args.save_path)


if __name__ == '__main__':
    main()
