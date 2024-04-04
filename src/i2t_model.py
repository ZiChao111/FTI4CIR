import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
import MHTransformer
from utils import get_img_patch_feats, contrastive_loss, token_replace


class Phi(nn.Module):
    """
    Textual Inversion Phi network.
    Takes as input the visual features of an image and outputs the pseudo-word token embedding.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class IMG2TEXT(nn.Module):
    """
        Fine-grained Textual Inversion network, we simply named IMG2TEXT.
    """

    def __init__(self, img_patch_dim, token_feat, phi_s, phi_a, num_k, hy_regLoss, temperature, tf_layer,
                 tf_head, topk, epsilon):
        super().__init__()
        self.num_k = num_k
        self.topk = topk
        self.epsilon = epsilon
        self.phi_s = phi_s
        self.phi_a = phi_a
        self.hy_regLoss = hy_regLoss
        self.temperature = temperature

        self.local_atte_fc = nn.Sequential(nn.Linear(img_patch_dim, token_feat), nn.Sigmoid())

        self.cosine_criterion = nn.CosineEmbeddingLoss()
        self.criterion_target = torch.as_tensor([1])
        self.ortho_loss = torch.nn.MSELoss()

        self.transformer = MHTransformer.Transformer(dim_self=img_patch_dim, num_heads=tf_head, dim_ref=img_patch_dim,
                                                     num_layers=tf_layer)
        self.templates = nn.Parameter(torch.randn(1, num_k, img_patch_dim))

    def get_latent_local_attributes_feats(self, featuremap):
        batch_size = featuremap.shape[0]
        feature_dim = featuremap.shape[2]

        initial_templates = self.templates.expand(batch_size, self.num_k, feature_dim)
        cat_feature = torch.cat([initial_templates, featuremap], dim=1)
        latent_local_feats = self.transformer(cat_feature, mask=None)[:, :self.num_k, :]
        latent_local_feats = self.local_atte_fc(latent_local_feats)

        return latent_local_feats

    def get_img_local_attr_feats(self, img_global_feat, image_patch_feats):
        bs = image_patch_feats.shape[0]  # [128, 257, 1024]
        latent_local_feats = self.get_latent_local_attributes_feats(image_patch_feats)

        # Preliminary screening based on attention score
        attention_weights = torch.matmul(latent_local_feats, img_global_feat.unsqueeze(dim=2)).squeeze(dim=2)
        attention_weights = F.softmax(attention_weights, dim=1)

        local_attr_num = []
        sorted_indices = torch.argsort(attention_weights, dim=1, descending=True)
        sorted_indices = sorted_indices[:, :self.topk]
        selected_local_feats = []

        for i in range(bs):
            mask = attention_weights[i] > self.epsilon
            non_indices = torch.nonzero(mask).squeeze()
            num_r = non_indices.numel() if non_indices.numel() < self.topk else self.topk
            if num_r < 1:
                num_r = 1
            # Ensure the order of attribute features
            select_indices = sorted_indices[i][:num_r]
            select_indices = torch.sort(select_indices, dim=0).values
            select_id = torch.cat((select_indices, sorted_indices[i][num_r:]), dim=0)
            local_attr_num.append(num_r)
            selected_local_feats.append(latent_local_feats[i, select_id, :])

        selected_local_feats = torch.stack(selected_local_feats, dim=0)

        return F.normalize(selected_local_feats, dim=-1), local_attr_num

    def img_to_text(self, img, clip_model, modification_text):
        # inference
        with torch.no_grad():
            # Extract global and patch features from the image
            img_global_feat = clip_model.encode_image(img)
            img_patch_feats = get_img_patch_feats(img, clip_model)

            # Get local attribute features and count
            img_local_attr_feat, local_attr_num = self.get_img_local_attr_feats(img_global_feat, img_patch_feats)
            img_subj_token = self.phi_s(img_global_feat)
            img_attr_tokens = self.phi_a(img_local_attr_feat)

            text_list = []
            bs = img_global_feat.shape[0]
            for i in range(bs):
                # Generate the composed description for each image
                text = f"a photo of * with {'* ' * local_attr_num[i]}but " + modification_text[i]
                text_list.append(text)

            # Tokenize the composed description
            text = clip.tokenize(text_list, truncate=True).cuda(non_blocking=True)
            # Replace tokens to obtain pseudo-word-based features
            pseudo_word_based_feat = token_replace(local_attr_num, text, img_subj_token, img_attr_tokens, clip_model, 0)
        return pseudo_word_based_feat

    def orthogonal_loss(self, img_salient_local_feats):
        # Orthogonal loss
        batch_size, length, dim = img_salient_local_feats.size()
        img_salient_local_feats = F.normalize(img_salient_local_feats, p=2, dim=-1)

        cosine_score = torch.matmul(img_salient_local_feats, img_salient_local_feats.permute(0, 2, 1))

        eye_matrix = torch.eye(length).unsqueeze(0).repeat(batch_size, 1, 1).to(img_salient_local_feats.device)

        return self.ortho_loss(cosine_score, eye_matrix)

    def cosine_loss(self, pseudo_word_based_feat, img_global_feat):
        cosine_loss = self.cosine_criterion(img_global_feat, pseudo_word_based_feat, self.criterion_target.cuda())
        return cosine_loss

    def semantic_regularization_loss(self, subject, attribute, pseudo_word_based_feat, img_subj_token, img_attr_tokens,
                                     clip_model, local_attr_num):
        # Generate text inputs for regularization loss
        t_both = ["a photo of " + subject[s_id] + " with " + attribute[s_id] for s_id in range(len(attribute))]
        t_subj = ["a photo of * with " + attribute[s_id] for s_id in range(len(attribute))]

        bs = len(local_attr_num)
        t_attr = []
        for i in range(bs):
            text = f"a photo of " + subject[i] + " with " + f"{'* ' * local_attr_num[i]}"
            t_attr.append(text)

        # Tokenize text inputs
        t_both = clip.tokenize(t_both, truncate=True).cuda(non_blocking=True)
        t_subj = clip.tokenize(t_subj, truncate=True).cuda(non_blocking=True)
        t_attr = clip.tokenize(t_attr, truncate=True).cuda(non_blocking=True)

        # Encode text inputs using the clip model
        with torch.no_grad():
            t_both_feat = clip_model.encode_text(t_both)

        # Replace tokens in subject and attribute text inputs
        t_subject_feat = token_replace(local_attr_num, t_subj, img_subj_token, img_attr_tokens, clip_model, 1)
        t_attribute_feat = token_replace(local_attr_num, t_attr, img_subj_token, img_attr_tokens, clip_model, 2)

        # Calculate cosine losses
        reg_attribute_loss = self.cosine_loss(t_attribute_feat, t_both_feat)
        reg_subject_loss = self.cosine_loss(t_subject_feat, t_both_feat)
        reg_both_loss = self.cosine_loss(pseudo_word_based_feat, t_both_feat)
        return reg_subject_loss + reg_attribute_loss + reg_both_loss

    def get_templates(self, local_attr_num):
        template_list = []
        bs = len(local_attr_num)
        for i in range(bs):
            template = f"a photo of * with {'* ' * local_attr_num[i]}"
            template_list.append(template)
        templates = clip.tokenize(template_list, truncate=True).cuda(non_blocking=True)
        return templates

    def getLoss(self, images, subject, attribute, clip_model):
        with torch.no_grad():
            img_global_feat = clip_model.encode_image(images)  # [batch_size, 768]
            img_patch_feats = get_img_patch_feats(images, clip_model)  # [batch_size, channel_dim, feature_dim]

        # Obtain local fine-grained features
        img_salient_local_feats, local_attr_num = self.get_img_local_attr_feats(img_global_feat, img_patch_feats)

        # Perform token mapping
        img_subj_token = self.phi_s(img_global_feat)
        img_attr_tokens = self.phi_a(img_salient_local_feats)

        templates = self.get_templates(local_attr_num)
        pseudo_word_based_feat = token_replace(local_attr_num, templates, img_subj_token, img_attr_tokens, clip_model,
                                               0)

        # compute the total loss
        img_text_loss = contrastive_loss(img_global_feat, pseudo_word_based_feat, self.temperature)
        ortho_loss = self.orthogonal_loss(img_salient_local_feats)
        reg_loss = self.semantic_regularization_loss(subject, attribute, pseudo_word_based_feat, img_subj_token,
                                                     img_attr_tokens, clip_model, local_attr_num)
        total_loss = img_text_loss + ortho_loss + self.hy_regLoss * reg_loss
        return total_loss
