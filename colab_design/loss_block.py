import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import sigmoid_focal_loss
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from transformers import CLIPTokenizer, CLIPModel

# ADE20k Class Names (150 classes)
ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass", "cabinet",
    "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair", "car",
    "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat",
    "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column",
    "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator",
    "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway",
    "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench",
    "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine",
    "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth",
    "television receiver", "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator",
    "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy",
    "washer", "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike",
    "cradle", "oven", "ball", "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle",
    "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray",
    "ashcan", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator",
    "glass", "clock", "flag"
]

class BoundaryAwareLoss(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)

    def forward(self, pred_masks, target_masks):
        if pred_masks.numel() == 0:
            return pred_masks.sum() * 0
        pred_input = pred_masks.unsqueeze(1) 
        target_input = target_masks.unsqueeze(1) 
        kernel = self.kernel.to(pred_masks.device)
        pred_edges = F.conv2d(pred_input, kernel, padding=1)
        target_edges = F.conv2d(target_input, kernel, padding=1)
        return F.l1_loss(pred_edges, target_edges)

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_mask=1, cost_dice=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  
        out_mask = outputs["pred_masks"].flatten(0, 1).flatten(1)  
        tgt_ids = torch.cat([v["class_labels"] for v in targets])
        tgt_mask = torch.cat([v["masks"] for v in targets])
        H_p, W_p = outputs["pred_masks"].shape[-2:]
        tgt_mask = F.interpolate(tgt_mask.unsqueeze(1), size=(H_p, W_p), mode='nearest').squeeze(1)
        tgt_mask = tgt_mask.flatten(1) 
        cost_class = -out_prob[:, tgt_ids]
        cost_mask = torch.cdist(out_mask, tgt_mask, p=1)
        out_mask_sig = out_mask.sigmoid()
        numerator = 2 * torch.mm(out_mask_sig, tgt_mask.t())
        denominator = out_mask_sig.sum(-1).unsqueeze(1) + tgt_mask.sum(-1).unsqueeze(0)
        cost_dice = 1 - (numerator / (denominator + 1e-6))
        C = self.cost_class * cost_class + self.cost_mask * cost_mask + self.cost_dice * cost_dice
        C = C.view(bs, num_queries, -1).cpu()
        indices = []
        sizes = [len(v["class_labels"]) for v in targets]
        for i, c in enumerate(C.split(sizes, -1)):
            if c.shape[-1] == 0: 
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue
            row_ind, col_ind = linear_sum_assignment(c[0]) 
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64)))
        return indices

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, num_parents=30, label_smoothing=0.1, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.boundary_loss_func = BoundaryAwareLoss()
        
        # Hierarchical Config
        self.num_parents = num_parents
        self.label_smoothing = label_smoothing
        self.hierarchy = None
        self.device_name = device # Store device string
        
        # To be computed on first forward or init
        # We compute it lazily or here if device is ready
        self._hierarchy_computed = False

    def _compute_hierarchy(self, device):
        print("Computing Class Hierarchy from CLIP Embeddings...")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        
        class_names = ADE20K_CLASSES[:self.num_classes] 
        inputs = tokenizer(class_names, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            embeddings = model.get_text_features(**inputs)
            
        embeddings = embeddings.cpu().numpy()
        # Cosine distance
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norm + 1e-8)
        distances = 1 - (embeddings @ embeddings.T)
        distances = (distances + distances.T) / 2 # Enforce symmetry
        np.fill_diagonal(distances, 0) # Enforce 0 diagonal for scipy
        distances = distances.clip(min=0)
        
        # Clustering
        condensed = squareform(distances) # Clip negative precision errors
        linkage_matrix = linkage(condensed, method='ward')
        cluster_labels = fcluster(linkage_matrix, self.num_parents, criterion='maxclust')
        
        self.hierarchy = torch.tensor(cluster_labels - 1, dtype=torch.long, device=device)
        self._hierarchy_computed = True
        print("Hierarchy Computed.")

    def create_soft_target(self, target_classes_o, device):
        """
        Create soft target for matched queries (B_matches, NumClasses)
        """
        N = target_classes_o.shape[0]
        soft_target = torch.ones(N, self.num_classes, device=device) * (self.label_smoothing / (self.num_classes - 1))
        
        for i, class_id in enumerate(target_classes_o):
            parent_id = self.hierarchy[class_id]
            
            # True class
            soft_target[i, class_id] = 1.0 - self.label_smoothing
            
            # Siblings
            sibling_mask = (self.hierarchy == parent_id)
            sibling_ids = torch.where(sibling_mask)[0]
            
            if len(sibling_ids) > 1:
                remaining_mass = self.label_smoothing / len(sibling_ids)
                # Distribute to siblings (including self, but self already boosted, so effectively boost 'other' siblings)
                # Actually user logic: "Distribute remaining mass among siblings"
                # Let's simple add boosted probability to siblings
                soft_target[i, sibling_ids] += remaining_mass

        # Renormalize
        soft_target = soft_target / (soft_target.sum(dim=1, keepdim=True) + 1e-8)
        return soft_target

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits'] # [B, Q, K+1]
        src_logits = src_logits[..., :-1]   # [B, Q, K]
        
        idx = self._get_src_permutation_idx(indices) # (Batch_idx, Query_idx) for Matches
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # Initialize Hierarchy if needed
        if not self._hierarchy_computed:
            self._compute_hierarchy(src_logits.device)

        # --- for "No Object" queries (Background), we usually push probabilities down -> entropy Max
        # But Mask2Former typically matches specific queries.
        # Unmatched queries (Backgound) are standard Focal Loss to 0.
        
        # 1. Standard Focal Loss (Classification Hard Imbalance)
        target_classes_onehot = torch.zeros_like(src_logits)
        target_classes_onehot[idx[0], idx[1], target_classes_o] = 1.0
        loss_focal = sigmoid_focal_loss(src_logits, target_classes_onehot, alpha=0.25, gamma=2.0, reduction="sum")
        loss_focal = loss_focal / num_boxes

        # 2. Hierarchical Loss Components (Only on MATCHED queries)
        # We only apply semantic guidance to positive objects.
        matched_logits = src_logits[idx] # [N_matches, K]
        
        if len(matched_logits) > 0:
            # Soft Targets
            soft_targets = self.create_soft_target(target_classes_o, src_logits.device)
            
            # KL Loss (Fineness)
            # LogSoftmax on logits
            log_probs = F.log_softmax(matched_logits, dim=1)
            loss_kl = F.kl_div(log_probs, soft_targets, reduction='batchmean')
            
            # Parent Loss (Coarseness)
            # Sum logits for parents
            parent_logits = torch.zeros(len(matched_logits), self.num_parents, device=src_logits.device)
            # There is probably a scatter_add_ way to do this faster, but loop is safe for now
            for pid in range(self.num_parents):
                child_mask = (self.hierarchy == pid)
                if child_mask.any():
                    # logsumexp of children logits for this parent
                    parent_logits[:, pid] = torch.logsumexp(matched_logits[:, child_mask], dim=1)
            
            target_parents = self.hierarchy[target_classes_o]
            loss_parent = F.cross_entropy(parent_logits, target_parents)
            
        else:
            loss_kl = torch.tensor(0.0, device=src_logits.device)
            loss_parent = torch.tensor(0.0, device=src_logits.device)

        return {'loss_ce': loss_focal, 'loss_kl': loss_kl, 'loss_parent': loss_parent}

    def loss_masks(self, outputs, targets, indices, num_boxes):
        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs['pred_masks'][src_idx] 
        target_masks = torch.cat([t['masks'][J] for t, (_, J) in zip(targets, indices)])
        
        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False).squeeze(1)

        # Use BCEWithLogits for AMP safety
        loss_sigmoid = F.binary_cross_entropy_with_logits(src_masks, target_masks)
        
        src_masks_sigmoid = src_masks.sigmoid()
        
        src_masks_flat = src_masks_sigmoid.flatten(1)
        target_masks_flat = target_masks.flatten(1)
        numerator = 2 * (src_masks_flat * target_masks_flat).sum(1)
        denominator = src_masks_flat.sum(1) + target_masks_flat.sum(1)
        loss_dice = 1 - (numerator + 1) / (denominator + 1)
        loss_dice = loss_dice.mean()
        
        loss_boundary = self.boundary_loss_func(src_masks_sigmoid, target_masks)
        return {'loss_mask': loss_sigmoid, 'loss_dice': loss_dice, 'loss_boundary': loss_boundary}
    
    def loss_consistency(self, outputs, targets, indices, num_boxes):
        if "aux_outputs" not in outputs:
            return {'loss_consistency': torch.tensor(0.0).to(outputs['pred_logits'].device)}
        src_masks_high = outputs['pred_masks'] 
        loss = 0.0
        for i, aux in enumerate(outputs["aux_outputs"]):
            src_masks_low = aux["pred_masks"] 
            target_size = src_masks_low.shape[-2:]
            src_masks_high_down = F.interpolate(src_masks_high, size=target_size, mode='bilinear', align_corners=False)
            loss += F.l1_loss(src_masks_high_down.sigmoid(), src_masks_low.sigmoid())
        return {'loss_consistency': loss}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_masks(outputs, targets, indices, num_boxes))
        losses.update(self.loss_consistency(outputs, targets, indices, num_boxes))
        
        # User defined weights adapted
        w_focal = self.weight_dict.get('loss_ce', 0.25)
        w_parent = self.weight_dict.get('loss_parent', 0.20)
        w_kl = self.weight_dict.get('loss_kl', 0.40)
        
        w_mask = self.weight_dict.get('loss_mask', 5.0)
        w_dice = self.weight_dict.get('loss_dice', 5.0) 
        w_boundary = self.weight_dict.get('loss_boundary', 2.0)
        w_consistency = self.weight_dict.get('loss_consistency', 1.0)
        
        final_loss = (losses['loss_ce'] * w_focal + 
                     losses['loss_parent'] * w_parent + 
                     losses['loss_kl'] * w_kl +
                     losses['loss_mask'] * w_mask + 
                     losses['loss_dice'] * w_dice +
                     losses['loss_boundary'] * w_boundary +
                     losses['loss_consistency'] * w_consistency)
        
        return final_loss, losses
