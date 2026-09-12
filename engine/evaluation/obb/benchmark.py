"""Explicit Ultralytics-style OBB AP protocols, independent of model/dataset.

Numerically checked against the user-supplied FressDet reference functions.
The protocol name describes executed code, not a claim of paper reproduction.
"""
import numpy as np
import torch

from ...core import register
from ...rtv4.rotated_box_ops import rotated_iou
from .dota import DotaOBBEvaluator


def pairwise_probiou(first, second, eps=1e-7):
    """Gaussian/Hellinger OBB similarity; pixel xywh and radians, [N,M]."""
    first, second = first.float(), second.float()
    def covariance(box):
        variance = box[:,2:4].square()/12
        cosine, sine = box[:,4].cos(), box[:,4].sin()
        return (variance[:,0]*cosine.square()+variance[:,1]*sine.square(),
                variance[:,0]*sine.square()+variance[:,1]*cosine.square(),
                (variance[:,0]-variance[:,1])*cosine*sine)
    a,b,c = (v[:,None] for v in covariance(first))
    d,e,f = (v[None,:] for v in covariance(second))
    dx = first[:,0,None]-second[None,:,0]
    dy = first[:,1,None]-second[None,:,1]
    determinant = (a+d)*(b+e)-(c+f).square()
    location = ((a+d)*dy.square()+(b+e)*dx.square())/(determinant+eps)*.25
    cross = (c+f)*(-dx)*dy/(determinant+eps)*.5
    shape = (determinant/(4*((a*b-c.square()).clamp_min(0)*(d*e-f.square()).clamp_min(0)).sqrt()+eps)+eps).log()*.5
    distance = (location+cross+shape).clamp(eps,100)
    return 1-(1-(-distance).exp()+eps).sqrt()


def benchmark_matches(similarity, gt_labels, pred_labels, thresholds):
    """Released greedy IoU matching, including its two np.unique ordering steps."""
    overlaps = (similarity * (gt_labels[:,None] == pred_labels[None,:])).cpu().numpy()
    correct = np.zeros((len(pred_labels),len(thresholds)),dtype=bool)
    for column, threshold in enumerate(thresholds):
        pairs = np.argwhere(overlaps >= threshold)
        if len(pairs) > 1:
            pairs = pairs[np.argsort(overlaps[pairs[:,0],pairs[:,1]])[::-1]]
            pairs = pairs[np.unique(pairs[:,1],return_index=True)[1]]
            pairs = pairs[np.unique(pairs[:,0],return_index=True)[1]]
        if len(pairs):
            correct[pairs[:,1],column] = True
    return correct


def benchmark_ap(recall, precision):
    """101 abscissae + interpolated precision envelope + trapezoidal integral."""
    recall = np.r_[0.,recall,1.]
    envelope = np.maximum.accumulate(np.r_[1.,precision,0.][::-1])[::-1]
    abscissae = np.linspace(0,1,101)
    return float(np.trapz(np.interp(abscissae,recall,envelope),abscissae))


def probiou_fast_nms(boxes, scores, labels, threshold):
    """Class-aware fast NMS; a higher-score box may suppress even if removed."""
    status = labels.new_ones(len(labels))
    parent = labels.new_full((len(labels),),-1)
    overlap = scores.new_zeros(len(labels))
    for label in labels.unique():
        indices = torch.where(labels == label)[0]
        indices = indices[scores[indices].argsort(descending=True)]
        matrix = pairwise_probiou(boxes[indices],boxes[indices]).triu(diagonal=1)
        maximum, owner = matrix.max(dim=0)
        retained = maximum < threshold
        status[indices[retained]] = 0
        parent[indices[~retained]] = indices[owner[~retained]]
        overlap[indices[~retained]] = maximum[~retained]
    keep = torch.where(status == 0)[0]
    keep = keep[scores[keep].argsort(descending=True)]
    return keep, status, parent, overlap


@register()
class BenchmarkOBBEvaluator(DotaOBBEvaluator):
    """ProbIoU AP plus geometric-IoU AP on exactly the same final predictions.

Both use the released matching/interpolation algorithm. Geometric AP here is
therefore NOT the old DOTA-07 metric. Unsupported ignore semantics fail loudly.
"""
    STAT_NAMES = ("mAP50_95", "AP50", "AP75", "riou_mAP50_95", "riou_AP50", "riou_AP75")

    def __init__(self, dataset, selection_metric="AP50", require_complete=True):
        self.require_complete = bool(require_complete)
        self.protocol = "ultralytics_obb_probiou_interp101_trapz_v1"
        # Match the reference's float32 torch.linspace thresholds exactly.
        super().__init__(dataset,iou_thresholds=torch.linspace(.5,.95,10).tolist(),
                         use_07_metric=False,selection_metric=selection_metric)

    def accumulate(self, verbose=True):
        if self.require_complete and set(self.predictions) != set(range(len(self.dataset))):
            raise ValueError("Benchmark OBB evaluation requires a prediction entry for every image")
        truths, labels, scores = [],[],[]
        correctness = {"probiou":[],"riou":[]}
        for image_id in range(len(self.dataset)):
            gt = self.dataset.get_ground_truth(image_id)
            if gt["difficulty"].bool().any() or len(gt.get("ignore_boxes",[])):
                raise ValueError("Benchmark OBB protocol has no ignore-region semantics")
            prediction = self.predictions.get(image_id,dict(boxes=torch.empty(0,5),scores=torch.empty(0),labels=torch.empty(0,dtype=torch.long)))
            truths.append(gt["labels"].cpu().numpy())
            labels.append(prediction["labels"].numpy())
            scores.append(prediction["scores"].numpy())
            for name, overlap in (("probiou",pairwise_probiou),
                                  ("riou",lambda a,b: rotated_iou(a,b,model_space=False))):
                similarity = overlap(gt["boxes"].float().cpu(),prediction["boxes"])
                correctness[name].append(benchmark_matches(similarity,gt["labels"].cpu(),prediction["labels"],self.iou_thresholds))
        truth = np.concatenate(truths) if truths else np.empty(0,dtype=int)
        prediction_labels = np.concatenate(labels) if labels else np.empty(0,dtype=int)
        confidence = np.concatenate(scores) if scores else np.empty(0)
        order = np.argsort(-confidence)
        class_ids, class_counts = np.unique(truth,return_counts=True)
        results = {}
        for name in correctness:
            tp = np.concatenate(correctness[name],axis=0)[order] if correctness[name] else np.empty((0,10))
            ap = np.zeros((len(class_ids),10))
            for index, (class_id,count) in enumerate(zip(class_ids,class_counts)):
                positives = tp[prediction_labels[order] == class_id]
                if not len(positives):
                    continue
                cum_tp, cum_fp = positives.cumsum(0),(1-positives).cumsum(0)
                recall = cum_tp/(count+1e-16)
                precision = cum_tp/(cum_tp+cum_fp)
                ap[index] = [benchmark_ap(recall[:,j],precision[:,j]) for j in range(10)]
            results[name] = ap
        primary, geometric = results["probiou"],results["riou"]
        summary = lambda ap: [float(ap.mean()),float(ap[:,0].mean()),float(ap[:,5].mean())] if len(ap) else [0.,0.,0.]
        self.stats = np.asarray(summary(primary)+summary(geometric))
        self.metrics = dict(zip(self.STAT_NAMES,map(float,self.stats)))
        self.per_class = dict.fromkeys(self.dataset.classes)
        self.per_class_metrics = {}
        for row,class_id in enumerate(class_ids):
            name = self.dataset.classes[int(class_id)]
            self.per_class[name] = float(primary[row,0])
            self.per_class_metrics[name] = dict(zip(self.STAT_NAMES,summary(primary[row:row+1])+summary(geometric[row:row+1])))
        if verbose:
            print(f"Benchmark OBB: {len(self.dataset)} images; {self.protocol}")

    def summarize(self):
        print(f"{self.protocol}; checkpoint selection={self.selection_metric}")
        for name,value in self.metrics.items():
            print(f"  {name}: {value:.6f}")
