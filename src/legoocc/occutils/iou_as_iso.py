import numpy as np
import torch
import torch.distributed as dist


def _ensure_batch_dim(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.ndim == 3 else x  # [D,H,W] -> [1,D,H,W]; 其余保持不变

def _safe_div(num, den, eps=1e-8):
    return num / (den + eps)

class SSCMetricsTorch:
    """
    PyTorch 实现（无 NumPy 依赖）
    - 忽略标签 255
    - completion（二分类占据）: precision/recall/IoU
    - semantic + completion: per-class IoU（tp/fp/fn）
    - 可选分布式统计 all_reduce（自动检测 dist 是否初始化）
    """

    def __init__(self, n_classes: int, device = None, dtype=torch.float64):
        self.n_classes = int(n_classes)
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.reset()

    # ------ 与你原接口一致的两个函数（但用 torch 实现） ------
    def hist_info(self, n_cl: int, pred: torch.Tensor, gt: torch.Tensor):
        """
        返回:
            hist: [n_cl, n_cl] 混淆矩阵
            correct: 标注内像素中预测正确的数量
            labeled: 标注内像素总数
        """
        assert pred.shape == gt.shape
        pred = pred.to(torch.int64)
        gt   = gt.to(torch.int64)

        valid = (gt >= 0) & (gt < n_cl)  # 排除 255 / 其它无效
        labeled = int(valid.sum().item())
        correct = int((pred[valid] == gt[valid]).sum().item())

        idx = (gt[valid] * n_cl + pred[valid]).to(torch.int64)
        hist = torch.bincount(idx, minlength=n_cl * n_cl).reshape(n_cl, n_cl).to(self.device)
        return hist, correct, labeled

    @staticmethod
    def compute_score(hist: torch.Tensor, correct: int, labeled: int):
        """
        计算 IoU、mIoU、mIoU(no background)、freq-IoU、pixel acc
        返回:
            iu: [C] per-class IoU (torch.Tensor)
            mean_IU: float
            mean_IU_no_back: float
            mean_pixel_acc: float
        """
        hist = hist.to(torch.float64)
        diag = torch.diag(hist)
        union = hist.sum(dim=1) + hist.sum(dim=0) - diag
        iu = _safe_div(diag, union)
        mean_IU = torch.nanmean(iu).item()
        mean_IU_no_back = torch.nanmean(iu[1:]).item() if iu.numel() > 1 else float("nan")
        freq = _safe_div(hist.sum(dim=1), hist.sum())
        freq_IU = (iu[freq > 0] * freq[freq > 0]).sum().item()
        mean_pixel_acc = (correct / labeled) if labeled != 0 else 0.0
        return iu, mean_IU, mean_IU_no_back, mean_pixel_acc

    # ------ 主流程 ------
    def add_batch(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                  nonempty = None,
                  nonsurface = None):

        # 统一到 [B, ...]
        if y_pred.ndim == y_true.ndim:
            y_pred = _ensure_batch_dim(y_pred)
            y_true = _ensure_batch_dim(y_true)

        y_pred = y_pred.to(self.device)
        y_true = y_true.to(self.device)

        self.count += 1.0

        # ----- Completion（二分类：空/占据） -----
        mask = (y_true != 255)
        if nonempty is not None:
            mask = mask & nonempty.to(self.device).bool()
        if nonsurface is not None:
            mask = mask & nonsurface.to(self.device).bool()

        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)
        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn

        # ----- Semantic + Completion（逐类） -----
        mask_sem = (y_true != 255)
        if nonempty is not None:
            mask_sem = mask_sem & nonempty.to(self.device).bool()

        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(y_pred, y_true, mask_sem)
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum

    def get_stats(self, distributed: bool = True):
        """
        返回:
            {
              "precision": float,
              "recall": float,
              "iou": float,            # completion IoU
              "iou_ssc": Tensor[C],    # per-class IoU
              "iou_ssc_mean": float    # mean over classes[1:]
            }
        """
        completion_tp = self.completion_tp.clone()
        completion_fp = self.completion_fp.clone()
        completion_fn = self.completion_fn.clone()
        tps = self.tps.clone()
        fps = self.fps.clone()
        fns = self.fns.clone()

        if distributed and dist.is_available() and dist.is_initialized():
            for t in [completion_tp, completion_fp, completion_fn, tps, fps, fns]:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        # completion metrics
        precision = _safe_div(completion_tp, completion_tp + completion_fp).item() if completion_tp > 0 else 0.0
        recall    = _safe_div(completion_tp, completion_tp + completion_fn).item() if completion_tp > 0 else 0.0
        iou_c     = _safe_div(completion_tp, completion_tp + completion_fp + completion_fn).item() if completion_tp > 0 else 0.0

        # semantic+completion IoU per class
        iou_ssc = _safe_div(tps, tps + fps + fns + 1e-5)  # [C]
        iou_ssc_mean = torch.mean(iou_ssc[1:]).item() if iou_ssc.numel() > 1 else float("nan")

        return {
            "precision": precision,
            "recall": recall,
            "iou": iou_c,
            "iou_ssc": iou_ssc,             # torch.Tensor[C], 保留 tensor 便于后续处理/打印
            "iou_ssc_mean": iou_ssc_mean,
        }

    def reset(self):
        self.completion_tp = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.completion_fp = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.completion_fn = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        self.tps = torch.zeros(self.n_classes, device=self.device, dtype=self.dtype)
        self.fps = torch.zeros(self.n_classes, device=self.device, dtype=self.dtype)
        self.fns = torch.zeros(self.n_classes, device=self.device, dtype=self.dtype)

        self.hist_ssc = torch.zeros((self.n_classes, self.n_classes), device=self.device, dtype=self.dtype)
        self.labeled_ssc = 0
        self.correct_ssc = 0

        self.precision = 0.0
        self.recall = 0.0
        self.iou = 0.0
        self.count = torch.tensor(1e-8, device=self.device, dtype=self.dtype)

        self.iou_ssc = torch.zeros(self.n_classes, device=self.device, dtype=torch.float32)
        self.cnt_class = torch.zeros(self.n_classes, device=self.device, dtype=torch.float32)

    # ------ 细节计算 ------
    def get_score_completion(self, predict: torch.Tensor, target: torch.Tensor, valid_mask = None):
        """
        completion: 空/占据二分类（>0 为占据）
        返回: tp_sum, fp_sum, fn_sum （float64 tensors 标量）
        """
        # 忽略 255
        predict = predict.clone()
        target  = target.clone()
        predict[target == 255] = 0
        target[target == 255]  = 0

        B = predict.shape[0]
        y_pred = (predict.reshape(B, -1) > 0)
        y_true = (target.reshape(B, -1) > 0)

        if valid_mask is not None:
            m = valid_mask.reshape(B, -1).bool()
            y_pred = torch.where(m, y_pred, torch.zeros_like(y_pred))
            y_true = torch.where(m, y_true, torch.zeros_like(y_true))

        tp = (y_true & y_pred).sum(dtype=self.dtype)
        fp = (~y_true & y_pred).sum(dtype=self.dtype)
        fn = (y_true & ~y_pred).sum(dtype=self.dtype)
        return tp, fp, fn

    def get_score_semantic_and_completion(self, predict: torch.Tensor, target: torch.Tensor, valid_mask = None):
        """
        逐类统计 tp/fp/fn
        返回: tp_sum[C], fp_sum[C], fn_sum[C]
        """
        predict = predict.clone()
        target  = target.clone()
        predict[target == 255] = 0
        target[target == 255]  = 0

        B = predict.shape[0]
        y_pred = predict.reshape(B, -1).to(torch.int64)
        y_true = target.reshape(B, -1).to(torch.int64)

        if valid_mask is not None:
            m = valid_mask.reshape(B, -1).bool()
            # 将无效位置的标签置为一个不会等于任何类的值（这里置为 -1）
            y_true = torch.where(m, y_true, torch.full_like(y_true, -1))
            y_pred = torch.where(m, y_pred, torch.full_like(y_pred, -2))  # 与 y_true 不同，避免误计为 tp

        tp_sum = torch.zeros(self.n_classes, device=self.device, dtype=self.dtype)
        fp_sum = torch.zeros(self.n_classes, device=self.device, dtype=self.dtype)
        fn_sum = torch.zeros(self.n_classes, device=self.device, dtype=self.dtype)

        # 逐类统计（简单直接，足够快；如需更快可用 one-hot / scatter 优化）
        for j in range(self.n_classes):
            pred_is_j = (y_pred == j)
            true_is_j = (y_true == j)
            tp = (pred_is_j & true_is_j).sum(dtype=self.dtype)
            fp = (pred_is_j & ~true_is_j).sum(dtype=self.dtype)
            fn = (~pred_is_j & true_is_j).sum(dtype=self.dtype)

            tp_sum[j] += tp
            fp_sum[j] += fp
            fn_sum[j] += fn

        return tp_sum, fp_sum, fn_sum




class SSCMetrics:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def hist_info(self, n_cl, pred, gt):
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    @staticmethod
    def compute_score(hist, correct, labeled):
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mean_IU = np.nanmean(iu)
        mean_IU_no_back = np.nanmean(iu[1:])
        freq = hist.sum(1) / hist.sum()
        freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
        mean_pixel_acc = correct / labeled if labeled != 0 else 0

        return iu, mean_IU, mean_IU_no_back, mean_pixel_acc

    def add_batch(self, y_pred, y_true, nonempty=None, nonsurface=None):

        if y_pred.shape[0] != 1:
            y_pred = y_pred[None]
            y_true = y_true[None]

        self.count += 1
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)

        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn

        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(
            y_pred, y_true, mask
        )
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum

    # def get_stats(self):
    #     if self.completion_tp != 0:
    #         precision = self.completion_tp / (self.completion_tp + self.completion_fp)
    #         recall = self.completion_tp / (self.completion_tp + self.completion_fn)
    #         iou = self.completion_tp / (
    #             self.completion_tp + self.completion_fp + self.completion_fn
    #         )
    #     else:
    #         precision, recall, iou = 0, 0, 0
    #     iou_ssc = self.tps / (self.tps + self.fps + self.fns + 1e-5)
    #     return {
    #         "precision": precision,
    #         "recall": recall,
    #         "iou": iou,
    #         "iou_ssc": iou_ssc,
    #         "iou_ssc_mean": np.mean(iou_ssc[1:]),
    #     }
    def get_stats(self, distributed=True):
        completion_tp = self.completion_tp
        completion_fp = self.completion_fp
        completion_fn = self.completion_fn
        tps = self.tps.copy()
        fps = self.fps.copy()
        fns = self.fns.copy()

        if distributed:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            completion_tp = torch.tensor(completion_tp, dtype=torch.float64, device=device)
            completion_fp = torch.tensor(completion_fp, dtype=torch.float64, device=device)
            completion_fn = torch.tensor(completion_fn, dtype=torch.float64, device=device)
            tps = torch.tensor(tps, dtype=torch.float64, device=device)
            fps = torch.tensor(fps, dtype=torch.float64, device=device)
            fns = torch.tensor(fns, dtype=torch.float64, device=device)

            import torch.distributed as dist
            dist.all_reduce(completion_tp, op=dist.ReduceOp.SUM)
            dist.all_reduce(completion_fp, op=dist.ReduceOp.SUM)
            dist.all_reduce(completion_fn, op=dist.ReduceOp.SUM)
            dist.all_reduce(tps, op=dist.ReduceOp.SUM)
            dist.all_reduce(fps, op=dist.ReduceOp.SUM)
            dist.all_reduce(fns, op=dist.ReduceOp.SUM)

            completion_tp = completion_tp.item()
            completion_fp = completion_fp.item()
            completion_fn = completion_fn.item()
            tps = tps.cpu().numpy()
            fps = fps.cpu().numpy()
            fns = fns.cpu().numpy()

        if completion_tp != 0:
            precision = completion_tp / (completion_tp + completion_fp)
            recall = completion_tp / (completion_tp + completion_fn)
            iou = completion_tp / (completion_tp + completion_fp + completion_fn)
        else:
            precision, recall, iou = 0, 0, 0
        iou_ssc = tps / (tps + fps + fns + 1e-5)
        return {
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "iou_ssc": iou_ssc,
            "iou_ssc_mean": np.mean(iou_ssc[1:]),
        }

    def reset(self):

        self.completion_tp = 0
        self.completion_fp = 0
        self.completion_fn = 0
        self.tps = np.zeros(self.n_classes)
        self.fps = np.zeros(self.n_classes)
        self.fns = np.zeros(self.n_classes)

        self.hist_ssc = np.zeros((self.n_classes, self.n_classes))
        self.labeled_ssc = 0
        self.correct_ssc = 0

        self.precision = 0
        self.recall = 0
        self.iou = 0
        self.count = 1e-8
        self.iou_ssc = np.zeros(self.n_classes, dtype=np.float32)
        self.cnt_class = np.zeros(self.n_classes, dtype=np.float32)

    def get_score_completion(self, predict, target, nonempty=None):
        predict = np.copy(predict)
        target = np.copy(target)

        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]  # batch size
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, _C, 129600), 60*36*60=129600
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = np.zeros(predict.shape)
        b_true = np.zeros(target.shape)
        b_pred[predict > 0] = 1
        b_true[target > 0] = 1
        p, r, iou = 0.0, 0.0, 0.0
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :]  # GT
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]

            tp = np.array(np.where(np.logical_and(y_true == 1, y_pred == 1))).size
            fp = np.array(np.where(np.logical_and(y_true != 1, y_pred == 1))).size
            fn = np.array(np.where(np.logical_and(y_true == 1, y_pred != 1))).size
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        return tp_sum, fp_sum, fn_sum

    def get_score_semantic_and_completion(self, predict, target, nonempty=None):
        target = np.copy(target)
        predict = np.copy(predict)
        _bs = predict.shape[0]  # batch size
        _C = self.n_classes  # _C = 12
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, 129600), 60*36*60=129600

        cnt_class = np.zeros(_C, dtype=np.int32)  # count for each class
        iou_sum = np.zeros(_C, dtype=np.float32)  # sum of iou for each class
        tp_sum = np.zeros(_C, dtype=np.int32)  # tp
        fp_sum = np.zeros(_C, dtype=np.int32)  # fp
        fn_sum = np.zeros(_C, dtype=np.int32)  # fn

        for idx in range(_bs):
            y_true = target[idx, :]  # GT
            y_pred = predict[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_pred = y_pred[
                    np.where(np.logical_and(nonempty_idx == 1, y_true != 255))
                ]
                y_true = y_true[
                    np.where(np.logical_and(nonempty_idx == 1, y_true != 255))
                ]
            for j in range(_C):  # for each class
                tp = np.array(np.where(np.logical_and(y_true == j, y_pred == j))).size
                fp = np.array(np.where(np.logical_and(y_true != j, y_pred == j))).size
                fn = np.array(np.where(np.logical_and(y_true == j, y_pred != j))).size

                tp_sum[j] += tp
                fp_sum[j] += fp
                fn_sum[j] += fn

        return tp_sum, fp_sum, fn_sum