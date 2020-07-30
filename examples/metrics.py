import torch


RangeTensor = Tuple[torch.Tensor, torch.Tensor]


def f1(predicted_range: RangeTensor, target_range: RangeTensor) -> torch.Tensor:
    predicted_start, predicted_end = predicted_range
    target_start,    target_end    = target_range
    
    overlap              = predicted_end - target_start
    overlap[overlap < 0] = 0.0

    precision = overlap / (predicted_end - predicted_start)
    recall    = overlap / (target_end    - target_start)
    f1        = 2 * precision * recall / (precision + recall)
    return f1