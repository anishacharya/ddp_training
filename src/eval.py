import torch
from typing import Dict
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def inference(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              mode: str):
    """
    :param model:
    :param data_loader:
    :param mode:
    :return:
    """
    true_labels, pred_labels, pred_scores = [], [], []

    # Inference
    model.eval()

    with torch.no_grad():
        if torch.cuda.is_available():
            model.cuda()

        for (feature, labels) in tqdm(data_loader, desc="Evaluating"):
            if torch.cuda.is_available():
                feature = feature.cuda()
                labels = labels.cuda()

            logits = model(feature=feature, mode=mode)
            _, predicted_labels = torch.max(logits.data, dim=1)
            prob_pred = torch.nn.functional.softmax(logits, dim=1)

            # collect predictions
            true_labels.append(labels)
            pred_labels.append(predicted_labels)
            pred_scores.append(prob_pred[:, 1])

    model.train()

    # Flatten
    true_labels = torch.cat(true_labels, dim=0).contiguous().cpu().numpy()
    pred_labels = torch.cat(pred_labels, dim=0).contiguous().cpu().numpy()
    pred_scores = torch.cat(pred_scores, dim=0).contiguous().cpu().numpy()

    return true_labels, pred_labels, pred_scores


def basic_eval(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, mode: str) -> Dict:
    """
    Standard evaluation pipeline
    """
    true_labels, pred_labels, pred_scores = inference(model=model, data_loader=data_loader, mode=mode)
    acc = accuracy_score(y_true=true_labels, y_pred=pred_labels)

    return acc