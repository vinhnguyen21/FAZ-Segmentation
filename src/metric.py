from sklearn.metrics import f1_score
def multi_label_f1(y_gt, y_pred):
    """ Calculate F1 for each class

    Parameters
    ----------
    y_gt: torch.Tensor
        groundtruth
    y_pred: torch.Tensor
        prediction

    Returns
    -------
    list
        F1 of each class
    """
    f1_out = []
    gt_np = y_gt.to("cpu").numpy()
    pred_np = (y_pred.to("cpu").numpy() > 0.5) * 1.0
    assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"
    for i in range(gt_np.shape[1]):
        try:
            f1_out.append(f1_score(gt_np[:, i], pred_np[:, i]))
        except:
            f1_out.append(0)
    return f1_out
