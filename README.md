# FocalLoss


def soft_max_entropy_with(labels,logits):
    prob =  np.exp(logits) / np.sum(np.exp(logits))
    soft_entropy = -np.sum(labels * np.log(prob))
    return soft_entropy
