import torch


class Statistics(object):
    """Accumulator for loss statistics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear statistics"""
        self.n_words = 0
        self.loss = 0
        self.accuracy = 0

    def update(self, loss, accuracy, n_words):
        total_words = self.n_words + n_words
        self.loss = self.loss * (self.n_words / total_words) + loss * (
            n_words / total_words)
        self.accuracy = self.accuracy * (
            self.n_words / total_words) + accuracy * (n_words / total_words)
        self.n_words = total_words


def accuracy(predict, target, ignore_index=None):
    """Accuracy of prediction."""
    if ignore_index is None:
        correct = torch.sum(predict == target).float().item()
        total = target.numel()
    else:
        correct = torch.sum((predict == target) &
                            (target != ignore_index)).float().item()
        total = torch.sum(target != ignore_index).item()
    accu = correct / total
    return accu, total
