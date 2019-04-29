class Statistics(object):
    """Accumulator for loss statistics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear statistics"""
        self.n_words = 0
        self.loss = 0

    def update(self, loss, n_words):
        total_words = self.n_words + n_words
        self.loss = self.loss * (self.n_words / total_words) + loss * (
            n_words / total_words)
        self.n_words = total_words
