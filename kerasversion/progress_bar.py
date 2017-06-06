import sys

class ProgressBar(object):
    def __init__(self, total=100, stream=sys.stderr):
        self.total = total
        self.stream = stream
        self.last_len = 0
        self.curr = 0

    def count(self):
        self.curr += 1
        self.print_progress(self.curr)

    def print_progress(self, value):
        self.stream.write('\b' * self.last_len)
        pct = 100 * self.curr / self.total
        out = '{:.2f}% [{}/{}]'.format(pct, self.curr, self.total)
        self.last_len = len(out)
        self.stream.write(out)
        self.stream.flush()