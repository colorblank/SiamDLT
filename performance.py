import time

class Logger:

    def __init__(self):
        self.time_table = {}
        self.percent = {}
        self.st = {}
        self.total_time = 0

    def start(self, name):
        self.st[name] = time.time()

    def end(self, name):
        spend = time.time() - self.st[name]
        try:
            self.time_table[name]
        except:
            self.time_table[name] = 0.
        self.total_time += spend
        self.time_table[name] += spend

    def log(self):
        print('performance log:')
        for k in self.time_table:
            print('key: {}, time: {:.3f}, percent: {:.2f}%'.format(k, self.time_table[k], 100. * self.time_table[k] / self.total_time))
        print('performance log finished')

