import sys

import config

class Logger():
    def __init__(self, ofp=sys.stderr):
        self.ofp = ofp
    
    def __call__(self, msg, log_time=False, level=None):
        assert level in ('DEBUG', 'VERBOSE', None)
        if level == 'DEBUG' and not config.DEBUG: return
        if level == 'VERBOSE' and not config.VERBOSE: return

        if config.LOG_TIME:
            time = datetime.fromtimestamp(time.time()).strftime(
                '%Y-%m-%d %H:%M:%S: ')
            msg = time + msg
        self.ofp.write(msg.strip() + "\n")

def log_progress(tree, i):
    msg = "\n".join([
        'imbalanced train error: {0}'.format(tree.imbal_train_err[i]),
        'imbalanced test error: {0}'.format(tree.imbal_test_err[i]),
        'x1 split feat {0}'.format(tree.split_x1[i]),
        'x2 split feat {0}'.format(tree.split_x2[i]),
        'rule score {0}'.format(tree.scores[i])])
    log(msg, log_time=False, level='VERBOSE')

log = Logger()
