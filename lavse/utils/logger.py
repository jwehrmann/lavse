from collections import OrderedDict
import logging


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return f'{self.val:6g}'
        # for stats
        return f'{self.val:.3f} ({self.avg:.3f})'


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """
            Concatenate the meters in one log line
        """
        s = ''
        for k, v in self.meters.items():
            s += f'{k.title()} {v}\t'
        return s.rstrip()

    def tb_log(self, tb_logger, prefix='data/', step=None):
        """
            Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.add_scalar(prefix + k, v.val, step)
    
    def update_dict(
        self, val_metrics, epoch, count, path
    ):
        
        for metric_name, metric_val in val_metrics.items():
            try:
                v = metric_val.item()
            except AttributeError:
                v = metric_val

            self.update(
                k=f'{metric_name}', v=v, n=0
            )
        self.update(k='valid/count', v=count, n=0)


def create_logger(level='info'):

    level = eval(f'logging.{level.upper()}')

    logging.basicConfig(
        format='%(asctime)s - [%(levelname)-8s] - %(message)s',
        level=level
    )

    logger = logging.getLogger(__name__)
    return logger


def get_logger():
    logger = logging.getLogger(__name__)
    return logger
    