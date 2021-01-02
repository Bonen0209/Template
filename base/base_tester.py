import shutil
import torch
from abc import abstractmethod


class BaseTester:
    """
    Base class for all testers
    """
    def __init__(self, model, criterion, metric_ftns, plot_ftns, config):
        self.config = config
        self.logger = config.get_logger('tester', config['tester']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.non_blocking = config['data_loader']['args']['pin_memory']
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.plot_ftns = plot_ftns

        # delete the save and log directory created for training
        shutil.rmtree(config.save_dir)

        self._resume_checkpoint(config.resume)

    @abstractmethod
    def _test(self):
        """
        Testing logic
        """
        raise NotImplementedError

    def test(self):
        """
        Full testing logic
        """
        result = self._test()

        # save logged informations into log dict
        log = result

        # print logged informations to the screen
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        self.logger.info("Checkpoint loaded. Testing")
