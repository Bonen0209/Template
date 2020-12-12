import torch
import torchvision
from base import BaseTester
from utils import MetricTracker
from tqdm import tqdm


class Tester(BaseTester):
    """
    Tester class
    """
    def __init__(self, model, criterion, metric_ftns, plot_ftns, config, data_loader):
        super().__init__(model, criterion, metric_ftns, plot_ftns, config)
        self.config = config
        self.data_loader = data_loader

        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

    def _test(self):
        """
        Test logic

        :return: A log that contains information about testing
        """
        self.model.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            outputs = []
            targets = []
            for batch_idx, (data, target) in enumerate(tqdm(self.data_loader)):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                outputs.append(output)
                targets.append(target)

                self.test_metrics.update('loss', loss.item())

            outputs = torch.cat(outputs)
            targets = torch.cat(targets)
            for met in self.metric_ftns:
                self.test_metrics.update(met.__name__, met(outputs, targets))
            for plt in self.plot_ftns:
                image_path = self.config.log_dir / (plt.__name__ + '.png')
                torchvision.utils.save_image(plt(outputs, targets).float(), image_path, normalize=True)

        return self.test_metrics.result()
