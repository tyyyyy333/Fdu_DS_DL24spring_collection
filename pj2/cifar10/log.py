from torch.utils.tensorboard import SummaryWriter
import os

class Logger:
    def __init__(self, log_dir='runs/canet_experiment'):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalars(self, phase, loss, acc, step):
        self.writer.add_scalar(f'Loss/{phase}', loss, step)
        self.writer.add_scalar(f'Accuracy/{phase}', acc, step)

    def log_lr(self, lr, step):
        self.writer.add_scalar('LearningRate', lr, step)

    def log_model_graph(self, model, input_tensor):
        self.writer.add_graph(model, input_tensor)

    def log_text(self, tag, text, step):
        self.writer.add_text(tag, text, step)
        
    def log_histograms(self, model, step):
        for name, param in model.named_parameters():
            self.writer.add_histogram(name, param, step)

    def log_attention_map(self, tag, attention_tensor, step):
        """
        attention_tensor: shape (B, C, H, W) or (1, 1, H, W)
        """
       # if attention_tensor.dim() == 4 and attention_tensor.size(1) == 1:
       #     img_grid = make_grid(attention_tensor, normalize=True, scale_each=True)
       #     self.writer.add_image(tag, img_grid, step)
        pass

    def close(self):
        self.writer.close()
        
if __name__ == '__main__':
    log = Logger()