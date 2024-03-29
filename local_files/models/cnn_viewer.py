"""
Class CNN_Viewer will inherit from class CNN and complete it for result analysis and testing our models.
"""

from cnn import CNN


class CnnViewer(CNN):
    def __init__(self, model_path: str, model_name: str = 'UNet', device: str = 'cpu'):
        super().__init__(model_name, device)

        self.restore_model(model_path)
