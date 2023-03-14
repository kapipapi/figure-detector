import torch


class Detector:
    def __init__(self, yolo_root_path, yolo_weights_path):
        self.model = torch.hub.load(yolo_root_path, 'custom', path=yolo_weights_path, source='local')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

    def detect_figures(self, img):
        results = self.model(img)
        return results.crop(save=False)

    def detect_with_letters(self, img, letter_model):
        results = self.model(img)
        results = results.crop(save=False)

        for result in results:
            cropped = result["im"]
            result["letter"] = letter_model.classify_letter(cropped)

        return results
