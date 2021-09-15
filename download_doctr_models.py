from doctr.models import ocr_predictor

if __name__ == "__main__":
    doctr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)