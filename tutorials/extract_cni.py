from src.document.CNI import CNI

filename = 'tutorials/model_CNI.png'
cni = CNI(recto_path=filename)
cni.align_images()
cni.clean_images()
cni.extract_ocr()
results = cni.export_ocr()
print(results)