from src.document.CNI import CNI

filename = 'data/CNI_caro3.jpg'
cni = CNI(recto_path=filename)
cni.align_images()
cni.clean_images()
cni.extract_ocr()
results = cni.export_ocr()
print(results)