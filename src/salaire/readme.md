# How to create preannotated file in the Label Studio Format with Doctr bounding boxes




`list_img_path = [Path(x) for x in os.listdir(img_folder_path)]
doctr_transformer = DoctrTransformer()
list_doctr_docs = doctr_transformer._get_doctr_docs(list_img_path)
annotations = AnnotationJsonCreator(list_img_path, output_path).transform(list_doctr_docs)`

where : 
img_folder_path : forlder where your images are in jpg, png, jpeg format
output_path: path/name_preannotation.json
