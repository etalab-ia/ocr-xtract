def convert_from_ls(result):
    if 'original_width' not in result or 'original_height' not in result:
        return None

    value = result['value']
    w, h = result['original_width'], result['original_height']

    if all([key in value for key in ['x', 'y', 'width', 'height']]):
        return w * value['x'] / 100.0, \
               h * value['y'] / 100.0, \
               w * value['width'] / 100.0, \
               h * value['height'] / 100.0


if __name__ == "__main__":
    """
    This script is for extracting the position of the OCR zones in an image with a fixed frame. To use it prepare a 
    LabelStudio instance with your image and ask the annotations of the labels you need. 
    Extract the result in json and run the script
    """
    import json
    from pathlib import Path

    with open(Path('data/CNI_robin_annotated.json'), 'r') as f:
        data = json.load(f)[0]
    res = []
    for label in data['annotations'][0]['result']:
        try:
            x, y, w, h = convert_from_ls(label)
            annotation = {'value': label['value']['rectanglelabels'][0],
                          'title': (int(x), int(y), int(x + w), int(y + h))}
            res.append(annotation)
        except:
            continue
    print(res)
