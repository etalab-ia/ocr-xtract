import requests
from pathlib import Path
import os

url = "http://127.0.0.1:5000/"

def get_list_of_files():
    response = requests.get(f'{url}/files')
    print(response.json())

def post_file(file):
    with open(file, 'rb') as fp:
        files = {'file': fp}
        response = requests.post(
            f'{url}', files=files
        )
    print(response.status_code)
    return response.json()


if __name__ == '__main__':
    file = Path("./data/CNI_caro2.jpg")
    #get_list_of_files()
    f = post_file(file)
    print(f['result'])
    #get_list_of_files()