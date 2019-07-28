"""
Client side code to perform a single API call to a tensorflow model up and running.
"""
import argparse
import json

import numpy as np
import requests
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import plot_util
from object_detection.utils import label_map_util
import object_detection.utils.ops as utils_ops
from PIL import Image



def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def pre_process(image_path):


    image = Image.open(image_path).convert("RGB")
    image_np = plot_util.load_image_into_numpy_array(image)

    # Expand dims to create  bach of size 1
    image_tensor = np.expand_dims(image_np, 0)
    formatted_json_input = json.dumps({"signature_name": "serving_default", "instances": image_tensor.tolist()})

    return formatted_json_input



def post_process(server_response, image_size):

    #print(server_response.text)
    response = json.loads(server_response.text)
    print(response )
    print("\n\n")
    output_dict = response['predictions'][0]
    #print(output_dict)
    names=["Google","Amazon","Paypal","Facebook","Dropbox"]
    for n,i in zip(names,range(5)):
        print(n+" : "+ str(   "{0:.9f}".format(output_dict[i]*100)  )+"%"    )


    # all outputs are float32 numpy arrays, so convert types as appropriate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs call to the tensorflow-serving REST API.')
    parser.add_argument('--server_url', dest='server_url', type=str, required=True,
                        help='URL of the tensorflow-serving accepting API call. '
                             'e.g. http://localhost:8501/v1/models/omr_500:predict')
    parser.add_argument('--image_path', dest='image_path', type=str,
                        help='Path to the jpeg image')


    # Map args to var
    args = parser.parse_args()
    server_url = args.server_url
    image_path = args.image_path


    # Build input data
    print(f'\n\nPre-processing input file {image_path}...\n')
    formatted_json_input = pre_process(image_path)
    print('Pre-processing done! \n')

    # Call tensorflow server
    headers = {"content-type": "application/json"}
    print(f'\n\nMaking request to {server_url}...\n')
    server_response = requests.post(server_url, data=formatted_json_input, headers=headers)
    #print(server_response)
    print(f'Request returned\n')



    # Post process output
    print(f'\n\nPost-processing server response...\n')
    image = Image.open(image_path).convert("RGB")
    image_np = load_image_into_numpy_array(image)
    output_dict = post_process(server_response, image_np.shape)
    #print(f'Post-processing done!\n')


