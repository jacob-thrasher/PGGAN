from cv2 import log
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import requests
from network import BATCH_SIZE, LEAKY_SLOPE, IMG_SIZE, SCALE, LATENT, F_MAPS

def show_image(img):
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

def save_graph(title, x_label, y_label, epoch, list1, list1_label, list2=None, list2_label=None):
    # title = title + "_at_epoch_{:04}".format(epoch)

    plt.figure(figsize=(10, 3)) 
    plt.title(title)
    plt.plot(list1, label=list1_label)

    if list2 is not None:
        plt.plot(list2, label=list2_label)
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    # plt.show()

    #TODO: Save to folder bc right now it isn't working for some reason
    filename = "DCGAN-torch/metrics/" + title + "_epoch_{:04}.png".format(epoch+1)
    # dir = os.path.join("metrics/"+filename)
    
    plt.savefig(filename)
    plt.close()

def save_images(images, epoch, n_cols=None):
  """Displays multiple images in grid format"""

  n_cols = n_cols or len(images)
  n_rows = (len(images) - 1) // n_cols + 1
  if images.shape[-1] == 1:
      images = np.squeeze(images, axis=-1)
  plt.figure(figsize=(n_cols, n_rows))
  for index, image in enumerate(images):

      image_adjusted = (image * 127.5 + 127.5) / 255.

      plt.subplot(n_rows, n_cols, index + 1)
      plt.imshow(image_adjusted.permute(1, 2, 0), cmap='binary')
      plt.axis("off")
    
  filename = "DCGAN-torch/grids/image_epoch_{:04}.png".format(epoch+1)
  plt.savefig(filename)
  plt.close()
  print("[UPDATE] Image grid saved\n")

def create_logfile(dir, filename):
    prefix = filename
    filename = filename + '.log'
    _path = os.path.join(dir, filename)

    count = 0
    while os.path.exists(_path):
        filename = prefix + '_' + str(count) + '.log'
        _path = os.path.join(dir, filename)
        count += 1

    logging.basicConfig(filename=_path, encoding='utf-8', level=logging.INFO)
    logging.info("START")

def send_telegram_msg(msg, id, token):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {"chat_id": id, "text": msg}
    r = requests.get(url, params=params)