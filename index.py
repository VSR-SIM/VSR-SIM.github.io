import datetime
from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image, ImageOps, ImageDraw, ImageFont, ExifTags
import numpy as np
from io import BytesIO
import base64
import re
import datetime

import skimage.io
import time
import torch

from NNfunctions import *
opt = GetOptions_allRnd_0317()
net = LoadModel(opt)
# opt = []
# net = []

resizeBool = True

app = Flask(__name__, template_folder='.')


def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'png')
    # img_io.seek(0)
    # return send_file(img_io, mimetype='image/png')

    img_str = base64.b64encode(img_io.getvalue(),quality=100)
    return img_str

def serve_pil_images(pil_img_list):
    img_dict = {}
    for idx,img in enumerate(pil_img_list):
        # img_io = BytesIO()
        # img.save(img_io, 'png')
        # img_dict[str(idx)] = base64.b64encode(img_io.getvalue())
        dt = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3]
        img.save('static/images/data/%s.jpg' % dt, quality=100)
        img_dict[str(idx)] = dt
    return jsonify(**img_dict)


def add_watermark(im):
    width, height = im.size

    draw = ImageDraw.Draw(im)
    font1 = ImageFont.truetype("fonts/Montserrat-Light.ttf", 40)
    font2 = ImageFont.truetype("fonts/Montserrat-SemiBold.ttf", 40)

    textwidth, textheight = draw.textsize("Image", font1)

    # calculate the x,y coordinates of the text
    margin = 5
    x = margin
    y = height - textheight - margin

    # draw watermark in the bottom right corner
    draw.text((x+1, y+1), "ML-SIM", font=font1, fill=(0,0,0,255))
    draw.text((x, y), "ML-SIM", font=font1,fill=(255,255,255,255))
    return im


@app.route("/images")
def images():
    return send_file('ML-SIM Test Images.zip',as_attachment=True)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def main(path):
    if path == '':
        return render_template('index.html', title='ML-SIM Frontpage')
    elif path == 'generic.html':
        return render_template('generic.html', title='ML-SIM Sign-up')
    else:
        return "<h1>Woops, we don't know that path. Why don't you try <a href='http://ML-SIM.com'>our frontpage</a></h1>"

@app.route("/testform",methods=['POST','GET'])
def test():

    print('Received request')
    t0 = time.perf_counter()

    vals = request.values
    file = request.files['file']
    for val in vals:
        print(val)

    outfile = 'static/images/data/' + datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3]

    ## commands

    cmd = vals['cmd']
    if cmd == 'superres':
        print('in superres',file)
        file.save('%s.tif' % outfile)
        img = skimage.io.imread('%s.tif' % outfile)
        sr,wf,out = EvaluateModel(net,opt,img,outfile)
        img_dict = {}
        img_dict['0'] = sr
        img_dict['1'] = wf
        img_dict['2'] = out
        torch.cuda.empty_cache()
    else:
        print('unknown cmd')
        return 0

    print('Finished processing',time.perf_counter()-t0)

    return jsonify(**img_dict)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)


