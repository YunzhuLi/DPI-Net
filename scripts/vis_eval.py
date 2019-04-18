import os
import cv2
import numpy as np
import imageio
import argparse
import scipy.misc

### Usage
# python vis_eval.py --env FluidShake \
#   --src_dir dump_FluidShake/eval_FluidShake/rollout_0 \
#   --st_idx 150 --ed_idx 300

parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', default='')
parser.add_argument('--format', default='gif')
parser.add_argument('--env', default='')
parser.add_argument('--st_idx', type=int, default=0)
parser.add_argument('--ed_idx', type=int, default=0)
parser.add_argument('--height', type=int, default=240)
parser.add_argument('--width', type=int, default=320)

args = parser.parse_args()

if args.env == 'FluidFall':
    st_x, ed_x = 60, 210
elif args.env == 'BoxBath':
    st_x, ed_x = 65, 215
elif args.env == 'FluidShake':
    st_x, ed_x = 40, 180
elif args.env == 'RiceGrip':
    st_x, ed_x = 90, 230

images = []
for i in range(args.st_idx, args.ed_idx):

    filename = os.path.join(args.src_dir, 'pred_%d.tga' % i)
    print(filename)
    img = scipy.misc.imread(filename)
    img = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_AREA)

    images.append(img[st_x:ed_x])

imageio.mimsave(args.src_dir + '.gif', images, duration=1./60.)

