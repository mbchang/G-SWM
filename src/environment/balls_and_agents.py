"""
Same as balls_gen, but
- with friction
- applying a force on an object will move it
- there is a special set of dedicated objects for which you automatically know where to apply the force
"""

import argparse
from collections import namedtuple
import cv2
from math import fabs
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import pygame
from pygame import Color, Rect
import shutil
import sys
import time

from pygame_ball import Ball, CollisionResponse

"""
From https://github.com/khuevu/pygames/tree/master/bouncingball
"""

ColorRGBA = namedtuple('ColorRGBA', ('r', 'g', 'b', 'a'))
ImageSize = namedtuple('ImageSize', ('w', 'h'))
data_root = 'data'

def has_overlap(x1, y1, r1, x2, y2, r2):
    does_overlap = np.sqrt((x1-x2)**2 + (y1-y2)**2) <= (r1+r2)
    print('({} {} {} overlaps with {} {} {}. Resampling.'.format(x1, y1, r1, x2, y2, r2))
    return does_overlap

def mkdirp(logdir):
    try:
        os.makedirs(logdir)
    except FileExistsError:
        overwrite = 'o'
        while overwrite not in ['y', 'n']:
            overwrite = input('{} exists. Overwrite? [y/n] '.format(logdir))
        if overwrite == 'y':
            shutil.rmtree(logdir)
            os.mkdir(logdir)
        else:
            raise FileExistsError

class BallWorld(object):
    def __init__(self, screen_size, num_balls, radius_range, world_id, fric_coeff):
        pygame.init()
        assert radius_range[1] >= radius_range[0]
        assert num_balls+2 <= 1/radius_range[1]
        assert screen_size >= 100

        self.screen_size = screen_size
        self.screen_width = screen_size
        self.screen_height = screen_size
        self.num_balls = num_balls
        self.radius_range = radius_range
        self.world_id = world_id
        self.fric_coeff = fric_coeff
        self.colors = plt.cm.rainbow(np.linspace(0,1,20))*255

        mkdirp(os.path.join(data_root, str(world_id)))

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), 0, 32)
        self.clock = pygame.time.Clock()

        pad = 5
        self.border = Rect(pad, pad, self.screen_width-pad, self.screen_height-pad)
        self.balls = self.initialize_balls()

    def initialize_balls(self):
        balls = []
        for i in range(self.num_balls):
            r = np.random.randint(
                low=int(self.radius_range[0]*min(self.screen_width, self.screen_height)), 
                high=int(self.radius_range[1]*min(self.screen_width, self.screen_height))+1)
            padding = 2*r
            x = np.random.randint(padding, self.screen_width-padding)
            y = np.random.randint(padding, self.screen_height-padding)

            while any(has_overlap(x, y, r, b.x, b.y, b.r) for b in balls):
                x = np.random.randint(padding, self.screen_width-padding)
                y = np.random.randint(padding, self.screen_height-padding)

            angle = np.random.randint(0, 360)
            color = ColorRGBA(*[int(x) for x in self.colors[np.random.randint(len(self.colors))]])
            speed = np.random.randint(int(self.screen_size)/100, int(self.screen_size)/50)

            ball = Ball(x=x, y=y, speed=speed, angle=angle, r=r, color=Color(*color), name='ball{}'.format(i))
            balls.append(ball)
        return balls

    def update(self):
        timeStep = 1
        
        while timeStep > CollisionResponse.T_EPSILON:
            tMin = timeStep
            #check collision with other balls
            for i in range(len(self.balls)):
                for j in range(len(self.balls)):
                    if i < j:
                        self.balls[i].detect_collision_with_other_ball(self.balls[j], tMin)
                        self.balls[i].log_collision('1)')
                        self.balls[j].log_collision('2)')
                    if self.balls[i].collision_response.t < tMin:
                        tMin = self.balls[i].collision_response.t
            #check collision with box border:
            for b in self.balls:
                b.detect_collision_with_box(self.border, tMin)
                if b.collision_response.t < tMin:
                    tMin = b.collision_response.t   
            # # apply friction
            for b in self.balls:
                b.apply_friction(self.fric_coeff, tMin)
            for b in self.balls:
                b.log('ball ')
                b.update(tMin)
                b.log('ball after update')
            timeStep -= tMin
    
    def log(self, ball, description):
        print(description, 'x', ball.x, 'y', ball.y )

    def draw(self):
        pygame.draw.rect(self.screen, Color("black"), self.border)
        for b in self.balls:
            b.draw(self.screen)
    
    def quit(self):
        sys.exit()

    def run(self, num_frames, target_size):
        pygame.key.set_repeat(30, 30)

        subsample = 5
        t0 = time.time()
        for i in range(num_frames*subsample):
            self.update()
            self.draw()
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.quit()
            pygame.display.flip()   
            if i % subsample == 0:
                x = pygame.surfarray.array3d(self.screen)
                x = cv2.resize(x, target_size)
                plt.imsave(os.path.join(data_root, str(self.world_id), '{}.png'.format(i//subsample)), x)
                plt.close()
        print('That took {}'.format(time.time()-t0))

def generate_data(num_videos):
    for i in range(num_videos):
        bw = BallWorld(screen_size=args.screen_size, num_balls=args.num_balls, radius_range=args.radius_range, world_id=i, fric_coeff=args.fric_coeff)
        bw.run(num_frames=args.num_frames, target_size=ImageSize(w=args.imsize, h=args.imsize))
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_videos', type=int, default=10000)
    parser.add_argument('--screen_size', type=int, default=100)
    parser.add_argument('--num_balls', type=int, default=4)
    parser.add_argument('--radius_range', nargs='+', type=float, default=(0.1, 0.1))
    parser.add_argument('--fric_coeff', type=float, default=0.01)
    parser.add_argument('--num_frames', type=int, default=20)
    parser.add_argument('--imsize', type=int, default=64)
    args = parser.parse_args()

    generate_data(args.num_videos)
