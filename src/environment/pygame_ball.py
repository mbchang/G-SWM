import math
import numpy as np
from pygame import Rect
import pygame

"""
From https://github.com/khuevu/pygames/tree/master/bouncingball
"""





class CollisionResponse(object):
    T_EPSILON = 0.00005
    def __init__(self, t=float('inf')):
        self.t =  t

    def get_new_x(self, current_speed_x, current_x):
        if self.t > self.T_EPSILON:
            return current_x + current_speed_x * (self.t - self.T_EPSILON)
        else:
            return current_x

    def get_new_y(self, current_speed_y, current_y):
        if self.t > self.T_EPSILON:
            return current_y + current_speed_y * (self.t - self.T_EPSILON)
        else:
            return current_y
        
    def get_exact_new_x(self, current_speed_x, current_x):
        return current_x + current_speed_x * self.t

    def get_exact_new_y(self, current_speed_y, current_y):
        return current_y + current_speed_y * self.t
    
    def reset(self):
        self.t = float('inf')
    

class Ball(object):
    max_speed_x = 5#float('-inf')
    max_speed_y = 5#float('-inf')

    def _set_speed(self, speed, angle):
        self.speed = speed
        self.angle = angle
        self.speed_x = math.cos(math.radians(self.angle)) * self.speed
        self.speed_y = math.sin(math.radians(self.angle)) * self.speed
    
    def __init__(self, x, y, speed, angle, r, color, name=''):
        self.name = name
        self.collision_response = CollisionResponse()
        self.x = x
        self.y = y
        self._set_speed(speed, angle)
        self.collision_response.new_speed_x = self.speed_x
        self.collision_response.new_speed_y = self.speed_y
        self.r = r
        self.color = color

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.r)

    # leapfrog integrator?
    def apply_accel(self, accel_x, accel_y, time):
        print('old speed x: {} old speed y: {} accel_x: {} accel_y: {}'.format(
            self.speed_x, self.speed_y, accel_x, accel_y))
        self.speed_x = accel_x * time + self.speed_x
        self.speed_y = accel_y * time + self.speed_y
        # technically you should just apply a force given speed and angle?
        print('new speed x: {} new speed y: {}'.format(self.speed_x, self.speed_y))

    def apply_friction(self, fric_coeff, time):
        fric_coeff = 0.01
        old_speed_x = self.speed_x
        old_speed_y = self.speed_y
        # calculate angle given self.speed_x and self.speed_y
        angle = math.atan2(self.speed_y, self.speed_x)
        # then calcualte friction
        fric_x = -math.cos(angle) * fric_coeff
        fric_y = -math.sin(angle) * fric_coeff
        # then apply accel
        self.apply_accel(fric_x, fric_y, time)
        # opposite sign
        if old_speed_x * self.speed_x < 0:  
            self.speed_x = 0
        if old_speed_y * self.speed_y < 0:
            self.speed_y = 0

    def update(self, time):
        if self.collision_response.t < time or math.fabs(self.collision_response.t - time) < CollisionResponse.T_EPSILON:
            self.x = self.collision_response.get_new_x(self.speed_x, self.x)
            self.y = self.collision_response.get_new_y(self.speed_y, self.y)
            print('old speed ', self.speed_x, self.speed_y)
            print('new speed', self.collision_response.new_speed_x, self.collision_response.new_speed_y)
            self.speed_x = self.collision_response.new_speed_x
            self.speed_y = self.collision_response.new_speed_y
        else:
            self.x = self.speed_x * time + self.x
            self.y = self.speed_y * time + self.y

        #reset response
        self.collision_response.t = float('inf')

    def detect_collision_with_other_ball(self, o, timeStep):
        #relative position: 
        collisibility = self._collision_time_with_other_ball(o, timeStep)

        #if self.tmp_collision_response.t - CollisionResponse.T_EPSILON > timeStep:
            #print self.name, 'not going to collide because of ', self.tmp_collision_response.t
            #return
        #if o.tmp_collision_response.t - CollisionResponse.T_EPSILON > timeStep:
            #print o.name, 'not going to collide because of ', o.tmp_collision_response.t
            #return

        if collisibility.t - CollisionResponse.T_EPSILON > timeStep:
            print('not going to collide')
            return
        print('COLLISION BETWEEN ', self.name, 'and ', o.name)
        print(self.name, 'v', self.speed_x, self.speed_y)
        print(o.name, 'v', o.speed_x, o.speed_y)
        self.collision_response.t = collisibility.t
        o.collision_response.t = collisibility.t
        collide_point_x = self.collision_response.get_exact_new_x(self.speed_x, self.x)
        collide_point_y = self.collision_response.get_exact_new_y(self.speed_y, self.y)
        o_collide_point_x = o.collision_response.get_new_x(o.speed_x, o.x)
        o_collide_point_y = o.collision_response.get_new_y(o.speed_y, o.y)
        rotate_angle = math.atan2(o_collide_point_y - collide_point_y, o_collide_point_x - collide_point_x)
        #calculate the speed component of balls in the new coordination
        speed_p = self.speed_x * math.cos(rotate_angle) + self.speed_y * math.sin(rotate_angle)
        speed_q = - self.speed_x * math.sin(rotate_angle) + self.speed_y * math.cos(rotate_angle)

        o_speed_p = o.speed_x * math.cos(rotate_angle) + o.speed_y * math.sin(rotate_angle)
        o_speed_q = - o.speed_x * math.sin(rotate_angle) + o.speed_y * math.cos(rotate_angle)

        #get new speed for each ball
        #assume the same mass density       
        mass_ratio = o.r ** 3 / self.r ** 3 
        new_speed_p = ((1 - mass_ratio) * speed_p + 2 * mass_ratio * o_speed_p) / (1 + mass_ratio)
        new_speed_q = speed_q
        print('new speed p, q', new_speed_p, new_speed_q)
        o_new_speed_p = (2 * speed_p + (mass_ratio - 1) * o_speed_p) / (1 + mass_ratio)
        o_new_speed_q = o_speed_q
        print('o_new_speed', o_new_speed_p, o_new_speed_q)
        self.collision_response.new_speed_x = new_speed_p * math.cos(-rotate_angle) + new_speed_q * math.sin(-rotate_angle)
        self.collision_response.new_speed_y = - new_speed_p * math.sin(-rotate_angle) + new_speed_q * math.cos(-rotate_angle)
        o.collision_response.new_speed_x = o_new_speed_p * math.cos(-rotate_angle) + o_new_speed_q * math.sin(-rotate_angle)
        o.collision_response.new_speed_y = - o_new_speed_p * math.sin(-rotate_angle) + o_new_speed_q * math.cos(-rotate_angle)
        print('after THE COLLISION')
        print(self.name, self.collision_response.new_speed_x, self.collision_response.new_speed_y)
        print(o.name, o.collision_response.new_speed_x, o.collision_response.new_speed_y)

    def _collision_time_with_other_ball(self, o, timeStep):
        x = self.x - o.x
        y = self.y - o.y
        r = self.r + o.r
        #relative speed of ball to other ball
        speed_x = self.speed_x - o.speed_x
        speed_y = self.speed_y - o.speed_y
        a = speed_x ** 2 + speed_y **2
        b = (x * speed_x + y * speed_y) * 2
        c = x **2 + y **2 - r ** 2
        delta = b ** 2 - 4 * a * c
        response = CollisionResponse()  
        if a == 0:
            if b != 0:
                t = -c / b
                if t > 0:
                    response.t = t
                    return response
            else:
                return response
        elif delta < 0:
            return response
        else:
            t1 = (- b - math.sqrt(delta)) / ( 2 * a )
            t2 = (- b + math.sqrt(delta)) / ( 2 * a )       
            if t1 > 0:
                response.t = t1
            elif t2 > 0:
                response.t = t2
            return response         
        
    def detect_collision_with_box(self, box, timeStep):
        #check vertical collision:
        collisibility = self.detect_collision_with_vertical_line(box.x, timeStep)
        if collisibility.t < self.collision_response.t:
            self.collision_response = collisibility
        collisibility = self.detect_collision_with_vertical_line(box.x + box.width, timeStep)
        if collisibility.t < self.collision_response.t:
            self.collision_response = collisibility
        #check horizontal collision
        collisibility = self.detect_collision_with_horizontal_line(box.y, timeStep)
        if collisibility.t < self.collision_response.t:
            self.collision_response = collisibility
        collisibility = self.detect_collision_with_horizontal_line(box.y + box.height, timeStep)
        if collisibility.t < self.collision_response.t:
            self.collision_response = collisibility
        
    def detect_collision_with_vertical_line(self, x, timeStep):
        if self.speed_x == 0:
            return CollisionResponse()
        if x > self.x:      
            distance = x - self.x - self.r
        else:
            distance = x - self.x + self.r
        time = distance / self.speed_x
        if time > 0 and (time < timeStep or math.fabs(time - timeStep) < CollisionResponse.T_EPSILON):
            response = CollisionResponse(time)
            response.t = time
            response.new_speed_x = - self.speed_x
            response.new_speed_y = self.speed_y
            return response
        else:
            return CollisionResponse()
            
    def detect_collision_with_horizontal_line(self, y, timeStep):
        if self.speed_y == 0:
            return CollisionResponse()
        if y > self.y:
            distance = y - self.y - self.r
        else:
            distance = y - self.y + self.r
        time = distance / self.speed_y
        if time > 0 and (time < timeStep or math.fabs(time - timeStep) < CollisionResponse.T_EPSILON): 
            response = CollisionResponse(time)
            response.new_speed_x = self.speed_x
            response.new_speed_y = - self.speed_y
            response.t = time
            return response
        else:
            return CollisionResponse()
    
    def log(self, description):
        print(description, self.name, self.x, self.y, self.speed_x, self.speed_y)
    
    def log_collision(self, description):
        print(description, self.name, self.collision_response.t, self.collision_response.new_speed_x, self.collision_response.new_speed_y)

# COLLISION TESTING
if __name__=="__main__":
    #ball goes against the wall
    b3 = Ball(20, 40, 3, 0, 5, None)
    box = Rect(0, 0, 400, 400)
    b3.detect_collision_with_box(box, 1000)
    #print 'collision time', b3.collision_response.t
    #print 'b3 new speed', b3.collision_response.new_speed_x, b3.collision_response.new_speed_y
    b3.update(b3.collision_response.t)
    #print 'b3 x, y', b3.x, b3.y
    b1 = Ball(300, 30, 5, -30, 15, None)
    box2 = Rect(5, 5, 490, 590)
    print('before collision', b1.speed_x, b1.speed_y)
    b1.detect_collision_with_box(box2, 1000)
    print('collision time ', b1.collision_response.t)
    print('b1 new speed', b1.collision_response.new_speed_x, b1.collision_response.new_speed_y)
    b1.update(1000)
    print('b1 after update',  b1.x, b1.y, b1.speed_x, b1.speed_y)
    b1.detect_collision_with_box(box2, 2)
    b1.log_collision('next collision')
    b1.update(1)
    b1.log('b1 next collision')
        
    
    
    
