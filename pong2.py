import random
import pygame, sys
from pygame.locals import *
import numpy as np
import cv2
import queue

pygame.init()
fps = pygame.time.Clock()
#colors
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLACK = (0,0,0)
#globals
WIDTH = 600
HEIGHT = 400       
BALL_RADIUS = 10
PAD_WIDTH = 20
PAD_HEIGHT = 80
HALF_PAD_WIDTH = PAD_WIDTH / 2
HALF_PAD_HEIGHT = PAD_HEIGHT / 2
ball_pos = [0,0]
ball_vel = [0,0]
paddle1_vel = [0,0]
paddle2_vel = [0,0]
l_score = 0
r_score = 0

paddle1_queue = queue.Queue()
paddle2_queue = queue.Queue()

#canvas declaration
window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption('Hello World')
# helper function that spawns a ball, returns a position vector and a velocity vector
# if right is True, spawn to the right, else spawn to the left
def ball_init(right):
    global ball_pos, ball_vel # these are vectors stored as lists
    ball_pos = [WIDTH/2,HEIGHT/2]
    horz = random.randrange(2,4)
    vert = random.randrange(1,3)
    
    if right == False:
        horz = - horz
        
    ball_vel = [horz,-vert]
# define event handlers
def init():
    global paddle1_pos, paddle2_pos, paddle1_vel, paddle2_vel,l_score,r_score  # these are floats
    global score1, score2  # these are ints
    paddle1_pos = [HALF_PAD_WIDTH ,HEIGHT/2]
    paddle2_pos = [WIDTH - HALF_PAD_WIDTH,HEIGHT/2]
    l_score = 0
    r_score = 0
    if random.randrange(0,2) == 0:
        # ball spawns to the right
        ball_init(True)
    else:
        # ball spawns to the left
        ball_init(False)
#draw function of canvas
def draw(canvas):
    ret, frame = cap.read() 
    global paddle1_pos, paddle2_pos, ball_pos, ball_vel, l_score, r_score
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Display the resulting frame
    # creates the background image by using a frame
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)
    canvas.blit(frame, (0,0))
  
    pygame.draw.line(canvas, WHITE, [WIDTH / 2, 0],[WIDTH / 2, HEIGHT], 1)
    pygame.draw.line(canvas, WHITE, [PAD_WIDTH, 0],[PAD_WIDTH, HEIGHT], 1)
    pygame.draw.line(canvas, WHITE, [WIDTH - PAD_WIDTH, 0],[WIDTH - PAD_WIDTH, HEIGHT], 1)
    pygame.draw.circle(canvas, WHITE, [WIDTH//2, HEIGHT//2], 70, 1)
    # update paddle's vertical position, keep paddle on the screen
    if paddle1_pos[1] > HALF_PAD_HEIGHT and paddle1_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
        paddle1_pos[1] += paddle1_vel[1]
    elif paddle1_pos[1] == HALF_PAD_HEIGHT and paddle1_vel[1] > 0:
        paddle1_pos[1] += paddle1_vel[1]
    elif paddle1_pos[1] == HEIGHT - HALF_PAD_HEIGHT and paddle1_vel[1] < 0:
        paddle1_pos[1] += paddle1_vel[1]

    if paddle1_pos[0] > HALF_PAD_WIDTH and paddle1_pos[0] < WIDTH - HALF_PAD_WIDTH:
        paddle1_pos[0] += paddle1_vel[0]
    elif paddle1_pos[0] == HALF_PAD_WIDTH and paddle1_vel[0] > 0:
        paddle1_pos[0] += paddle1_vel[0]
    elif paddle1_pos[0] == WIDTH - HALF_PAD_WIDTH and paddle1_vel[0] < 0:
        paddle1_pos[0] += paddle1_vel[0]
    
    if paddle2_pos[1] > HALF_PAD_HEIGHT and paddle2_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
        paddle2_pos[1] += paddle2_vel[1]
    elif paddle2_pos[1] == HALF_PAD_HEIGHT and paddle2_vel[1] > 0:
        paddle2_pos[1] += paddle2_vel[1]
    elif paddle2_pos[1] == HEIGHT - HALF_PAD_HEIGHT and paddle2_vel[1] < 0:
        paddle2_pos[1] += paddle2_vel[1]

    if paddle2_pos[0] > HALF_PAD_WIDTH and paddle2_pos[0] < WIDTH - HALF_PAD_WIDTH:
        paddle2_pos[0] += paddle2_vel[0]
    elif paddle2_pos[0] == HALF_PAD_WIDTH and paddle2_vel[0] > 0:
        paddle2_pos[0] += paddle2_vel[0]
    elif paddle2_pos[0] == WIDTH - HALF_PAD_WIDTH and paddle2_vel[0] < 0:
        paddle2_pos[0] += paddle2_vel[0]
    print("s")
    print(paddle1_vel, paddle1_pos)
    print(paddle2_vel, paddle2_pos)
    print("e")
    #update ball
    ball_pos[0] += int(ball_vel[0])
    ball_pos[1] += int(ball_vel[1])
    #draw paddles and ball
    pygame.draw.circle(canvas, RED, (int(ball_pos[0]), int(ball_pos[1])), 20, 0)
    print([paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT], [paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT], [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT], [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT])
    pygame.draw.polygon(canvas, GREEN, [[paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT], [paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT], [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT], [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT]], 0)
    pygame.draw.polygon(canvas, GREEN, [[paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT], [paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT], [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT], [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT]], 0)
    #ball collision check on top and bottom walls
    if int(ball_pos[1]) <= BALL_RADIUS:
        ball_vel[1] = - ball_vel[1]
    if int(ball_pos[1]) >= HEIGHT + 1 - BALL_RADIUS:
        ball_vel[1] = -ball_vel[1]
    
    #ball collison check on gutters or paddles
    if int(ball_pos[0]) in range( int(paddle1_pos[0] - HALF_PAD_WIDTH - BALL_RADIUS), int(paddle1_pos[0] + HALF_PAD_WIDTH + BALL_RADIUS), 1) and int(ball_pos[1]) in range(int(paddle1_pos[1] - HALF_PAD_HEIGHT),int(paddle1_pos[1] + HALF_PAD_HEIGHT),1):
        ball_vel[0] = -ball_vel[0]
        ball_vel[0] *= 1.5
        ball_vel[1] *= 1.5
    elif int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH:
        r_score += 1
        ball_init(True)
        
    if int(ball_pos[0]) in range( int(paddle2_pos[0] - HALF_PAD_WIDTH - BALL_RADIUS), int(paddle2_pos[0] + HALF_PAD_WIDTH + BALL_RADIUS), 1) and int(ball_pos[1]) in range(int(paddle2_pos[1] - HALF_PAD_HEIGHT),int(paddle2_pos[1] + HALF_PAD_HEIGHT),1):
        ball_vel[0] = -ball_vel[0]
        ball_vel[0] *= 1.5
        ball_vel[1] *= 1.5
    elif int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH:
        l_score += 1
        ball_init(False)
    #update scores
    myfont1 = pygame.font.SysFont("Comic Sans MS", 20)
    label1 = myfont1.render("Score "+str(l_score), 1, (255,255,0))
    canvas.blit(label1, (50,20))
    myfont2 = pygame.font.SysFont("Comic Sans MS", 20)
    label2 = myfont2.render("Score "+str(r_score), 1, (255,255,0))
    canvas.blit(label2, (470, 20))  
    
    
#keydown handler
def keydown(event):
    global paddle1_vel, paddle2_vel
    
    if event.key == K_UP:
        paddle2_vel = -8
    elif event.key == K_DOWN:
        paddle2_vel = 8
    elif event.key == K_w:
        paddle1_vel = -8
    elif event.key == K_s:
        paddle1_vel = 8
#keyup handler
def keyup(event):
    global paddle1_vel, paddle2_vel
    
    if event.key in (K_w, K_s):
        paddle1_vel = 0
    elif event.key in (K_UP, K_DOWN):
        paddle2_vel = 0
init()
cap = cv2.VideoCapture(0)
#game loop
while cap.isOpened():
    draw(window)
    ret, frame = cap.read()
    fist_cascade = cv2.CascadeClassifier('fist.xml')
    palm_cascade = cv2.CascadeClassifier('palm.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    fists = fist_cascade.detectMultiScale(gray, 1.3, 5)
    palms = palm_cascade.detectMultiScale(gray, 1.3, 5)
    #Get move direction from one frame
    for (x, y, w ,h) in fists:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
        # left paddle
        paddle1_queue.put((x, y, w, h))

    for (x, y, w ,h) in palms:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
        # right paddle
        paddle2_queue.put((x, y, w, h))

    cv2.imshow('frame',frame)
        
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            keydown(event)
        elif event.type == KEYUP:
            keyup(event)
        elif event.type == QUIT:
            pygame.quit()
            sys.exit()

    preX = None
    preY = None
    xdirection = 0
    ydirection = 0
    if paddle1_queue.qsize() > 2:
        while paddle1_queue.qsize() > 0:
            x,y,w,h = paddle1_queue.get()
            if preX == None or preX < x:
                xdirection += 1
            elif preX == None or preX > x:
                xdirection -= 1
            if preY == None or preY < y:
                ydirection += 1
            elif preY == None or preY > y:
                ydirection -= 1
            preX = x
            preY = y
        if xdirection > 0:
            paddle1_vel[0] = 10
        elif xdirection < 0:
            paddle1_vel[0] = -10
        else:
            paddle1_vel[0] = 0
        if ydirection > 0:
            paddle1_vel[1] = 10
        elif ydirection < 0:
            paddle1_vel[1] = -10
        else:
            paddle1_vel[1] = 0

    preX = None
    preY = None
    xdirection = 0
    ydirection = 0
    if paddle2_queue.qsize() > 2:
        while paddle2_queue.qsize() > 0:
            x,y,w,h = paddle2_queue.get()
            if preX == None or preX < x:
                xdirection += 1
            elif preX == None or preX > x:
                xdirection -= 1
            if preY == None or preY < y:
                ydirection += 1
            elif preY == None or preY > y:
                ydirection -= 1
            preX = x
            preY = y
        if xdirection > 0:
            paddle2_vel[0] = 10
        elif xdirection < 0:
            paddle2_vel[0] = -10
        else:
            paddle2_vel[0] = 0
        if ydirection > 0:
            paddle2_vel[1] = 10
        elif ydirection < 0:
            paddle2_vel[1] = -10
        else:
            paddle2_vel[1] = 0

    pygame.display.update()
    fps.tick(60)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()