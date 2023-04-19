import pygame
import os
import neat

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome,config)
        #Core game code
        pygame.init()
        width = 1000
        height =  600
        screen_res = (width,height)
        pygame.display.set_caption("NEAT Pong Game")
        screen = pygame.display.set_mode(screen_res)
        red=(255,0,0)
        black=(0,0,0)
        white=(255,255,255)
        ball_obj = pygame.draw.circle(screen,white,center=(500,100),radius=10)
        #ball_obj.center = (500,100)
        paddle_obj = pygame.Rect(0,0,200,100)
        paddle_obj.center = (500,600)
        speed=[1,1]
        pspeed = [0,0]
        score = 0
        while True:
            #pygame.time.wait(1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
            
            screen.fill(black)

            #NEAT Inputs
            distx = paddle_obj.center[0] - ball_obj.center[0]
            disty = paddle_obj.center[1] - ball_obj.center[1]
            xspeed = speed[0]
            yspeed = speed[1]

            output=net.activate([distx,disty,xspeed,yspeed])

            #NEAT Outputs
            moveL = output[0]
            moveR = output[1]
            stay = output[2]

            if moveL > stay and moveL > moveR:
                pspeed[0]=-1
            elif moveR > stay and moveR > moveL:
                pspeed[0]=1
            elif stay > moveL and stay > moveR:
                pspeed[0]=0
            
            #NEAT rewards
            missball = -5
            hitball = 10

            ball_obj = ball_obj.move(speed)
            paddle_obj = paddle_obj.move(pspeed)
            #Checking for collisions
            if ball_obj.left<= 0 or ball_obj.right >=width:
                speed[0] = -speed[0]
            if ball_obj.top <= 0:
                speed[1] = -speed[1]
            if ball_obj.bottom >= height:
                score = score + missball
                genome.fitness = score
                break
            if paddle_obj.colliderect(ball_obj):
                score = score + hitball
                speed[1] = -speed[1]

            #Ending if too good
            if score > 1000:
                genome.fitness = score
                break
            pygame.draw.rect(screen,red,paddle_obj)
            pygame.draw.circle(screen,white,center=ball_obj.center,radius=40)

            pygame.display.flip()

def run(config_file):
    config=neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation,
                       config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    #Run for up to 30 generation
    winner = p.run(eval_genomes,30)
    #Display the winning genome
    print('\nBest genome:\n{!s}',format(winner))

    #Output the most fit genome against training data
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner,config)
    for xi,xo in zip(xor_inputs,xor_outputs):
        output=winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}",format(xi,xo,output))

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.ini")
    run(config_path)
