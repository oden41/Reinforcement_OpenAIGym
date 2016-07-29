import gym
import numpy

M = 5
T = 200
alpha = 0.1


def doOneIteration(env):
    theta = 2 * numpy.random.rand(4) - 1
    r_best = 0
    g = 0

    while g < 2000 and r_best < 200 * M:
        theta_new = theta.__add__(alpha * (2 * numpy.random.rand(4) - 1))
        r_total = 0
        for m in range(M):
            observation = env.reset()
            for t in range(T):
                # env.render()
                action = 0
                if theta_new.T.dot(observation) < 0:
                    action = 0
                else:
                    action = 1
                observation, reward, done, info = env.step(action)
                r_total += 1
                if done:
                    break
        if r_total > r_best:
            r_best = r_total
            theta = theta_new
        g += 1
    print("g:{0},   r_best:{1},   theta:{2}".format(g, r_best, theta))


env = gym.make('CartPole-v0')
doOneIteration(env)
