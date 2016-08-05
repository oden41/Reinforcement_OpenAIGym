#! /usr/bin/python
# -*- coding: utf-8 -*-
import gym
import numpy
# import csv


def doOneIteration(episode, env):
    theta = 2 * numpy.random.rand(4) - 1
    # theta = 2 * numpy.random.rand(4) - 10
    r_best = 0
    g = 0

    # f = open('data{0}.csv'.format(episode), 'ab')
    # csvWriter = csv.writer(f)

    while g < 2000 and r_best < 200 * M:
        # dataList = [g]
        theta_new = theta + alpha * (2 * numpy.random.rand(4) - 1)
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
        # dataList.append(r_best)
        # csvWriter.writerow(dataList)
        g += 1
    print("g:{0},   r_best:{1},   theta:{2}".format(g, r_best, theta))
    # f.close()
    if g >= 2000:
        return False
    else:
        return True


if __name__ == "__main__":
    M = 5
    T = 200
    alpha = 0.1
    success = 0
    fail = 0

    for i in range(100):
        env = gym.make('CartPole-v0')
        result = doOneIteration(i, env)
        if result:
            success += 1
        else:
            fail += 1
    print("success:{0},fail={1}".format(success, fail))
