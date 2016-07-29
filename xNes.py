#! /usr/bin/python
# -*- coding: utf-8 -*-
import gym
import numpy
from scipy.linalg import expm

def doOneIteration(env):
    Lambda = 8
    M = 5
    T = 200
    sigma = 0.5
    B = numpy.identity(4)
    I = numpy.identity(4)
    eta_sigma = eta_B = (3 * (3 + numpy.log(4))) / (5 * 4 * numpy.sqrt(4))
    weight = []
    for i in range(Lambda):
        denom = 0.0
        for j in range(Lambda):
            denom += max(0, numpy.log(Lambda/2 + 1) - numpy.log(j + 1))
        weight.append(max(0, numpy.log(Lambda/2 + 1) - numpy.log(i + 1))/denom - 1.0/Lambda)
    weight = numpy.asarray(weight)
    theta = 2 * numpy.random.rand(4) - 1
    theta_best = numpy.zeros(4)
    r_best = 0
    g = 0

    while g < 2000 and r_best < 200 * M:
        list = []
        for j in range(Lambda):
            z_j = numpy.random.randn(4)
            theta_j = theta + sigma * B.dot(z_j)
            list.append([z_j, theta_j, 0])

        for j in range(Lambda):
            for m in range(M):
                observation = env.reset()
                for t in range(T):
                    # env.render()
                    action = 0
                    if list[j][1].T.dot(observation) < 0:
                        action = 0
                    else:
                        action = 1
                    observation, reward, done, info = env.step(action)
                    list[j][2] += 1
                    if done:
                        break
        list.sort(key=lambda x: (x[2]), reverse=True)
        r_best = list[0][2]
        theta_best = list[0][1]
        sum1 = numpy.zeros(4)
        for j in range(Lambda):
            sum1 += weight[j] * list[j][0]
        G_m = sigma * B.dot(sum1)
        sum2 = numpy.zeros((4, 4))
        for j in range(Lambda):
            sum2 += weight[j] * (list[j][0].dot(list[j][0].T) - I)
        G_M = sum2
        G_sigma = numpy.trace(G_M)/4
        G_B = G_M - G_sigma * I
        theta += G_m
        sigma *= numpy.exp(eta_sigma * G_sigma / 2)
        B = B.dot(expm(eta_B * G_B / 2))
        g += 1
        print("g:{0},   r_best:{1},   theta:{2}".format(g, r_best, theta_best))
    if g >= 2000:
        return False
    else:
        return True


if __name__ == "__main__":

    success = 0
    fail = 0

    for i in range(1):
        env = gym.make('CartPole-v0')
        result = doOneIteration(env)
        if result:
            success += 1
        else:
            fail += 1
    print("success:{0},fail={1}".format(success, fail))
