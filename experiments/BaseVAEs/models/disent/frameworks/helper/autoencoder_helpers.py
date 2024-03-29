# python 3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 17:01:37 2017

@author: Oscar Li

Source: https://github.com/OscarcarLi/PrototypeDL
"""
import os
import torch 

def list_of_distances(X, Y, norm='l2'):
    '''
    Given a list of vectors, X = [x_1, ..., x_n], and another list of vectors,
    Y = [y_1, ... , y_m], we return a list of vectors
            [[d(x_1, y_1), d(x_1, y_2), ... , d(x_1, y_m)],
             ...
             [d(x_n, y_1), d(x_n, y_2), ... , d(x_n, y_m)]],
    where the distance metric used is the sqared euclidean distance.
    The computation is achieved through a clever use of broadcasting.
    '''
    # XX = torch.reshape(list_of_norms(X), shape=(-1, 1))
    # YY = torch.reshape(list_of_norms(Y), shape=(1, -1))
    # output = XX + YY - 2 * torch.matmul(X, torch.transpose(Y))
    # return output
    
    XX = list_of_norms(X, norm=norm).view(-1, 1)
    YY = list_of_norms(Y, norm=norm).view(1, -1)
    return XX + YY - 2 * torch.matmul(X, torch.transpose(Y, 0, 1))

def list_of_norms(X, norm='l2'):
    '''
    X is a list of vectors X = [x_1, ..., x_n], we return
        [d(x_1, x_1), d(x_2, x_2), ... , d(x_n, x_n)], where the distance
    function is the squared euclidean distance.
    '''
    if norm == 'l2':
        return torch.sum(torch.pow(X, 2), axis=1)
    elif norm == 'l1':
        return torch.sum(torch.abs(X), axis=1)
    
    return None

