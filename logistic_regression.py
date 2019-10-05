#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 13:39:54 2018

@author: ls
"""
###
print('>==starting the logistic regression==<')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')
print('the basic packages have been imported')
parser=argparse.ArgumentParser(description='logistic regression for data')
parser.add_argument('-inf','--input_file',type=str,help='input file')
parser.add_argument('-s','--sep',type=str,help='the seperator of a file',choices=['comma','tab'])
parser.add_argument('-itcep','--intercept',type=float,help='the intercept of model')
parser.add_argument('-it','--iteration',type=int,help='the iteration steps in learning')
parser.add_argument('-al','--alpha',type=float,help='the learn rate of the logistic regression')
parser.add_argument('-bch','--batchsize',type=int,help='the batchsize of samples')
parser.add_argument('-tds','--theta_distribution',type=str,help='display the distribution of theta',choices=['yes','no'])
parser.add_argument('-rf','--result_file',type=str,help='the result directory')
parser.add_argument('-cp','--cost_plot',type=str,help='disply the cost plot or not',choices=['yes','no'])
parser.add_argument('-vf','--valid_file',type=str,help='the valid_file we need')
parser.add_argument('-rg','--regression',type=str,help='type of the regression',choices=['line','logistic'])
parser.add_argument('-im','--initial_model',type=str,help='the way we initiate the model',choices=['zeros','random'])
args=parser.parse_args()
input_file=args.input_file
sep=args.sep
intercept=args.intercept
regression=args.regression
result=args.result_file
if regression=='logistic':
    print('starting logistic regression')
    stop=args.iteration
    alpha=args.alpha
    batchsize=args.batchsize
    theta_dis=args.theta_distribution  
    cost_plot=args.cost_plot
    valid_file=args.valid_file
    init_model=args.initial_model
    print('the parameters for logistic regression have been accepted ')
    #result='machine_learning/'
    def read_file(input_file,sep):
        if sep=='comma':       
            data=pd.read_csv(input_file,sep=',')
        if sep=='tab':
            data=pd.read_csv(input_file,sep='\t')
        return data
    def display(input_file):
        if sep=='comma':       
            data=pd.read_csv(input_file,sep=',')
        if sep=='tab':
            data=pd.read_csv(input_file,sep='\t')
        names=data.columns.tolist()
        data_negative=data.loc[data[names[2]]==0]
        data_positive=data.loc[data[names[2]]==1]
        plt.scatter(data_negative[names[0]],data_negative[names[1]],marker='x',color='red',s=12)
        plt.scatter(data_positive[names[0]],data_positive[names[1]],marker='o',color='blue',s=12)
        plt.savefig(result+'display.svg')
        plt.savefig(result+'display.png')
        plt.clf()
    display(input_file)
    print('the raw data has been displayed')
    data=read_file(input_file,'comma')
    print('data has been loaded')
    def insert_intercept():
        global data, intercept
        data.insert(0,'intercept',float(intercept))
        data=data.values
        x=data[:,0:data.shape[1]-1]
        y=data[:,data.shape[1]-1:data.shape[1]]
        return x,y
    x,y=insert_intercept()
    def init_theta(init_model):
        if init_model=='zeros':       
            theta=np.zeros([1,x.shape[1]])
        if init_model=='random':
            theta=np.random.random_sample((1,x.shape[1]))
        return(theta)
    theta=init_theta(init_model)
    def sigmoid(x,theta):
        return 1/(1+np.exp(-np.dot(x,theta.T)))
    def cost(x,y,theta):
        left=np.multiply(y-1,np.log(1-sigmoid(x,theta)))
        right=np.multiply(y,np.log(sigmoid(x,theta)))
        result=np.sum(left-right)/len(y)
        return result
    def gradient(x,y,theta):
        grad=np.zeros(theta.shape)
        for i in range(theta.shape[1]):
            grad[0,i]=np.sum(np.multiply((sigmoid(x,theta)-y),x[:,i:i+1]))/len(y)
        return grad
    def shuffledata(data):
        np.random.shuffle(data)
        cols=data.shape[1]
        x=data[:,0:cols-1]
        y=data[:,cols-1:]
        return x,y
    def decent_iteration(x,y,theta,threshold,alpha,batchsize):
            i=0
            k=0
            x,y=shuffledata(data)
            grad=np.zeros(theta.shape)
            costs=[cost(x,y,theta)]
            for i in range(threshold):
                grad=gradient(x[k:k+batchsize],y[k:k+batchsize],theta)
                k+=batchsize
                if k>=x.shape[0]:
                    k=0
                    x,y=shuffledata(data)            
                theta=theta-alpha*grad
                costs.append(cost(x,y,theta))
                i+=1
            return theta ,costs
    def decent_cost(x,y,theta,threshold,alpha,batchsize):
            i=0
            k=0
            x,y=shuffledata(data)
            grad=np.zeros(theta.shape)
            costs=[cost(x,y,theta)]
            while True:
                grad=gradient(x[k:k+batchsize],y[k:k+batchsize],theta)
                k+=batchsize
                if k>=x.shape[0]:
                    k=0
                    x,y=shuffledata(data)            
                theta=theta-alpha*grad
                costs.append(cost(x,y,theta))
                i+=1
                if costs[-2]-costs[-1]<threshold:
                    break
                elif i>1e6:
                    exit('iteration has reached 1e6')
                else:
                    continue
    print('starting the gradient descending')
    theta_final,cost_final=decent_iteration(x,y,theta,stop,alpha,batchsize)
    if theta_dis=='yes':
        list_theta=theta_final.tolist()
        with open(result+'theta.txt','w') as file:
            for line in list_theta: 
                for j in line:
                    file.write(str(j))
    if cost_plot=='yes':  
        plt.plot(cost_final)
        plt.savefig(result+'cost.png')
        plt.savefig(result+'cost.svg')
        plt.clf()
    if valid_file!=None:
        valid_dataframe=pd.read_csv(valid_file,sep=',')
        valid_dataframe.insert(0,'intercept',float(intercept))
        valid_dataframe=valid_dataframe.values
        pred_list=[]
        for i in range(valid_dataframe.shape[0]):
            temp=valid_dataframe[i,:-1]
            pred=sigmoid(temp,theta_final)
            pred_list.append(pred)
        with open(result+'logistic_prediction.txt','w') as file:
            for i in pred_list:               
                    file.write(str(i)+'\n')
#line regression
elif regression=='line':
    if sep=='comma':
        line_file=pd.read_csv(input_file,sep=',')  
    elif sep=='tab':
        line_file=pd.read_csv(input_file,sep='\t')
    line_file.insert(0,'intercept',float(intercept))
    line_matrix=line_file.values
    linex,liney=line_matrix[:,0:line_matrix.shape[1]-1],line_matrix[:,line_matrix.shape[1]-1:line_matrix.shape[1]]
    theta=np.linalg.inv(np.matmul(linex.T,linex))
    theta=np.matmul(np.matmul(theta,linex.T),liney)
    print(theta)
    predict=np.matmul(linex,theta)
    predict.tolist()
    with open(result+'line_prediction.txt','w') as file:
            for i in predict:
                for j in i:               
                    file.write(str(j)+'\n')
print('analysis has been finished')