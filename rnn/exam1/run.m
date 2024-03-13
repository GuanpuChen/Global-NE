clear all
close all
clc

global N n q
N=2;  %number of agents;
n=2;  %number of dimension;
q=1;

x_ini=-4*ones(n,N);
x_ini=[4,-4;4,-4]
x_ini=[0,-3;0,-3]
x_ini=[2,-2.5;2,-2.5]
x_ini=[-4,-0.5;-4,-0.5]

%x_ini=[0.5,4.25;0.5,4.25]

x_ini=[-1.5,3.5;-1.5,3.5]
x_ini=[3.5,4.25;3.5,4.25]

%x_ini=3*ones(n,N);

sigma_ini=-0.02*ones(q,N);%

opt=zeros(n,N);%




terminal=0.0001;


       
%[x11,x22,X_opt,sigma_opt,round]= Alg1(x_ini,sigma_ini,terminal)
[x33,x44,X_opt,round]= GD(x_ini,terminal)
%[x55,x66,X_opt,round]= OMD(x_ini,terminal)
%[x77,x88,X_opt,round]= EG(x_ini,terminal)
