clear all
close all
clc

global N n q
N=10;  %number of agents;
n=2;  %number of dimension;
q=20;

x_ini=-3*ones(n,N);

%x_ini=-0.1*ones(n,N);
%x_ini=3*ones(n,N);

sigma_ini=-0.02*ones(q,N);%

opt=zeros(n,N);%





terminal=0.00001;


       

%[tn,xn,X_opt,sigma_opt,round]= Alg1(x_ini,sigma_ini,terminal)
%[tp,xp,X_opt,sigma_opt,round]= PGD(x_ini,sigma_ini,terminal)
%[to,xo,X_opt,sigma_opt,round]=penalty(x_ini,sigma_ini,terminal)
[ta,xa,X_opt,sigma_opt,round]= proximal(x_ini,sigma_ini,terminal)
%[te,xe,X_opt,sigma_opt,round]= SGD(x_ini,sigma_ini,terminal)
