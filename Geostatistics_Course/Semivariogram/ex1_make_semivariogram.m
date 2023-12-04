% compute experimental semivariogram map: ge(i,j)
% 
% dependencies: 
%   funk_semivar_mean_var.m
%
%
% input:
% Z1 - matrix of variable Z1, x-coord and y-coord (global coordinates please!) 
% Z2 - matrix of variable Z2, x-coord and y-coord (global coordinates please!) 
%
% if Z1 and Z2 are identical, then this is ordinary semivariogram map
% if Z1 and Z2 are different, then this is the cross-semivariogram map
%
% ant- number of classes in variogram (depending on how many observations you have)
% maxdist - set distance to avoid too few observations in each class
%

clear all;
close all;

X=load('precipitation.txt')

[n,f]=size(X);

u1 = X(:,2);
u2 = X(:,3);
z  = X(:,4);

figure
plot(u1,u2,'o')
axis equal
grid on
xlabel('x-coordinates')
ylabel('y-coordinates')
title('Precipitation excercise')

Z1=[u1, u2, z]; % input to semivariogram function


maxdist = 17.8e+03; % 17.8 km
ant = 8; % number of classes in semivariogram

[hegam_precipitation] = funk_semivar_mean_var(Z1,Z1,ant,maxdist);

hlag=hegam_precipitation(:,2)

C0=0.0;
C1=6.0e+04-C0
a=9000
pr =log(20); %=log(1/(1-.95))=practical range equal to 95% of total variance
eg = 1; % 1. order exponential
ch=semivar_mod(hlag,C0,C1,a,pr,eg)

figure
plot(hegam_precipitation(:,2),hegam_precipitation(:,3),'ob')
hold on
grid on

plot(hlag,(C1+C0)-ch,'r')
legend('experimental','model','Location','NorthEast')
xlabel('lagdistance, h (m)')
ylabel('\gamma(h)')
title('Semivariogram for precipitation data')