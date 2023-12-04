%Areal precipitation
%interpolation of in one single location point
%
%exercise is modifed from:
%www.ce.utexas.edu/prof/maidment/ce394k/rainfall/rainfall.htm
%Thanks Christine and David!
%
%

clear all;
close all;

%covariance model
C0=0;           %nugget, zero in example
C1=500 - C0;    %C1 + C0 = sill
a=2000;        %range
pr=log(1-0.95); %practical range at 95% of C1 + C0

%data, in this case you leave out observation 8:
X=[
1      7000   4000   620.00
2      3000   4000   590.00
3     -2000   5000   410.00
4    -10000   1000   390.00
5     -3000  -3000  1050.00
6     -7000  -7000   980.00
7      2000  -3000   600.00
8      2000 -10000   410.00
%9      0        0    810.00
];

Xv = [9      0        0    810.00];
st = Xv(1,1)
xo = Xv(1,2)
yo = Xv(1,3)
zo = Xv(1,4)

stations = X(:,1);      % id-number of gauge stations
x_obs = X(:,2);         % x-coordinates
y_obs = X(:,3);         % y-coordinates
z_obs = X(:,4);         % precipitation (mm/y) 



% ordinary kriging

[n,m] =size(x_obs);


[n,ff]=size(x_obs);

%calculate distances between all observations
for i=1:n
  for j=i:n  %save a bit CPU because of symmetry
   rd(i,j)=sqrt((x_obs(i)-x_obs(j)).^2+(y_obs(i)-y_obs(j)).^2);
  end
end

rd=rd+rd'; %make symmetric matrix


%exponential semivariogram model
egh = C0 + C1*(1-exp(pr.*(rd/a))); 

%exponential covariance model
ech = C0 + C1 - egh; 

%add ones to calculate Lagrange multiplier

lagrange_row=ones(1,n); % a row of ones
ech=[ech;lagrange_row];

lagrange_column=[lagrange_row';0]; % remember zero in ch(n+1,n+1)

ech=[ech,lagrange_column]; %and column of ones

% or calculate (and save) by:
 m1 = [xo',  yo'];
 m2 = [x_obs,y_obs];
%
% calculate distances between all estimation points ( in this case
% only one, the left out observation) and all observations
tic; 
ro=find_norm(m1,m2); 
t3=toc;

%exponential semivariogram model
ego = C0 + C1*(1-exp(pr.*(ro/a))); 

%exponential covariance model
ecc = C0 + C1 - ego; 



%remember ones_row = ones(1,n*m);
eco=[ecc';1];
  
%calculate kriging weights
We=(ech^-1)*eco;

[nn,mm]=size(z_obs);

%ordinary kriging BLUE-estimates
z_est=We(1:nn,:)'*z_obs;


%%calculate kriging error


% if there are some noisy observations, a nugget value for these
% observations might be included in the covariance matrix, and the
% total variance is the average of the sum of the diagonal elements.
% But this is not the case for these observations, thus we do it the
% simpel way
var0 = C0 + C1;

% this is a bit more advanced
var0 = trace(ech(1:nn,1:nn))/nn;    % var0 = C0 + C1 if C0 and C1 are constants.
                                    % by sum(diag), measurement errors
                                    % may be added by adding a number
                                    % to diagonal elements
%this is one way to do it:
%%kriging variance
%for i=1:n*m
%  sk(:,i) = var0 - We(:,i)'*eco(:,i);
%end

sk = var0 - We'*eco;

%kriging error:
stdk=(sk).^(1/2);

%[st, xo/1000, yo/1000, zo, z_est, sk]
fprintf('\n%2d %5d %5d %d %f %f\n',st,xo,yo,zo,z_est, sk)