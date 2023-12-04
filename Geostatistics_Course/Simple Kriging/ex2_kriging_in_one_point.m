% Areal precipitation
% interpolation of in one single location point
% 
%
%exercise is modifed from:
%www.ce.utexas.edu/prof/maidment/ce394k/rainfall/rainfall.htm
%Thanks Christine and David!
%
%

clear all;
close all;

% same data example as in compendium
Z=[
1      61   139   477
2      63   140   696
];

% estimation in:
xo = 65;
yo = 137;

stations = Z(:,1);      % id-number of gauge stations
x_obs = Z(:,2);         % x-coordinates
y_obs = Z(:,3);         % y-coordinates
z_obs = Z(:,4);         % precipitation (mm/y) 



% ordinary kriging in all grid points

[n,m] =size(x_obs);


[n,ff]=size(x_obs);

%calculate distances between all observations
for i=1:n
  for j=i:n  %save a bit CPU because of symmetry
   rd(i,j)=sqrt((x_obs(i)-x_obs(j)).^2+(y_obs(i)-y_obs(j)).^2);
  end
end

rd=rd+rd' %make symmetric matrix

%covariance model
C0=0;           %nugget, zero in example
C1=10 - C0;    %C1 + C0 = sill
a = 10;        %range
pr=log(1-0.95); %practical range at 95% of C1 + C0

%exponential semivariogram model
egh = C0 + C1*(1-exp(pr.*(rd/a)))

%exponential covariance model
ech = C0 + C1 - egh 


%add ones to calculate Lagrange multiplier

lagrange_row=ones(1,n); % a row of ones
ech=[ech;lagrange_row]

lagrange_column=[lagrange_row';0]; % remember zero in ch(n+1,n+1)

ech=[ech,lagrange_column] %and column of ones

% or calculate (and save) by:
 m1 = [xo',  yo'];
 m2 = [x_obs,y_obs];
%
% calculate distances between all grid points and all observations
tic; 
ro=find_norm(m1,m2) 
t3=toc;

%exponential semivariogram model
ego = C0 + C1*(1-exp(pr.*(ro/a))); 

%exponential covariance model
ecc = C0 + C1 - ego 



%remember ones_row = ones(1,n*m);
eco=[ecc';1]
  
%calculate kriging weights
X=(ech^-1)*eco

[n,m]=size(X);

lambda=X(1:n-1,1),

%ordinary kriging BLUE-estimates
z_est=lambda'*z_obs


%calculate kriging error


% if there are some noisy observations, a nugget value for these
% observations might be included in the covariance matrix, and the
% total variance is the average of the sum of the diagonal elements.
% But this is not the case for these observations, thus we do it the
% simpel way
varX = C0 + C1;

% this is a bit more advanced
varX = trace(ech(1:n-1,1:n-1))/(n-1);    % varX = C0 + C1 if C0 and C1 are constants.
                                    % by sum(diag), measurement errors
                                    % may be added by adding a number
                                    % to diagonal elements

Vark = varX - X'*eco; % kriging variance

%kriging error:
stdk=(Vark).^(1/2)


