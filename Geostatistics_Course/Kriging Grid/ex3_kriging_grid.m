%Areal precipitation
%interpolation of point information to grid
%
%exercise is modifed from:
%www.ce.utexas.edu/prof/maidment/ce394k/rainfall/rainfall.htm
%Thanks Christine and David!
%
%

clear all;
close all;

%data:
X=[
1      7000   4000   620.00
2      3000   4000   590.00
3     -2000   5000   410.00
4    -10000   1000   390.00
5     -3000  -3000  1050.00
6     -7000  -7000   980.00
7      2000  -3000   600.00
8      2000 -10000   410.00
9      0        0    810.00
];

stations = X(:,1);      % id-number of gauge stations
x_obs = X(:,2);         % x-coordinates
y_obs = X(:,3);         % y-coordinates
z_obs = X(:,4);         % precipitation (mm/y) 

%make a regular grid
%
%first, find coordinates of corner in grid
%let it be a small distance of 200 m outside min and max values
%
outside = 500; %you may change this later

minx = min(x_obs) - outside;
maxx = max(x_obs) + outside;
miny = min(y_obs) - outside;
maxy = max(y_obs) + outside;

%specify how dense (or sparse) you want to make the grid
%eg. 100 m spacing in x- and y- direction
%
dx = 100;
dy = 100;

%generate the coordinates of the regular grid:
[XI,YI]=meshgrid(minx:dx:maxx,miny:dy:maxy);

%interpolate precipitation by nearest neighbour method (similar to Theissen
%polygons
ZI_near = griddata(x_obs,y_obs,z_obs,XI,YI,'nearest');

%plot result
figure;
meshc(XI,YI,ZI_near), hold
plot3(x_obs,y_obs,z_obs,'o'), hold off
title('Nearest neighbour method');
colorbar;

%try other methods, eg. cubic
ZI_v4 = griddata(x_obs,y_obs,z_obs,XI,YI,'v4');

%plot result
figure;
meshc(XI,YI,ZI_v4), hold
plot3(x_obs,y_obs,z_obs,'o'), hold off
title('Matlab interpolation method v4');
colorbar;

%plot vaules in xy-grid
figure
imagesc(ZI_near);
title('Nearest neighbour method');
colorbar;

figure
imagesc(ZI_v4);
title('Matlab interpolation method v4'); % cubic spline
colorbar;

%compare the results
diff = ZI_near-ZI_v4;
%figure
%imagesc(diff);
%title('Difference between nearest neighbour method and v4');
%colorbar;

% ordinary kriging in all grid points

[n,m]=size(XI);
xo=reshape(XI,1,n*m);
yo=reshape(YI,1,n*m);

ones_row=ones(1,n*m);
x_obs_rows=x_obs*ones_row;   %which is a matrix of equal columns
y_obs_rows=y_obs*ones_row;   %which is a matrix of equal columns

x=[xo;x_obs_rows];
y=[yo;y_obs_rows];

[n,m]=size(x_obs);

%calculate distances between all observations
for i=1:n
  for j=i:n  %save a bit CPU because of symmetry
   rd(i,j)=sqrt((x_obs(i)-x_obs(j)).^2+(y_obs(i)-y_obs(j)).^2);
  end
end

rd=rd+rd'; %make symmetric matrix

%covariance model
C0=0;           %nugget, zero in example
C1=500 - C0;    %C1 + C0 = sill
a=10000;        %range
pr=log(1-0.95); %practical range at 95% of C1 + C0

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
% calculate distances between all grid points and all observations
tic; 
ro=find_norm(m1,m2); 
t3=toc;

%exponential semivariogram model
ego = C0 + C1*(1-exp(pr.*(ro/a))); 

%exponential covariance model
ecc = C0 + C1 - ego; 

%remember ones_row = ones(1,n*m);
eco=[ecc';ones_row];
  
%calculate kriging weights
We=(ech^-1)*eco;

[nn,mm]=size(z_obs);

%ordinary kriging BLUE-estimates
z_est=We(1:nn,:)'*z_obs;

[n,m]=size(XI);
z_kriging=reshape(z_est',n,m);

%plot result
figure;
meshc(XI,YI,z_kriging), hold
plot3(x_obs,y_obs,z_obs,'o'), hold off
title('Ordinary kriging')

figure
imagesc(z_kriging)
colorbar
title('Ordinary kriging')

%calculate kriging error

var0 = trace(ech(1:nn,1:nn))/nn;    % var0 = C0 + C1 if C0 and C1 are constants.
                                    % by sum(diag), measurement errors
                                    % may be added by adding a number
                                    % to diagonal elements
%kriging variance
for i=1:n*m
  sk(:,i) = var0 - We(:,i)'*eco(:,i);
end

%kriging error:
stdk=(sk).^(1/2);
kerror=reshape(stdk,n,m);


figure;
imagesc(real(kerror));     %in one point kriging variance is slightly negative
colorbar;
title('Kriging error')

%compare the kriging with nearest neighbour
diff2 = z_kriging-ZI_near;
%figure
%imagesc(diff2);
%title('Difference between ordinary kriging and nearest neigbour interpolation method');
%colorbar;

%compare the kriging with v4 method
diff3 = z_kriging-ZI_v4;
%figure
%imagesc(diff3);
%title('Difference between ordinary kriging and v4 interpolation method');
%colorbar;



%regression is equal to pure nugget 
%i.e. ecc=zeros everery where, cf. Goovaerts et al. 2005, p.7
[n,m] = size(ecc);
ecc_regression = zeros(n,m);
eco_regression = [ecc_regression';ones_row];
We_regression = (ech^-1)*eco_regression;
z_regression = We_regression(1:nn,:)'*z_obs;
[n,m] = size(XI);
pure_nugget_interpolation = reshape(z_regression',n,m);
% but it apply only a horizontal plane 

%this is better (c.f. help > multiple regression)
Xr = [ones(size(x_obs)) x_obs y_obs];
a = Xr\z_obs;                  
regression_surface_2D = a(1)+ a(2)*XI + a(3)*YI;
%or
%regression_surface_2D = Xr*a;

figure;
meshc(XI,YI,regression_surface_2D), hold
plot3(x_obs,y_obs,z_obs,'o'), hold off
title('Linear regression surface')


% and second order multiple regression 
% which you should avoid because of non-physical values in
% extrapolation domain
Xr2=[Xr,x_obs.^2,x_obs.*y_obs,y_obs.^2];
a2=Xr2\z_obs
ZI_reg2 = a2(1) + a2(2)*XI + a2(3)*YI + a2(4)*(XI.*XI) + a2(5)*(XI.*YI) + a2(6)*(YI.*YI);
figure;
meshc(XI,YI,ZI_reg2); hold;
plot3(x_obs,y_obs,z_obs,'o'), hold off
title('2.order multiple regression surface')


%include 2D trend (in a quasi-scientific way), 
%becuase in linear regression residuals are (in principle) random numers 
z_regression_points = a(1) + a(2)*x_obs + a(3)*y_obs;
% which is equivalent to
%z_regression_points = Xr*a;

z_residual = z_obs - z_regression_points;

% use same kriging weights as above
z_est_residual = We(1:nn,:)'*z_residual;  
	    % a better approach would be to calculate a new semivariogram
	    % based on the residuals.
z_est_residual_grid = reshape(z_est_residual,n,m);
z_est_m_trend = z_est_residual_grid + regression_surface_2D;


figure;
meshc(XI,YI,z_est_residual_grid), hold
plot3(x_obs,y_obs,z_residual,'o'), hold off
title('Ordinary kriging of residuals')


figure;
meshc(XI,YI,z_est_m_trend), hold
plot3(x_obs,y_obs,z_obs,'o'), hold off
title('Ordinary kriging with trend from linear regression')



%calculate average areal precipitation
fprintf('\nMean of observations:                    %.1f\n',mean(z_obs));
fprintf('Mean of pure nugget interpolation:       %.1f\n',sum(sum(pure_nugget_interpolation))/(n*m)); 
fprintf('Mean of linear regression surface:       %.1f\n',sum(sum(regression_surface_2D))/(n*m));
fprintf('Mean of nearest neigbour interpoloation: %.1f\n',sum(sum(ZI_near))/(n*m));
fprintf('Mean of MATLAB interpolation v4:         %.1f\n',sum(sum(ZI_v4))/(n*m));
fprintf('Mean of ordinary kriging interpolation:  %.1f\n',sum(sum(z_kriging))/(n*m));
fprintf('Mean of kriging with trend included   :  %.1f\n',sum(sum(z_est_m_trend))/(n*m));
