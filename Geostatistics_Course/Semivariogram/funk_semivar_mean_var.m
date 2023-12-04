% use:
% [hegam] = funk_semivar_mean_var(Z1,Z2,ant,maxdist) 
%
% function that compute experimental semivariogram (ge):
%     
% ge = [1/{2N(h)}]*sum_(i,j in R(h)) [Z(u_i) - Z(u_j)]^2
%
% where R(h) = {de <= | u_i - u_j | <= de} for i,j = N
% N is total number of observations, 
% de is the delta-lag distance, and
% N(h)number is number of observations in each semivariogram class
%
% input:
% Z1 - matrix of variable Z1, x-coord and y-coord (global coordinates please!) 
% Z2 - matrix of variable Z2, x-coord and y-coord (global coordinates please!) 
%
% if Z1 and Z2 are identical, then this is ordinary semivariogram
% if Z1 and Z2 are different, then this is the cross-semivariogram
%
% ant- number of classes in variogram (depending on how many observations you have)
% maxdist - set distance to avoid too few observations in each class
%
% output:
% hegam(:,1)  -  number of classes  
% hegam(:,2)  -  lag distance
% hegam(:,3)  -  intrinsic value
% hegam(:,4)  -  number of obs in variogram class
% hegam(:,5)  -  mean of observations in class
% hegam(:,6)  -  variance of observations in class    obs!
% hegam(:,7)  -  std(intrinsic)
% hegam(:,8)  -  min(intrinsic)
% hegam(:,9)  -  max(intrinsic)
%
% 

function[hegam] = funk_semivar_mean_var(Z1,Z2,ant,maxdist) 

[N1,ff]=size(Z1);

xobs1 = Z1(:,1);            % x (global) coordinates of variable 1
yobs1 = Z1(:,2);            % y (global) coordinates of variable 1
zo1   = Z1(:,3);            % variable 1

xobs1=xobs1-min(xobs1);     % let origo be minimum x and y coordinates
yobs1=yobs1-min(yobs1);     % local coordinates

%second variable 
[N2,ff]=size(Z2);

xobs2 = Z2(:,1);            % x (global) coordinates of variable 2
yobs2 = Z2(:,2);            % y (global) coordinates of variable 2
zo2   = Z2(:,3);            % variable 2

xobs2=xobs2-min(xobs2);     % let origo be minimum x and y coordinates
yobs2=yobs2-min(yobs2);     % local coordinates

k=0;
tic;
fprintf('be patient!\n');

% find max distance in sample
xobs = [xobs1; xobs2];
yobs = [yobs1; yobs2];

dmx = (max(xobs) - min(xobs));
dmy = (max(yobs) - min(yobs));

%maximum_distance = sqrt(dmx^2 + dmy^2);

% small distance which is the first distance in semivariogram class #1
% remember consistant units, e.g. in meter, input in function!
liten = .1; %10 cm distance between wells is VERY SMALL

% compute variogram intervalls ( de ) in the formula at top

var_intervall = maxdist/ant;

hegam = zeros(ant+2, 12);


% to initiate min values;
hegam(:,7) = max([zo1;zo2]);

%hegam(:,9) = max([zo1;zo2]);  %cancel this table

nugget_distance = var_intervall - liten;

for t = 1:ant+2
  hegam(t,1)=t;
  hegam(t,2)=var_intervall*t - nugget_distance;
end;


for i = 1:N1
  fprintf('%.2f\n',100*i/N1);

    for j = i:N2

       dx=(xobs1(i)-xobs2(j));
       dy=(yobs1(i)-yobs2(j));

       h = sqrt(dx^2 + dy^2);

       if (h < maxdist)

         % intrinsic hypothesis:
         % assumes stasjonarity in space with respect 
         % to differences of observations 

         intrin  = (zo1(i) - zo2(j))^2;

         % put intrinsic value in correct class

         t=1; 
         while (h > hegam(t+1,2))
           t=t+1;
         end;

         if ( t < ant+2)   % if distance exceeds max variogram distance
           hegam(t,3) = hegam(t,3) + intrin;
           hegam(t,4) = hegam(t,4) + 1;
           hegam(t,5) = hegam(t,5) + zo1(i) + zo2(j); 
	   % var(z) = E(z^2) - my^2
	   hegam(t,6) = hegam(t,6) + zo1(i).^2 + zo2(j).^2; 
	   
           hegam(t,7) = hegam(t,7) + intrin*intrin;

           if (hegam(t,8)  > intrin) hegam(t,8)  = intrin; end
           if (hegam(t,9)  < intrin) hegam(t,9)  = intrin; end

         end;                 % end if-test for max variogram class

      end;                    % check for max dist

    end;                      % end j-loop

end;                          % end i-loop

time=toc

% mean intrinsic value, which I use for std calculation 
hegam(:,3)=hegam(:,3)./hegam(:,4);

% mean observation of observations in class h
hegam(:,5)=hegam(:,5)./hegam(:,4);

% variance of observations in class h, identical to matlab function var(z)
%var(z) = sum(z^2)/(n-1) + mean(z)*(n/n-1)
hegam(:,6)=hegam(:,6)./(hegam(:,4)-1) - (hegam(:,5).^2).*(hegam(:,4)./hegam(:,4)-1);

% std of intrinsic value (I don't remember why I included this, never mind!)
hegam(:,7)=(hegam(:,7) - hegam(:,4).*(hegam(:,3).^2))./hegam(:,4);
hegam(:,7)=sqrt(hegam(:,7));

% and finally the semivariogram:
hegam(:,3)=hegam(:,3)./2;

