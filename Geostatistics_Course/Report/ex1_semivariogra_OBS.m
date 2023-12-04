% compute experimental semivariogram map: ge(i,j)
% 
% dependencies: 
%   funk_semivar_mean_var.m
%
%
% input:
% Z1 - matrix of variable Z1, x-coord and y-coord (global coordinates please!) 
% Z2 - matrix of variable Z2, x-coord and y-coord (global coordinates please!) 
% %
% if Z1 and Z2 are identical, then this is ordinary semivariogram map
% if Z1 and Z2 are different, then this is the cross-semivariogram map
%
% ant- number of classes in variogram (depending on how many observations you have)
% maxdist - set distance to avoid too few observations in each class
%

clear all;
close all;

fn_grunnvbronn = 'Grunnvannsborehull\GrunnvannBronn_20221130.shp';
fn_energibronn = 'Grunnvannsborehull\EnergiBronn_20221130.shp';

fn_bronnpark   = 'Grunnvannsborehull\BronnPark_20221130.shp';
fn_oppkomme    = 'Grunnvannsborehull\GrunnvannOppkomme_20221130.shp';
fn_sonderbor   = 'Grunnvannsborehull\Sonderboring_20221130.shp';


GRV = readgeotable(fn_grunnvbronn);
ENB = readgeotable(fn_energibronn);
		   	      
BRP = readgeotable(fn_bronnpark);
OPK = readgeotable(fn_oppkomme);
SND = readgeotable(fn_sonderbor);


% Make a vertical correction for lenght to bedrock if borehole is not vertical:
% geographical coordinates should also be 'megrated' according to
% azimuth, but this is not (yet) done
vdGRV = cosd(double(GRV.bhelnigrad)).*GRV.blengdber_; % max 4.247 m correction
vdENB = cosd(double(ENB.bhelnigrad)).*ENB.blengdber_; % max 2 m correction

% horizontal 'migration' of x- and y- coordinates according to
% boretAzimut

hdGRV = sind(double(GRV.bhelnigrad)).*GRV.blengdber_; % max 10.253 m horizontal correction
hdENB =	sind(double(ENB.bhelnigrad)).*ENB.blengdber_; % max 3.4641 m horizontal correction

dxGRV=hdGRV.*sind(double(GRV.bazimuth)); % max  4.436 m
dyGRV=hdGRV.*cosd(double(GRV.bazimuth)); % max 10.2530 m

dxENB=hdENB.*sind(double(ENB.bazimuth)); % max  0.7814 m
dyENB=hdENB.*cosd(double(ENB.bazimuth)); % max  3.4641 m



%% Section 2 Creation of a meshgrid 100 square meters big

% coordinates for the 'window' you want to 'crop out' from the dataset
wminx = 254100     
wmaxx = 268000     

wminy = 6620100
wmaxy = 6628700

wxlength = wmaxx-wminx
wylength = wmaxy-wminy

% make a meshgrid
dx=100
dy=100

wxvec=[wminx:dx:wmaxx];
wyvec=[wminy:dy:wmaxy];

[wxgrid,wygrid] = meshgrid(wxvec,wyvec);

% Make a grid with the square size defined from dx dy and wmin wmax for x
% and y
[nx,my] = size(wxgrid); % find size of grid


[ngrv,m] = size(GRV);


%% Section 3 Data Preprocessing: Migrate data, Filter away unwanted data

% insert "migrated" data in the dataseries 
% vertical migration
GRV.blengdber_ = vdGRV;
ENB.blengdber_ = vdENB;

% horizontal migration

grvx = GRV.Shape.X + dxGRV;
grvy = GRV.Shape.Y + dyGRV;
%
enbx = ENB.Shape.X + dxENB;
enby = ENB.Shape.Y + dyENB;


% find groundwater wells drilled in sediments or in bedrock
[sed_wells,fcc] = find(GRV.geolmedium == 'Løsmass');
%[bdr_wells,fcc] = find(GRV.geolmedium == 'Fjell');  

GRVsed = GRV(sed_wells,:);
%GRVbdr = GRV(bdr_wells,:);  %I don't need it

% IMPORTANT: cancel all boreholes with blengdber_ == NaN or blengdber_ == 0.

[i,j] = find(GRV.blengdber_ ~= NaN & GRV.blengdber_ > 0);

OBS_XYZ = [grvx(i), grvy(i), GRV.blengdber_(i)];

[i,j] = find(ENB.blengdber_ ~= NaN & ENB.blengdber_ > 0);

OBS_XYZ = [OBS_XYZ; enbx(i), enby(i), ENB.blengdber_(i)];


% IMPORTANT: filter out wells outside area of interest

[i,j] = find(OBS_XYZ(:,1) > wminx);
OBS_XYZ = OBS_XYZ(i,:);

[i,j] = find(OBS_XYZ(:,1) < wmaxx);
OBS_XYZ = OBS_XYZ(i,:);

[i,j] = find(OBS_XYZ(:,2) > wminy);
OBS_XYZ = OBS_XYZ(i,:);

[i,j] = find(OBS_XYZ(:,2) < wmaxy);
OBS_XYZ = OBS_XYZ(i,:);

% From this, The Borehole data is saved in OBS_XYZ,  with its X, Y and Z
% coordinates

% With OBS_XYZ we can now utilize the borehole data

%% Semivariogram

u1 = OBS_XYZ(:,1);
u2 = OBS_XYZ(:,2);
z  = OBS_XYZ(:,3);

figure
plot(u1,u2,'o')
axis equal
grid on
xlabel('x-coordinates')
ylabel('y-coordinates')
title('Granada Aarungen data')

Z1=[u1, u2, z]; % input to semivariogram function


maxdist = 17e+03; % 17 km
ant = 64; % number of classes in semivariogram 

[hegam_precipitation] = funk_semivar_mean_var(Z1,Z1,ant,maxdist);

hlag=hegam_precipitation(:,2)

C0 = 10; % 15
C1 = 40-C0; % C1 = 25
a= 3500;% 7500
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
title('Semivariogram for aarungen data')



%% Histogram of Z values

% Assuming OBS_XYZ is a table with columns 'x', 'y', and 'z'
% You can adjust the column names accordingly if they are different

% Load your data or replace this with your actual data

figure
histogram(z);

% Add labels and title
xlabel('Z Values');
ylabel('Frequency');




title('Histogram of Z Values');
