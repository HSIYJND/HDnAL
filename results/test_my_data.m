clc;
clear all;
close all;
addpath(genpath('Assessment\'));
load('abu-airport-2.mat');
O_Img      =   255*(data-min(min(min(data))))/(max(max(max(data)))-min(min(min(data))));
load('abu-airport-2-daae.mat');
N_Img      =   255*(array-min(min(min(array))))/(max(max(max(array)))-min(min(min(array))));
%% DAAE restore results
y_est = N_Img;
PSNR  =  psnr(O_Img/255, y_est/255);
SSIM  =  cal_ssim( O_Img, y_est, 0, 0 ); 
SAM   = SpectAngMapper(O_Img, y_est);
ERGAS = ErrRelGlobAdimSyn(O_Img, y_est);
disp('urban-5')
disp(PSNR),disp(SSIM),disp(SAM),disp(ERGAS)

clear all;
load('abu-airport-3.mat');
O_Img      =   255*(data-min(min(min(data))))/(max(max(max(data)))-min(min(min(data))));
load('abu-airport-3-daae.mat');
N_Img      =   255*(array-min(min(min(array))))/(max(max(max(array)))-min(min(min(array))));
%% DAAE restore results
y_est = N_Img;
PSNR  =  psnr(O_Img/255, y_est/255);
SSIM  =  cal_ssim( O_Img, y_est, 0, 0 ); 
SAM   = SpectAngMapper(O_Img, y_est);
ERGAS = ErrRelGlobAdimSyn(O_Img, y_est);
disp('airport-3')
disp(PSNR),disp(SSIM),disp(SAM),disp(ERGAS)

clear all;
load('abu-urban-5.mat');
O_Img      =   255*(data-min(min(min(data))))/(max(max(max(data)))-min(min(min(data))));
load('abu-urban-5-daae.mat');
N_Img      =   255*(array-min(min(min(array))))/(max(max(max(array)))-min(min(min(array))));
%% DAAE restore results
y_est = N_Img;
PSNR  =  psnr(O_Img/255, y_est/255);
SSIM  =  cal_ssim( O_Img, y_est, 0, 0 ); 
SAM   = SpectAngMapper(O_Img, y_est);
ERGAS = ErrRelGlobAdimSyn(O_Img, y_est);
disp('urban-5')
disp(PSNR),disp(SSIM),disp(SAM),disp(ERGAS)
