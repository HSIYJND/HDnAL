clc
clear
fileFolder=fullfile('C:\Users\JiangKai\Desktop\daae-TENSORFLOW\daae-TENSORFLOW\raw');
dirOutput=dir(fullfile(fileFolder,'*.mat'));
fileNames={dirOutput.name}';
for i=1:3
    load(char(fileNames(i)));
    imwrite(mat2gray(data(:,:,60)),[char(fileNames(i)),'.jpg'])
end