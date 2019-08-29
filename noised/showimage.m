clc
clear
fileFolder=fullfile('C:\Users\JiangKai\Desktop\daae-TENSORFLOW\daae-TENSORFLOW\noised');
dirOutput=dir(fullfile(fileFolder,'*.mat'));
fileNames={dirOutput.name}';
for i=1:3
    load(char(fileNames(i)));
    imwrite(mat2gray(Im(:,:,60)),[char(fileNames(i)),'.jpg'])
end