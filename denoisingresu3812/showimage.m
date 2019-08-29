clc
clear
fileFolder=fullfile('E:\daae-TENSORFLOW20190403\daae-TENSORFLOW20190403\daae-TENSORFLOW\daae-TENSORFLOW\denoisingresu3812');
dirOutput=dir(fullfile(fileFolder,'*.mat'));
fileNames={dirOutput.name}';
for i=1:9
    load(char(fileNames(i)));
    imwrite(mat2gray(array(:,:,50)),[char(fileNames(i)),'.jpg'])
end