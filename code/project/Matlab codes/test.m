close all
clearvars
clc

user = 1;
e = load(['/Users/saeedkazemi/Documents/reproducible-research/data/Step Scan Dataset/Barefoot/Participant',num2str(user),'.mat']);
e = struct2cell(e);
I = e{1,1};

for i =600:2:1000%:1210 
    imshow(I(:,:,i),[]) 
end