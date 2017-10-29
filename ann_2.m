%% Authors: Andrew Petersen & Dr A.
%  Course: CEC 495a
%  Assignment: Assignment 6, artificial neural network (ann)
%  Date Modified: 10/29/17
%%

clear all; close all; clc;

Igray = imread('ann/training.jpg');

BW = ~im2bw(Igray); 

SE = strel('disk',2);
BW2 = imdilate(BW, SE); 

labels = bwlabel(BW2);
Iprops = regionprops(labels);

Iprops( [Iprops.Area] < 1000 ) = [];
num = length( Iprops );

Ibox = floor( [Iprops.BoundingBox] );
Ibox = reshape(Ibox,[4 num]);

for k = 1:num
    col1 = Ibox(1,k);
    col2 = Ibox(1,k) + Ibox(3,k);
    row1 = Ibox(2,k);
    row2 = Ibox(2,k) + Ibox(4,k);
    subImage = BW2(row1:row2, col1:col2);
    
    subImageScaled = imresize(subImage, [24 12]);
    
    TPattern(k,:) = subImageScaled(:)';
end

TTarget = zeros(100,10);

for row = 1:10
    for col = 1:10
        TTarget( 10*(row-1) + col, row ) = 1;
    end
end


TPattern = TPattern';
TTarget = TTarget';

for i = 1:10
mynet = newff([zeros(288,1) ones(288,1)], [24 10], {'logsig' 'logsig'}, 'traingdx');
mynet.trainParam.epochs = 500;
mynet = train(mynet,TPattern,TTarget);
end

correct = [0 0 0 0 0 0];
for j = 1:4
    if j == 1
       Igray = imread('ann/196128.jpg');
       unknown = [1 9 6 1 2 8];
    elseif j == 2
       Igray = imread('ann/480000.jpg');
       unknown = [4 8 0 0 0 0];
    elseif j == 3
        Igray = imread('ann/480096.jpg');
        unknown = [4 8 0 0 9 6];
    elseif j == 4
        Igray = imread('ann/603032.jpg');
        unknown = [6 0 3 0 3 2];
    end 
    UPattern = [];
   BW = ~im2bw(Igray);
   SE = strel('disk',2); 
   BW2 = imdilate(BW, SE); 

   labels = bwlabel(BW2);
   Iprops = regionprops(labels);

   Iprops( [Iprops.Area] < 1000 ) = [];
   num = length( Iprops );

   Ibox = floor( [Iprops.BoundingBox] );
   Ibox = reshape(Ibox,[4 num]);


   for k = 1:num
      col1 = Ibox(1,k);
      col2 = Ibox(1,k) + Ibox(3,k);
      row1 = Ibox(2,k);
      row2 = Ibox(2,k) + Ibox(4,k);
      subImage = BW2(row1:row2, col1:col2);
      subImageScaled = imresize(subImage, [24 12]);
      UPattern(k,:) = subImageScaled(:)';
   end

   UPattern = UPattern';
   Y = sim(mynet,UPattern);

   [w,index] = sort(Y,'descend');
   postcode = index(1,:)-1;

   for k =1:6
      if postcode(k) == unknown(k)
         correct(j) = correct(j)+1;
      end
   end
end

perc = (correct/6)*100;
disp('Unknown Image   Total Runs   Correct Digits   Correct(%)')
fprintf('196128            %d            %d           %4.2f  \n',i,correct(1),perc(1))
fprintf('480000            %d            %d           %4.2f  \n',i,correct(2),perc(2))
fprintf('480096            %d            %d           %4.2f  \n',i,correct(3),perc(3))
fprintf('603032            %d            %d           %4.2f  \n',i,correct(4),perc(4))
