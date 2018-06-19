clc; clear; close all;
rng(1); n = 100;
r = sqrt(rand(n,1));
t = 2*pi*rand(n,1);
data1 = [r.*cos(t),r.*sin(t)];

r2 = sqrt(3*rand(n,1)+1);
t2 = 2*pi*rand(n,1);
data2 = [r2.*cos(t2),r2.*sin(t2)];

figure;
plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
hold on
plot(data2(:,1),data2(:,2),'b*','MarkerSize',10)
ezpolar(@(x)1);ezpolar(@(x)2);
axis equal
hold off

% Data generate
data3 = [data1;data2];
theclass = ones(200,1);
theclass(1:n) = -1;

% Train the SVM Classifier
% cl = fitcsvm(data3,theclass,'kernelFunction','RBF','BoxConstraint',Inf,'ClassNames',[-1,1]);
cl = fitcsvm(data3,theclass,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[-1,1]);
% Predict scores over the grid
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(data3(:,1)):d:max(data3(:,1)),min(data3(:,2)):d:max(data3(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(cl,xGrid);

% plot the data and the decision boundary
figure;
h(1:2) = gscatter(data3(:,1),data3(:,2),theclass,'rb','.');
hold on
ezpolar(@(x)1);
h(3) = plot(data3(cl.IsSupportVector,1),data3(cl.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1','Support Vectors'});
axis equal
hold off


r1 = sqrt(rand(2*n,1));                     % Random radii
t1 = [pi/2*rand(n,1); (pi/2*rand(n,1)+pi)]; % Random angles for Q1 and Q3
X1 = [r1.*cos(t1) r1.*sin(t1)];             % Polar-to-Cartesian conversion

r2 = sqrt(rand(2*n,1));
t2 = [pi/2*rand(n,1)+pi/2; (pi/2*rand(n,1)-pi/2)]; % Random angles for Q2 and Q4
X2 = [r2.*cos(t2) r2.*sin(t2)];
X = [X1;X2];  % predictors
Y = ones(4*n,1);
Y(2*n + 1:end) = -1; % Labels

figure;
gscatter(X(:,1),X(:,2),Y);
title('Scatter Diagram of Simulated Data')

SVMModel1 = fitcsvm(X,Y,'KernelFunction','mysigmoid','Standardize',true);

% % Compute the scores over a grid
% d = 0.02; % step size of the grid
% [x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),min(X(:,2):d:max(X(:,2))));
% xGrid = [x1Grid(:),x2Grid(:)];
% [~,scores1] = predict(SVMModel1,xGrid);

% figure;
% h(1:2) = gscatter(X(:,1),X(:,2),Y);
% hold on
% h(3) = plot(X(SVMModel1.IsSupportVector,1),X(SVMModel1.IsSupportVector,2),'ko','MarkerSize',10);
%   % Support vectors
% contour(x1Grid,x2Grid,reshape(scores1(:,2),size(x1Grid)),[0 0],'k');
%   % Decision boundary
% title('Scater Diagram with the Decision boundary')
% legend({'-1','1','Support Vectors'},'Location','Best');
% hold off

%Compute the scores over a grid
d = 0.02; % Step size of the grid
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];        % The grid
[~,scores1] = predict(SVMModel1,xGrid); % The scores

figure;
h(1:2) = gscatter(X(:,1),X(:,2),Y);
hold on
h(3) = plot(X(SVMModel1.IsSupportVector,1),X(SVMModel1.IsSupportVector,2),'ko','MarkerSize',10);
    % Support vectors
contour(x1Grid,x2Grid,reshape(scores1(:,2),size(x1Grid)),[0 0],'k');
    % Decision boundary
title('Scatter Diagram with the Decision Boundary')
legend({'-1','1','Support Vectors'},'Location','Best');
hold off

CVSVMModel1 = crossval(SVMModel1);
misclass1 = kfoldLoss(CVSVMModel1);
misclass1;