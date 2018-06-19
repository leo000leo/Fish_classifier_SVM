%% pre-processing
close all;
clear all;
clc;

trainingSet  = imageSet(fullfile('Fish trainingSet'),'recursive');
testingSet   = imageSet(fullfile('Fish testingSet'),'recursive');

trainingFeatures = [];
trainingLabels   = [];
testingFeatures  = [];
testingLabels    = [];
FeatureSize      = 17;
tic;
%%  Extract trainingSet Features
for num  = 1 : numel(trainingSet)
   
    numImages = trainingSet(num).Count;
    features  = zeros(numImages, FeatureSize, 'single');
    
    for n = 1:numImages
    img  = read(trainingSet(num),n);
%     figure;
%     subplot(231);  imshow(img);
    
    %  normalize
    img  = imresize(img, [256  512]);
%     subplot(232);  imshow(img);
    rect = [60 48 280 106];
    img  = imcrop(img,rect);
%     subplot(233);  imshow(img);
  toc;
    %% Extract featrue 
    grayimg = rgb2gray(img);
%     subplot(234);  imshow(grayimg); 
    img2 = rgb2hsv(img);  
%     subplot(235);  imshow(img2); 
    hsvimg = rgb2gray(img2);
%     subplot(236);  imshow(hsvimg);
    
    % Different view
    
   %灰度共生矩阵纹理统计
    glcms  = graycomatrix(hsvimg,'numlevels',64,'offset',[0 1;-1 1;-1 0;-1 -1]);
    stats  = graycoprops(glcms,{'contrast','correlation','energy','homogeneity'});
    ga1    = glcms(:,:,1);
    ga2    = glcms(:,:,2);
    ga3    = glcms(:,:,3);
    ga4    = glcms(:,:,4);
    energya1 = 0; energya2 = 0; energya3 = 0; energya4 = 0;
    for i = 1:64
        for j = 1:64
            energya1 = energya1+sum(ga1(i,j)^2);
            energya2 = energya2+sum(ga2(i,j)^2);
            energya3 = energya3+sum(ga3(i,j)^2);
            energya4 = energya4+sum(ga4(i,j)^2);
            j = j+1;
        end
        i = i+1;
    end
    
    s1 = 0; s2 = 0; s3 = 0; s4 = 0; s5 = 0;
    %对比度
    for m = 1:4
        s1 = stats.Contrast(1,m)+ s1;
        m = m+1;
    end
    %相关性
      for m = 1:4
        s2 = stats.Correlation(1,m)+ s2;
        m = m+1;
      end
    %熵
      for m = 1:4
        s3 = stats.Energy(1,m)+ s3;
        m = m+1;
      end
    %平稳度
       for m = 1:4
        s4 = stats.Homogeneity(1,m)+ s4;
        m = m+1;
       end
     %二阶距(能量)   
       s5 =0.000001*(energya1+energya2+energya3+energya4);  
    t0 =[s1,s2,s3,s4,s5];
    
    %% 颜色特征提取
    r  = img(:,:,1);
    g  = img(:,:,2);
    b  = img(:,:,3);
    h  = img2(:,:,1);
    s  = img2(:,:,2);
    v  = img2(:,:,3);

    r  = double(r); g = double(g); b = double(b);
    h  = double(h); s = double(s); v = double(v);
    
    %一阶矩
    ravg = mean2(r);
    gavg = mean2(g);
    bavg = mean2(b);
    
    havg = mean2(h);
    savg = mean2(s);
    vavg = mean2(v);    
    %二阶距
    rstd = std(std(r));
    gstd = std(std(g));
    bstd = std(std(b));
    
    hstd = std(std(h));
    sstd = std(std(s));
    vstd = std(std(v));

    features(n, :) = [t0,ravg,gavg,bavg,rstd,gstd,bstd,havg,savg,vavg,hstd,sstd,vstd];
    end  
    
    labels = repmat(trainingSet(num).Description, numImages, 1);

    trainingFeatures = [trainingFeatures; features];   %#ok<AGROW>
    trainingLabels   = [trainingLabels;   labels  ];   %#ok<AGROW>
    
end
% %归一化
% trainingFeatures = trainingFeatures.';
% [trainingFeatures,PS] = mapminmax(trainingFeatures); 
% trainingFeatures = trainingFeatures.';


  %% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
 classifier = fitcecoc(trainingFeatures, trainingLabels);

 %% Extract features from the test set. The procedure is similar to what
 % was shown earlier and is encapsulated as a helper function for brevity.
for num  = 1 : numel(testingSet)
   
    numImages = testingSet(num).Count;
    features  = zeros(numImages, FeatureSize, 'single');
    
    for n = 1:numImages
    img  = read(testingSet(num),n);
%     figure;
%     subplot(231);  imshow(img);
    
    %  normalize
    img  = imresize(img, [256  512]);
%     subplot(232);  imshow(img);
    rect = [64 58 286 110];
    img  = imcrop(img,rect);
%     subplot(233);  imshow(img);
  
    %% Extract featrue 
    grayimg = rgb2gray(img);
%     subplot(234);  imshow(grayimg); 
    img2 = rgb2hsv(img);  
%     subplot(235);  imshow(img2); 
    hsvimg = rgb2gray(img2);
%     subplot(236);  imshow(hsvimg);
    
    % Different view
    
   %灰度共生矩阵纹理统计
    glcms  = graycomatrix(hsvimg,'numlevels',64,'offset',[0 1;-1 1;-1 0;-1 -1]);
    stats  = graycoprops(glcms,{'contrast','correlation','energy','homogeneity'});
    ga1    = glcms(:,:,1);
    ga2    = glcms(:,:,2);
    ga3    = glcms(:,:,3);
    ga4    = glcms(:,:,4);
    energya1 = 0; energya2 = 0; energya3 = 0; energya4 = 0;
    for i = 1:64
        for j = 1:64
            energya1 = energya1+sum(ga1(i,j)^2);
            energya2 = energya2+sum(ga2(i,j)^2);
            energya3 = energya3+sum(ga3(i,j)^2);
            energya4 = energya4+sum(ga4(i,j)^2);
            j = j+1;
        end
        i = i+1;
    end
    
    s1 = 0; s2 = 0; s3 = 0; s4 = 0; s5 = 0;
    %对比度
    for m = 1:4
        s1 = stats.Contrast(1,m)+ s1;
        m = m+1;
    end
    %相关性
      for m = 1:4
        s2 = stats.Correlation(1,m)+ s2;
        m = m+1;
      end
    %熵
      for m = 1:4
        s3 = stats.Energy(1,m)+ s3;
        m = m+1;
      end
    %平稳度
       for m = 1:4
        s4 = stats.Homogeneity(1,m)+ s4;
        m = m+1;
       end
     %二阶距(能量)   
       s5 =0.000001*(energya1+energya2+energya3+energya4);  
    t0 =[s1,s2,s3,s4,s5];
    
    %% 颜色特征提取
    r  = img(:,:,1);
    g  = img(:,:,2);
    b  = img(:,:,3);
    h  = img2(:,:,1);
    s  = img2(:,:,2);
    v  = img2(:,:,3);

    r  = double(r); g = double(g); b = double(b);
    h  = double(h); s = double(s); v = double(v);
    
    %一阶矩
    ravg = mean2(r);
    gavg = mean2(g);
    bavg = mean2(b);
    
    havg = mean2(h);
    savg = mean2(s);
    vavg = mean2(v);    
    %二阶距
    rstd = std(std(r));
    gstd = std(std(g));
    bstd = std(std(b));
    
    hstd = std(std(h));
    sstd = std(std(s));
    vstd = std(std(v));

    features(n, :) = [t0,ravg,gavg,bavg,rstd,gstd,bstd,havg,savg,vavg,hstd,sstd,vstd];
    end  
    
    labels = repmat(testingSet(num).Description, numImages, 1);

    testingFeatures = [testingFeatures; features];   %#ok<AGROW>
    testingLabels   = [testingLabels;   labels  ];   %#ok<AGROW>
    
end

% testingFeatures = testingFeatures.';
% [testingFeatures,PS] = mapminmax(testingFeatures);
% testingFeatures = testingFeatures.';
%%
% Make class predictions using the test features.
tic;
predictedLabels = predict(classifier, testingFeatures);

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testingLabels, predictedLabels);

DisplayMatrix(confMat)
 toc;


