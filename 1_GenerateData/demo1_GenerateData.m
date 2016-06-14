%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Author] Terry Taewoong Um (terry.t.um@gmail.com) %
% Adaptive Systems Lab., University of Waterloo     %
% https://www.facebook.com/terryum.io/              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Please leave the author information when you reuse the codes.

% [x_train, y_train, x_test, y_test] = GenerateData(opt, bShow, nData, sigma, test_ratio)

% opt : {class_gaussian(defalut), class_half_circle, class_full_circle, class_spiral,
%        regress_quad, regress_exp, regress_xsinx, regress_exp_xsinx, 
%        twolink_linear (Terry's Lie group toolbox required, 
%        [Link] https://github.com/terryum/Human-Robot-Motion-Simulator-based-on-Lie-Group)}
% bShow : {0: Don't plot, 1(defalut): plot}
% nData : number of data (defalut: 1000)
% sigma : noise standard deviation (default: 1)
% test_ratio : ratio of test data (default: 0.2)

close all;  clearvars;

% For classification
GenerateData();                 
GenerateData('class_half_circle');
GenerateData('class_full_circle', true, 800);
GenerateData('class_spiral', true, 1200, 1.2, 0.3);

% For regression
GenerateData('regress_quad');
GenerateData('regress_exp');
GenerateData('regress_xsinx', true, 1200, 0.4);
GenerateData('regress_exp_xsinx', true, 1200, 0.3, 0.3);
GenerateData('multi_linear', true, 1200, 0.5);

% % Terry's Lie group toolbox required 
[x_train, y_train, x_test, y_test] = GenerateData('multi_twolink', true, 2000, 0.1);

