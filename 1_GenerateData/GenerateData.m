%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Author] Terry Taewoong Um (terry.t.um@gmail.com) %
% Adaptive Systems Lab., University of Waterloo     %
% https://www.facebook.com/terryum.io/              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Please leave the author information when you reuse the codes.

% Input
% opt : {gaussian(defalut), half_circle, full_circle, spiral}
% bShow : {0: Don't plot, 1(defalut): plot}
% nData : number of data (defalut: 1000)
% sigma : noise standard deviation (default: 1)
% test_ratio : ratio of test data (default: 0.2)

function [x_train, y_train, x_test, y_test] = GenerateData(opt, bShow, nData, sigma, test_ratio)

    % 1. Assign default parameter values 
    if nargin < 5
        test_ratio = 0.2;
        if nargin < 4
            sigma = 1;
            if nargin < 3
                nData = 1000;
                if nargin < 2
                    bShow = true; 
                    if nargin < 1
                        opt = 'gaussian';
                    end
                end
            end
        end
    end
    
    % 2. Set the number of data     
    nTest = floor(nData*test_ratio);
    nTrain = nData - nTest;
    nClass_0 = nData/2;
    nClass_1 = nData-nData/2;
    
    noise_0 = randn(nClass_0,2)*sigma;
    noise_1 = randn(nClass_1,2)*sigma;
       
    % 3. Generate the data     
    % opt : {gaussian(defalut), half_circle, full_circle, spiral}
    switch opt
        case 'gaussian'
            center_0 = [0 1];
            center_1 = [3 2];
            class_0 = ones(nClass_0,1)*center_0 + noise_0;
            class_1 = ones(nClass_1,1)*center_1 + noise_1;
    
        case 'half_circle'
            r_0 = 5;
            r_1 = 10;
            theta_0 = pi*rand(nClass_0, 1);        % theta = [0:pi]
            theta_1 = pi*rand(nClass_1, 1);              
            r_0_noise = r_0*ones(nClass_0, 1) + noise_0(:,1);
            r_1_noise = r_1*ones(nClass_1, 1) + noise_1(:,2);
            class_0 = [r_0_noise.*cos(theta_0), r_0_noise.*sin(theta_0)];
            class_1 = [r_1_noise.*cos(theta_1), r_1_noise.*sin(theta_1)];
            
        case 'full_circle'
            r_0 = 5;
            r_1 = 10;
            theta_0 = 2*pi*rand(nClass_0, 1);        % theta = [0:2pi]
            theta_1 = 2*pi*rand(nClass_1, 1);              
            r_0_noise = r_0*ones(nClass_0, 1) + noise_0(:,1);
            r_1_noise = r_1*ones(nClass_1, 1) + noise_1(:,2);
            class_0 = [r_0_noise.*cos(theta_0), r_0_noise.*sin(theta_0)];
            class_1 = [r_1_noise.*cos(theta_1), r_1_noise.*sin(theta_1)];
            
        case 'spiral'   
            r_increase = 20;
            r_0 = 5;
            theta_overlab = (1/4)*pi;

            theta_0 = linspace(-theta_overlab, theta_overlab+2*pi, nClass_0);       
            theta_1 = linspace(-theta_overlab-pi, theta_overlab+pi, nClass_1);  
            r_0_spiral = linspace(r_0, r_0+r_increase, nClass_0);
            r_1_spiral = linspace(r_0, r_0+r_increase, nClass_1);

            class_0 = [r_0_spiral.*cos(theta_0); r_0_spiral.*sin(theta_0)]' + noise_0;
            class_1 = [r_1_spiral.*cos(theta_1); r_1_spiral.*sin(theta_1)]' + noise_1;
    end

    % 4. Divide training and test sets
    x_all = [class_0; class_1];
    y_all = [0*ones(nClass_0,1); 1*ones(nClass_1,1)];
    idx_all = randperm(nData); % Randomly select the indices for training data
   
    x_train = x_all(idx_all(1:nTrain), :);            y_train = y_all(idx_all(1:nTrain), :);
    x_test = x_all(idx_all(nTrain+1:nData), :);       y_test = y_all(idx_all(nTrain+1:nData), :);
    
    % 5. Plot the data
    if bShow == true
        figure()
        fig_train = gscatter(x_train(:,1),x_train(:,2),y_train, 'rb', '..', 10);   
        hold all;
        fig_test = gscatter(x_test(:,1),x_test(:,2),y_test, 'mc', '..', 10);   
        hold off;
        axis('equal');
        legend([fig_train(1), fig_train(2), fig_test(1), fig_test(2)], 'Class 0 (train)', 'Class 1 (train)', 'Class 0 (test)', 'Class 1 (test)')
    end

end
