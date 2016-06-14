%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Author] Terry Taewoong Um (terry.t.um@gmail.com) %
% Adaptive Systems Lab., University of Waterloo     %
% https://www.facebook.com/terryum.io/              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Please leave the author information when you reuse the codes.

% [Input]
% opt : {class_gaussian(defalut), class_half_circle, class_full_circle, class_spiral,
%        regress_quad, regress_exp, regress_xsinx, regress_exp_xsinx, 
%        twolink_linear (Terry's Lie group toolbox required, 
%        [Link] https://github.com/terryum/Human-Robot-Motion-Simulator-based-on-Lie-Group)}
% bShow : {0: Don't plot, 1(defalut): plot}
% nData : number of data (defalut: 1000)
% sigma : noise standard deviation (default: 1)
% test_ratio : ratio of test data (default: 0.2)

% [Output]
% training and test data

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
                        opt = 'class_gaussian';
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
    noise = randn(nData,2)*sigma;
    noise2 = randn(nData,2)*sigma;
       
    % 3. Generate the data     
    % opt : {gaussian(defalut), half_circle, full_circle, spiral}
    switch opt
        case 'class_gaussian'
            center_0 = [0 1];
            center_1 = [3 2];
            class_0 = ones(nClass_0,1)*center_0 + noise_0;
            class_1 = ones(nClass_1,1)*center_1 + noise_1;
    
        case 'class_half_circle'
            r_0 = 5;
            r_1 = 10;
            theta_0 = pi*rand(nClass_0, 1);        % theta = [0:pi]
            theta_1 = pi*rand(nClass_1, 1);              
            r_0_noise = r_0*ones(nClass_0, 1) + noise_0(:,1);
            r_1_noise = r_1*ones(nClass_1, 1) + noise_1(:,2);
            class_0 = [r_0_noise.*cos(theta_0), r_0_noise.*sin(theta_0)];
            class_1 = [r_1_noise.*cos(theta_1), r_1_noise.*sin(theta_1)];
            
        case 'class_full_circle'
            r_0 = 5;
            r_1 = 10;
            theta_0 = 2*pi*rand(nClass_0, 1);        % theta = [0:2pi]
            theta_1 = 2*pi*rand(nClass_1, 1);              
            r_0_noise = r_0*ones(nClass_0, 1) + noise_0(:,1);
            r_1_noise = r_1*ones(nClass_1, 1) + noise_1(:,2);
            class_0 = [r_0_noise.*cos(theta_0), r_0_noise.*sin(theta_0)];
            class_1 = [r_1_noise.*cos(theta_1), r_1_noise.*sin(theta_1)];
            
        case 'class_spiral'   
            r_increase = 20;
            r_0 = 5;
            theta_overlab = (1/4)*pi;

            theta_0 = linspace(-theta_overlab, theta_overlab+2*pi, nClass_0);       
            theta_1 = linspace(-theta_overlab-pi, theta_overlab+pi, nClass_1);  
            r_0_spiral = linspace(r_0, r_0+r_increase, nClass_0);
            r_1_spiral = linspace(r_0, r_0+r_increase, nClass_1);

            class_0 = [r_0_spiral.*cos(theta_0); r_0_spiral.*sin(theta_0)]' + noise_0;
            class_1 = [r_1_spiral.*cos(theta_1); r_1_spiral.*sin(theta_1)]' + noise_1;
              
       case 'regress_quad'
            x_all = linspace(-8, 8, nData)';
            y_all = (x_all/2-1).^2 - 5*ones(nData,1) + noise(:,1);      
            
        case 'regress_exp'
            x_all = linspace(-8, 8, nData)';
            y_all = exp(x_all/2-1) - 5*ones(nData,1) + noise(:,1);
            
        case 'regress_xsinx'
            x_all = linspace(-8, 8, nData)';
            y_sin = sin(x_all+pi/4);
            y_lin = 0.5*x_all + 0.2;
            y_all = (y_lin.*y_sin) + noise(:,1);
            
        case 'regress_exp_xsinx'
            x_all = linspace(-8, 8, nData)';
            y_sin = sin(x_all+pi/4);
            y_lin = 0.5*x_all + 0.2;
            y_all = exp(y_lin.*y_sin/2) + noise(:,1);
        
        case 'multi_linear'
            x_train = [linspace(-8, 8, nTrain); linspace(10, -10, nTrain)]' + noise(1:nTrain,:);     
            y_train = [0.3*x_train(:,1)-0.2 -0.5*x_train(:,1)+0.2*x_train(:,2)+3] + noise2(1:nTrain,:);
            
            x_test = [linspace(-8, 8, nTest); linspace(10, -10, nTest)]';
            y_test = [0.3*x_test(:,1)-0.2 -0.5*x_test(:,1)+0.2*x_test(:,2)+3];
            
        case 'multi_twolink'
            nTarget = 3;        % The position of the endpoint of the robot
            robotModel = Model_TwoLink();
            
            x_train = [linspace(0,6*pi,nTrain); linspace(-pi/4,15*pi/4,nTrain);]' + noise(1:nTrain,:);  
            for ii=1:nTrain
                [T_EndPonint T_AllJoint]= FwdKin_Serial(robotModel, x_train(ii,:)); 
                y_train(ii, :) = squeeze(T_AllJoint(1:2,4,nTarget))' + noise2(ii,:);
            end
            x_test = [linspace(0,6*pi,nTest); linspace(-pi/4,15*pi/4,nTest);]';
            for ii=1:nTest
                [T_EndPonint T_AllJoint]= FwdKin_Serial(robotModel, x_test(ii,:)); 
                y_test(ii, :) = squeeze(T_AllJoint(1:2,4,nTarget))';
            end
    end

    % 4. Split training and test sets
    if ~strncmpi(opt, 'multi', 5)     % Training & test data are already splited in 'multi_' cases
        if strncmpi(opt, 'class', 5)
            x_all = [class_0; class_1];
            y_all = [0*ones(nClass_0,1); 1*ones(nClass_1,1)];
        end
        idx_all = randperm(nData); % Randomly select the indices for training data  
        x_train = x_all(idx_all(1:nTrain), :);            y_train = y_all(idx_all(1:nTrain), :);
        x_test = x_all(idx_all(nTrain+1:nData), :);       y_test = y_all(idx_all(nTrain+1:nData), :);       
    end
    
    % 5. Plot the data
    if bShow == true
        figure()
        if strncmpi(opt, 'class', 5)
            fig_train = gscatter(x_train(:,1),x_train(:,2),y_train, 'rb', '..', 10);   
            hold all;
            fig_test = gscatter(x_test(:,1),x_test(:,2),y_test, 'mc', '..', 10);   
            hold off;
            axis('equal');
            legend([fig_train(1), fig_train(2), fig_test(1), fig_test(2)], 'Class 0 (train)', 'Class 1 (train)', 'Class 0 (test)', 'Class 1 (test)')
            xlabel('x');    ylabel('y');
            
        elseif strncmpi(opt, 'regress', 7)
            fig_train = scatter(x_train, y_train, 'b.');            hold all;
            fig_test = scatter(x_test, y_test, 'c.');               hold off; 
            legend([fig_train, fig_test], 'Train data', 'Test data');
            xlabel('x');    ylabel('y');
        
        elseif strncmpi(opt, 'multi_linear', 12)
            fig_y1 = scatter3(x_train(:,1), x_train(:,2), y_train(:,1), 'b.'); hold on;
            fig_y2 = scatter3(x_train(:,1), x_train(:,2), y_train(:,2), 'k.');
            legend([fig_y1, fig_y2], 'y1', 'y2'); 
            xlabel('x1');   ylabel('x2');   zlabel('y1 & y2');

        else
            fig_train = scatter(y_train(:,1), y_train(:,2), 'b.');
            hold all;
            fig_test = scatter(y_test(:,1), y_test(:,2), 'c.');
            hold off; 
            legend([fig_train, fig_test], 'Train data', 'Test data');
            xlabel('y1');    ylabel('y2');
            
            figure()
            t = linspace(0,nTest-1,nTest);
            fig_test_y1 = scatter(t,  y_test(:,1), 'b.');    hold all;
            fig_test_y2 = scatter(t,  y_test(:,2), 'k.');    hold off;
            legend([fig_test_y1, fig_test_y2], 'Test y1', 'Test y2');     
            xlabel('t');    ylabel('y1 & y2');
        end
    end
end
