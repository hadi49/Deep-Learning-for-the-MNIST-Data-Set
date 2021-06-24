function [Yn, On, Yt, Ot] = ann1923114(N_ep, lr, data, bp,U,V,W,A,B,C,D)
%

%
% MA2647: Artificial Neural Network, project code (MNIST data).
%
% Usage:
%    [Yn On Yt Ot] = ann09demo(N_ep, lr, data, bp, gfx, Nh)
%
% Where, for the MNIST data sets,
% - N_ep the number of epochs, lr is the learning rate
% - data = 1 (use small training set), 2 (use large training set)
% - bp = 1, heuristic unscaled, 2 calculus based bp

% - U,V,W are the number of nodes on the hidden layers

% - Yn are the exact training labels/targets, Ot the training predictions
% - Yt are the exact testing  labels/targets, On the testing  predictions
% The MNIST CSV files are not to be altered in any way, and should be
% in the same folder as this matlab code.
%


% set some useful defaults
set(0,'DefaultLineLineWidth', 2);
set(0,'DefaultLineMarkerSize', 10);

% As a professional touch we should test the validity of our input
if or(N_ep <= 0, lr <= 0)
    error('N_ep and/or lr are not valid')
end
if and(data ~= 1, data ~= 2)
    error('data choice is not valid')
end
if ~ismember(bp,[1,2])
    error('back prop choice is not valid')
end


% obtain the raw training data
if data == 1
    A2 = readtable('mnist_train_100.csv');
    A1 =table2array(A2);
else
    A1 = readmatrix('mnist_train.csv');
end
% convert and NORMALIZE it into training inputs and target outputs
X_train = A1(:,2:end)'/255;  % beware - transpose, data is in columns!
N_train = size(X_train,2); % size(X_train,1/2) gives number of rows/columns
Y_train = zeros(5,N_train);
% set up the one-hot encoding - note that we have to increment by 1
for i=1:N_train
    if A1(i,1)==A
        Y_train(1,i) = 1;
    elseif A1(i,1)==B
        Y_train(2,i) = 1;
    elseif A1(i,1)==C
        Y_train(3,i) = 1;
    elseif A1(i,1)==D
        Y_train(4,i) = 1;
    else
        Y_train(5,i) = 1;
    end
    
end

% default variables
Ni   = 784;             % number of input nodes
No   = 5;              % number of output nodes
% set up weights and biases
W2 = 0.5-rand(784,U); b2 = zeros(U,1);
W3 = 0.5-rand(U, V); b3 = zeros(V,1);
W4 = 0.5-rand(V,W); b4 = zeros(W,1);
W5 = 0.5-rand(W, 5); b5 = zeros(5,1);
% set up a sigmoid activation function for layers 2 and 3
sig2  = @(x) 1./(1+exp(-x));
dsig2 = @(x) exp(-x)./((1+exp(-x)).^2);
% Softmax layer
sig5  = @(x) exp(x)/sum(exp(x));



% we'll calculate the performance index at the end of each epoch
pivec = zeros(1,N_ep);  % row vector

% we now train by looping N_ep times through the training set
for epoch = 1:N_ep
    mixup = randperm(N_train);
    for j = 1:N_train
        i = mixup(j);
        % get X_train(:,i) as an input to the network
        
        a1 = X_train(:,i);
        
        % forward prop to the next layer, activate it, repeat
        n2 = W2'*a1 + b2; a2 = sig2(n2);
        n3 = W3'*a2 + b3; a3 = sig2(n3);
        n4 = W4'*a3 + b4; a4 = sig2(n4);
        n5 = W5'*a4 + b5; a5 = sig5(n5);
        % this is then the output
        y = a5;
        
        % calculate A, the diagonal matrices of activation derivatives
        A2 = diag(dsig2(n2));
        A3 = diag(dsig2(n3));
        A4 = diag(dsig2(n4));
        
        % we calculate the error in this output, and get the S5 vector
        e5 = Y_train(:,i) - y;
        S5 = -e5;
        
        % back prop the error
        % - bp = 1, heuristic unscaled, 2 calculus based bp
        if  bp == 1
            e4 = W5*e5; S4 = -2*A4*e4;
            e3 = W4*e4; S3 = -2*A3*e3;
            e2 = W3*e3; S2 = -2*A2*e2;
        elseif bp == 2
            S4 = A4*W5*S5;
            S3 = A3*W4*S4;
            S2 = A2*W3*S3;
            
            
        else
            error('bp has improper value')
        end
        
        % and use a learning rate to update weights and biases
        W5 = W5 - lr * a4*S5'; b5 = b5 - lr * S5;
        W4 = W4 - lr * a3*S4'; b4 = b4 - lr * S4;
        W3 = W3 - lr * a2*S3'; b3 = b3 - lr * S3;
        W2 = W2 - lr * a1*S2'; b2 = b2 - lr * S2;

    end
    
    for i=1:N_train
        y = sig5(W5'*sig2(W4'*sig2(W3'*sig2(W2'*X_train(:,i)+b2)+b3)+b4) + b5);
        xent = -sum(Y_train(:,i).*log(y));
        pivec(epoch) = pivec(epoch) + xent;
    end
end
% add a subplot; plot the performance index vs (row vector) epochs

subplot(2,2,1);
plot([1:N_ep],pivec,'b');
title('Loss function for Calculus based Backpropagation')
xlabel('epochs'); ylabel('performance index');
% loop through the training set and evaluate accuracy of prediction
wins = 0; wins1=0;wins2=0;wins3=0;wins4=0;wins5= 0;
y_pred = zeros(5,N_train);
for i = 1:N_train
    y_pred(:,i) = sig5(W5'*sig2(W4'*sig2(W3'*sig2(W2'*X_train(:,i)+b2)+b3)+b4) + b5);
    % y_pred(:,i)
    [~, indx1] = max(y_pred(:,i));
    [~, indx2] = max(Y_train(:,i));
    if and(indx1==A, indx2==A); wins1 = wins1+1;
    elseif and(indx1==B, indx2==B); wins2 = wins2+1;
    elseif and(indx1==C, indx2==C); wins3 = wins3+1;
    elseif and(indx1==D, indx2==D); wins4 = wins4+1;
    else and(indx1==5, indx2==5); wins5 = wins5+1;
    end
    barcol = 'r';
    if indx1 == indx2; wins = wins+1; barcol = 'b'; end
    
    %   if gfx ~= 0
    % plot the output 10-vector at top right
    subplot(2,2,2); bar(0:4,y_pred(:,i),barcol);
    title('predicted output (approximate one-hot)')
    % plot the MNIST image at bottom left
    Bi = reshape(1-X_train(:,i),[28,28]);
    subplot(2,2,3);
    imh = imshow(Bi','InitialMagnification','fit');
    subplot(2,2,4);
    b = bar(categorical({'Wins','Losses'}), [wins i-wins]);
    ylim([0,N_train]);
    b.FaceColor = 'flat'; b.CData(1,:) = [1 0 0]; b.CData(2,:) = [0 0 1];
    a = get(gca,'XTickLabel'); set(gca,'XTickLabel',a,'fontsize',18)
    drawnow
    
end
disp(['training set wins: ',num2str(100*wins/N_train),'%'])
disp([num2str(A),'= ',num2str(wins1)])
disp([num2str(B),'= ',num2str(wins2)])
disp([num2str(C),'= ',num2str(wins3)])
disp([num2str(D),'= ',num2str(wins4)])
disp(['None= ',num2str(wins5)])

% assign outputs
Yn=Y_train; On=y_pred;

% obtain the raw test data
if data == 1
    A1 = readmatrix('mnist_test_10.csv');
else
    A1 = readmatrix('mnist_test.csv');
end
% convert and NORMALIZE it into testing inputs and target outputs
X_test = A1(:,2:end)'/255;
% the number of data points
N_test = size(X_test,2);
Y_test = zeros(5,N_test);
% set up the one-hot encoding - recall we have to increment by 1
for i=1:N_test
    if A1(i,1)==A
        Y_test(1,i) = 1;
    elseif A1(i,1)==B
        Y_test(2,i) = 1;
    elseif A1(i,1)==C
        Y_test(3,i) = 1;
    elseif A1(i,1)==D
        Y_test(4,i) = 1;
    else
        Y_test(5,i) = 1;
    end
end

% loop through the test set and evaluate accuracy of prediction
wins = 0;wins1=0;wins2=0;wins3=0;wins4=0;wins5= 0;
y_pred = zeros(5,N_test);
for i = 1:N_test
    y_pred(:,i) = sig5(W5'*sig2(W4'*sig2(W3'*sig2(W2'*X_test(:,i)+b2)+b3)+b4) + b5);
    
    [~, indx1] = max(y_pred(:,i));
    [~, indx2] = max(Y_test(:,i));
    
    
    if and(indx1==A, indx2==A); wins1 = wins1+1;
    elseif and(indx1==B, indx2==B); wins2 = wins2+1;
    elseif and(indx1==C, indx2==C); wins3 = wins3+1;
    elseif and(indx1==D, indx2==D); wins4 = wins4+1;
    elseif and(indx1==5, indx2==5); wins5 = wins5+1;
    end
    barcol = 'r';
    if indx1 == indx2; wins = wins+1; barcol = 'b'; end
    %   barcol = 'r';
    %   if indx1 == indx2; wins = wins+1; barcol = 'b'; end
    
    % plot the output 10-vector at top right
    subplot(2,2,2); bar(0:4,y_pred(:,i),barcol);
    title('predicted output (approximate one-hot)')
    % plot the MNIST image at bottom left
    Bi = reshape(1-X_test(:,i),[28,28]);
    subplot(2,2,3); imh = imshow(Bi','InitialMagnification','fit');
    % animate the wins and losses bottom right
    subplot(2,2,4);
    b = bar(categorical({'Wins','Losses'}), [wins i-wins]);
    ylim([0,N_test]);
    b.FaceColor = 'flat'; b.CData(1,:) = [1 0 0]; b.CData(2,:) = [0 0 1];
    a = get(gca,'XTickLabel'); set(gca,'XTickLabel',a,'fontsize',18)
    drawnow;
    pause(0.01)
    
end
disp(['testing  set wins: ',num2str(100*wins/N_test),'%'])
disp([num2str(A),'= ',num2str(wins1)])
disp([num2str(B),'= ',num2str(wins2)])
disp([num2str(C),'= ',num2str(wins3)])
disp([num2str(D),'= ',num2str(wins4)])
disp(['None= ',num2str(wins5)])
% assign outputs
% X = [name,' will be ',num2str(age),' this year.'];
% disp(X)
Yt=Y_test; Ot=y_pred;
figure
plotconfusion(Yt,Ot);
figure
plotconfusion(Yn,On);
