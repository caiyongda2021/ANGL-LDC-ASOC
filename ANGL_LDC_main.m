%code for paper 'Adaptive Neighbors Graph Learning for Large-Scale Data
%Clustering using Vector Quantization and Self-Regularization-ASOC'
%author: Yongda Cai, e-mail: caiyongda2021@email.szu.edu.cn

close all
clear;clc
%environment setting
addpath 'datasets'
addpath 'functions'
name = 'covtype_uni.mat';
starttime = tic;
result_iter = 1;

while(result_iter<21)
    load(name);
    k_max = 10;
    NITER = 30;
    c = length(unique(Y));
    num_VQ = c*20;

    X = full(double(X));
    X_old = X;
    [num_old,dim_old] = size(X_old);
    X_old = X_old-repmat(mean(X_old),[num_old,1]);

    %%medium-scale data set
    %[y, X_VQ, ~, sumD, ~] = litekmeans(X_old, num_VQ);
    
    %%large-scale data set
    sample_anchor = randperm(num_old,num_VQ*200);
    X_anchor = X_old(sample_anchor,:);
    [~, X_VQ, ~, sumD, ~] = litekmeans(X_anchor, num_VQ);
    dist_XVQ = L2_distance_1(X_old',X_VQ');
    [~,y] = find(dist_XVQ==(min(dist_XVQ'))');
    
    w = hist(y, num_VQ);
    S = zeros(num_VQ);
    distX_VQ = L2_distance_1(X_VQ',X_VQ');
    [distX_VQ_sort, idx_sort] = sort(distX_VQ,2);
    lambda = mean(distX_VQ_sort(:));
    lambda1 = lambda;
    for iter1 = 1:num_VQ
        S(iter1,idx_sort(iter1,2:k_max+1)) = 1 - ((k_max-1)*distX_VQ_sort(iter1,2:k_max+1))./repmat(sum(distX_VQ_sort(iter1,2:k_max+1),2),1,k_max);
    end
    S(find(S<0))=0;
    A = (S+S')/2;
    A = 1./(1-A)-1;
    %A0 = w'.*A;
    A0 = A;
    D0 = diag(sum(A0));
    L0 = D0 - A0;
    [H, temp, evs]=eig1(L0, c, 0);
    if sum(evs(1:c+1)) < 0.00000000001
        error('The original graph has more than %d connected component', c);
    end

    for iter = 1:NITER
        dist_h = L2_distance_1(H',H');
        dist_xh = sqrt(distX_VQ+lambda*dist_h);

        S1 = zeros(num_VQ);
        for i = 1:num_VQ
            idxa0 = idx_sort(i,2:k_max+1);
            S1(i,idxa0) = 1 - ((k_max-1)*dist_xh(i,idxa0))./repmat(sum(dist_xh(i,idxa0),2),1,k_max);
        end
        S1(find(S1<0))=0;
        A = (S1+S1')/2;
        A = 1./(1-A)-1;
        A1 = w'.*A;
        A1(find(A1==Inf))=0;
        D = diag(sum(A1));
        L1 = D-A1;

        H_old = H;
        [H1,ev, ~] = svd(L1);
        H = H1(:,num_VQ-c+1:num_VQ);
        evs = diag(ev);
        fn1 = sum(evs(num_VQ-c+1:num_VQ));
        fn2 = sum(evs(num_VQ-c:num_VQ));
        if fn1 > 0.00000000001
            lambda = 2*lambda;
        elseif fn2 < 0.00000000001
            lambda = lambda/2;  H = H_old;
        else
            break;
        end 

    end

    [clusternum, y_VQ]=graphconncomp(sparse(A)); y_VQ = y_VQ';
    
    if(max(y_VQ==c)) %if the VQ data do not has c clusters,regenerate the VQ data and retrain the model
        Mdl = fitcknn(X_VQ,y_VQ,'NumNeighbors',1); 
        Y_clster = predict(Mdl,X_old); 
        result_ANGL_LDC(result_iter,:) = ClusteringMeasure_new(Y, Y_clster)
        result_iter = result_iter+1;
    end
end
time = toc(starttime);
mean(result_ANGL_LDC)
std(result_ANGL_LDC)
time/20

% feature visualization
% cm = colormap(jet(c));
% rl = randperm(c);
% for i=1:c
%     plot(X(Y==rl(i),1),X(Y==rl(i),2),'.', 'color', cm(i,:)); hold on;
% end


