classdef Transcoder
    properties
        data
        tree = struct('Root',[],'LeftGroup',[],'RightGroup',[],'Feature',[],'Value',[],'isLeaf',[],'Class',[]);
        
    end
    
    methods
        
        function obj = Transcoder(data)
            %Constructor for the class "Tree". Basically it loads the data.
            obj.data = data;
        end
        
        function y = positiveMin(obj, A)
            %This method receives a vector A and outputs the positive
            %minimal value in A
            A = A(A>=0);
            if(isempty(A))
                y = 1;
            else
                y = min(A);
            end
            
            if(isinf(y))
                y = 1;
            end
        end
        
        function data = correctsMisclassifedSamples(obj, data)
            %Some samples from the dataset are misclassified. This method
            %solves this issue.
            for i=1:size(data,1)
                if( data(i,12) == obj.positiveMin(data(i,12:22)) )
                    data(i,11) = 0;
                else
                    data(i,11) = 1;
                end
            end
        end
        
        function data = simplifyCosts(obj, data)
            %This function gets the minimal cost1 and ignores the rest of
            
            n = size(data,1);
            
            for i=1:n
                data(i,13) = obj.positiveMin(data(i,13:22));
            end
            
            data(:,14:22) = [];
             
        end
        
        function y = findOne(obj, x)
            %This method finds the position of all number "1" in vector x
            
            a=0;
            c=0;
            for i=1:size(x)
                if (x(i)==1)
                    c(i-a)=i;
                else
                    a=a+1;
                end
            end
            y=c';
        end
        
        function stop = stopCond(obj, data, N, flag)
            %This method decides if the algorithm must stop tree induction
            %at a given node
            
            if(flag==1)
                stop=1;
                return;
            end
            
            stop = 0;
            
            if (isempty(data))
                stop = 0;
                return;
            end
            
            %--------------------------------------------------------------------------
            %this block of code doesnt let the algorithm fall in infinite
            %recursion beacause of repeated feature vectors
            a = zeros(size(data,1),10);
            b = ones(size(data,1),1)*9;
            
            for i=1:size(data,1)
                a(i,:) = [0 0 0 0 0 0 1 0 0 0];
            end
            
            x = sum(data(:,1:10) == a,2);
            y = (x == b);
            
            out_aux = obj.findOne(y);
            ps = length(out_aux)/size(data,1);
            
            if(ps>=0.8)
                stop = 1;
                return;
            end
            %--------------------------------------------------------------------------
            
            if (length(data(:,11)) <= 0.05*N)
                stop = 1;
            end
        end
        
        function [cost0,cost1] = cost0Xcost1(obj,data)
            %This method calculates the cost of classifying all the samples in a
            %node as class 0 and as class 1
            
            cost0 = sum(data(:,12));
            cost1 = sum(data(:,13));
        end
        
        function class = Classify(obj,data)
            %This method classifies a leaf node trying to minimize the cost
            [cost0,cost1] = obj.cost0Xcost1(data);
            if (cost0 <= cost1)
                class = 0;
            else
                class = 1;
            end
        end
        
        function out = reduceOptions_v1(obj, n, b)
            %This method makes a uniform sampling of size "b" in vector "n"
            a = length(n);
            out = n(1:fix(a/b):a-1);
        end
        
        function out = reduceOptions_v2(obj, n, b)
            %This expands the values in n
            
            out = [n ;linspace(min(n),max(n),b)'];
            out = unique(sort(out));
        end
        
        function out = gini(obj, X)
            %This method calculates the gini diversity index of a vector of
            %classes
            
            if(isempty(X))
                out = 1e100;
                return;
            end
            
            idx0 = X(:,1)==0;
            idx1 = X(:,1)==1;
            
            count0 = sum(idx0);
            count1 = sum(idx1);
            
            p0= count0/(count0+count1);
            p1 = 1-p0;
            
            out = (1 - p0^2 - p1^2);
        end
        
        function [dataright, dataleft] = divideData(obj, data, feature, value)
            %This method divides the data in two subgroups: "dataright"
            %which satisfies the condition that feature "i" is >= than
            %value "v", and "dataleft" which is the complementar.
            
            dataright = data(data(:,feature) >= value , :);
            dataleft = data(data(:,feature) < value , :);
        end
        
        function [LeftGroup, RightGroup, feat, val, flag] = splitData(obj, data, Nt)
            %This method splits the tree in order to maximize the
            %information gain.
            
            n = cell(1,10);
            
            b =50; %number of values that will be used
            for i=1:6
                n{i} = unique(sort(data(:,i)));
                if (size(n{i},1)>=b)
                    n{i} = obj.reduceOptions_v1(n{i},b);
                end
            end
            
            b1 = 20; %number of features that will be used
            for i=7:10
                n{i} = unique(sort(data(:,i)));
                n{i} = obj.reduceOptions_v2(n{i},b1);
            end
            
            N = size(data,1);
            s = max(cellfun('length',n));
            val = -1*ones(10,s);
            WAIM = 1e100*ones(10,s);
            
            for i=1:10
                x = n{i};
                f = data(:,[i 11]);
                for j=1:length(n{i})
                    cRight = f(f(:,1) >= x(j), 2);
                    cLeft = f(f(:,1) < x(j), 2);
                    WAIM(i,j) = (sum(cRight~=-1)*obj.gini(cRight)/N) + (sum(cLeft~=-1)*obj.gini(cLeft)/N);
                    val(i,j) = x(j);
                end
            end
            
            minimum = min(min(WAIM));
            [x,y]=find(WAIM==minimum);
            feat = x(1);
            val = val(x(1),y(1));
            
            [RightGroup, LeftGroup] = obj.divideData(data,feat,val);
            
            sR = size(RightGroup,1);
            sL = size(LeftGroup,1);
            
            if(sR/Nt <= 0.01 || sL/Nt <= 0.01)
                flag=1;
            else
                flag=0;
            end
            
        end
        
        function outTree = simplifyTree(obj, tree_aux)
            %This function deletes unnecessary nodes
            
            cond = ( (~isempty(tree_aux.RightGroup)) && (~isempty(tree_aux.LeftGroup)) && ...
            (tree_aux.RightGroup.isLeaf == 1) && (tree_aux.LeftGroup.isLeaf == 1) && ...
            (tree_aux.RightGroup.Class == tree_aux.LeftGroup.Class));
            
            if(tree_aux.isLeaf == 0 && cond)
                tree_aux.Class = tree_aux.RightGroup.Class;
                tree_aux.LeftGroup = [];
                tree_aux.RightGroup = [];
                tree_aux.Feature = [];
                tree_aux.Value = [];
                tree_aux.isLeaf = 1;
            else
                if(~isempty(tree_aux.RightGroup))
                    if(tree_aux.RightGroup.isLeaf ~= 1)
                        tree_aux.RightGroup = obj.simplifyTree(tree_aux.RightGroup);
                    end
                end
                
                cond = ( (~isempty(tree_aux.RightGroup)) && (~isempty(tree_aux.LeftGroup)) && ...
                (tree_aux.RightGroup.isLeaf == 1) && (tree_aux.LeftGroup.isLeaf == 1) && ...
                (tree_aux.RightGroup.Class == tree_aux.LeftGroup.Class));
            
                if(tree_aux.isLeaf == 0 && cond)
                    tree_aux.Class = tree_aux.RightGroup.Class;
                    tree_aux.LeftGroup = [];
                    tree_aux.RightGroup = [];
                    tree_aux.Feature = [];
                    tree_aux.Value = [];
                    tree_aux.isLeaf = 1;
                end
                if(~isempty(tree_aux.LeftGroup))
                    if(tree_aux.LeftGroup.isLeaf ~= 1)
                        tree_aux.LeftGroup = obj.simplifyTree(tree_aux.LeftGroup);
                    end
                end
                cond = ( (~isempty(tree_aux.RightGroup)) && (~isempty(tree_aux.LeftGroup)) && ...
                (tree_aux.RightGroup.isLeaf == 1) && (tree_aux.LeftGroup.isLeaf == 1) && ...
                (tree_aux.RightGroup.Class == tree_aux.LeftGroup.Class));
                
                if(tree_aux.isLeaf == 0 && cond)
                    tree_aux.Class = tree_aux.RightGroup.Class;
                    tree_aux.LeftGroup = [];
                    tree_aux.RightGroup = [];
                    tree_aux.Feature = [];
                    tree_aux.Value = [];
                    tree_aux.isLeaf = 1;
                end
            end
            outTree = tree_aux;
        end
        
        function tree_aux = treeInduction(obj, root, N, a, flag)
            %This method creates a new node in the tree
            
            stop = obj.stopCond(root.Root,N,flag);
            
            if (stop == 1)
                c = obj.Classify(root.Root);
                root_out = struct('Root',[root.Root],'LeftGroup',[],'RightGroup',[],'Feature',[],'Value',[],'isLeaf',[1],'Class',[c]);
            else
                
                sLeft = struct('Root',[],'LeftGroup',[],'RightGroup',[],'Feature',[],'Value',[],'isLeaf',[],'Class',[]);
                sRight = sLeft;
                
                [sLeft.Root,sRight.Root,feat,val,flag] = obj.splitData(root.Root,N);
                
                sLeft = obj.treeInduction(sLeft,N,a,flag);
                sRight = obj.treeInduction(sRight,N,a,flag);
                
                root_out = root;
                root_out.LeftGroup = sLeft;
                root_out.RightGroup = sRight;
                root_out.Feature = feat;
                root_out.Value = val;
                root_out.isLeaf = 0;
                
            end
            
            tree_aux = root_out;
        end
        
        function obj = fitTree(obj)
            %This method fits a CBDT ("Cost Based Decision Tree")
            tic
            
            fprintf('Training started\n')
            
            obj.data = obj.correctsMisclassifedSamples(obj.data);
            
            obj.data = obj.simplifyCosts(obj.data);
            
            N = size(obj.data,1);
            obj.tree.Root = obj.data;
            
            obj.data = [];
            
            obj.tree = obj.treeInduction(obj.tree, N, 11, 0);

            obj.tree = obj.simplifyTree(obj.tree);
            
            toc
            fprintf('Finished training\n')
        end
        
        function c = predictOneSample(obj,tree_aux, sample)
            %Makes a prediction on a single sample
            if(tree_aux.isLeaf == 0)
                
                feat = tree_aux.Feature;
                val = tree_aux.Value;
                
                if(sample(1,feat)>= val)
                    c = obj.predictOneSample(tree_aux.RightGroup,sample);
                else
                    c = obj.predictOneSample(tree_aux.LeftGroup,sample);
                end
            else
                c = tree_aux.Class;
            end  
        end
        
        function preds = predict(obj, X)
           %Makes prediction on a set of samples. X is a matrix which each
           %line is a different feature vector.
           
           preds = zeros(size(X,1),1);
           
           for i=1:size(X,1)
               preds(i) = obj.predictOneSample(obj.tree, X(i,:));
           end
        end
        
        function [actCost,optimCost, accuracy] = predictionsCostAndAccuracy(obj, X)
            % Calculates the cost of the decisions and the accuracy. X is a
            % matrix which each line is a different feature vector and the
            % two last columns are related to the cost0 and cost1. actCost
            % is the cost achieved by the classifier and optimCost is the
            % cost if the classifier had 100% accuracy.
            
            preds = obj.predict(X);
            
            idx = X(:,11) == preds; %checks wich samples the classifier classified correctly
            accuracy = sum(idx)/length(preds);
            
            actCost = ones(length(preds),1); %pre allocates memmory
            optimCost = actCost;
            
            for i=1:length(preds)
                if(preds(i) == 0)
                    actCost(i) = X(i,12);
                else
                    actCost(i) = obj.positiveMin(X(i,13:end));
                end
                
                if(X(i,11) == 0)
                    optimCost(i) = X(i,12);
                else
                    optimCost(i) = obj.positiveMin(X(i,13:end));
                end
            end
            
            actCost = sum(actCost);
            optimCost = sum(optimCost);
        end
            
        
    end
    
end