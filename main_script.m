%main script for inducing CBDT trees and evaluating them

[data_qp22, data_qp27, data_qp32, data_qp37] = separateQP('alldata_depth0.mat');

%creates an instance of class Transcoder
transc_qp22 = Transcoder(data_qp22);
%fits a CBDT to the given data
transc_qp22 = transc_qp22.fitTree();
%evaluates the performance of the classifier
[actCost_qp22, optimCost_qp22, accuracy_qp22] = transc_qp22.predictionsCostAndAccuracy(transc_qp22.tree.Root);

%creates an instance of class Transcoder
transc_qp27 = Transcoder(data_qp27);
%fits a CBDT to the given data
transc_qp27 = transc_qp27.fitTree();
%evaluates the performance of the classifier
[actCost_qp27, optimCost_qp27, accuracy_qp27] = transc_qp27.predictionsCostAndAccuracy(transc_qp27.tree.Root);

%creates an instance of class Transcoder
transc_qp32 = Transcoder(data_qp32);
%fits a CBDT to the given data
transc_qp32 = transc_qp32.fitTree();
%evaluates the performance of the classifier
[actCost_qp32, optimCost_qp32, accuracy_qp32] = transc_qp32.predictionsCostAndAccuracy(transc_qp32.tree.Root);

%creates an instance of class Transcoder
transc_qp37 = Transcoder(data_qp37);
%fits a CBDT to the given data
transc_qp37 = transc_qp37.fitTree();
%evaluates the performance of the classifier
[actCost_qp37, optimCost_qp37, accuracy_qp37] = transc_qp37.predictionsCostAndAccuracy(transc_qp37.tree.Root);


actCost = sum([actCost_qp22, actCost_qp27, actCost_qp32, actCost_qp37]);
optimCost = sum([optimCost_qp22, optimCost_qp27, optimCost_qp32, optimCost_qp37]);
meanAcc = sum([accuracy_qp22, accuracy_qp27, accuracy_qp32, accuracy_qp37])/4;

fprintf('\n')
fprintf('Cost achieved by the CBDT transcoder: %14.6e\n', actCost)
fprintf('Ideal cost: %14.6e\n', optimCost)
fprintf('Mean Accuracy achieved by the CBDT transcoder: %5.2f%% \n', meanAcc*100)