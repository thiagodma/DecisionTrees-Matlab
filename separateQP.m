function [data_qp22, data_qp27, data_qp32, data_qp37] = separateQP(filename)
%This function separates the data into its correspondent QP's
load(filename);

idx= (qp==23 | qp==24 | qp==25);
data_qp22 = data(idx,:);

idx= (qp==28 | qp==29 | qp==30);
data_qp27 = data(idx,:);

idx= (qp==33 | qp==34 | qp==35);
data_qp32 = data(idx,:);

idx= (qp==38 | qp==39 | qp==40);
data_qp37 = data(idx,:);

end