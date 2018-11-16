function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values

m = size(X, 1);

num_labels = size(Theta2, 1);

%X = [ones(m, 1),X];% 输入特征添加偏执项
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
p_M =   sigmoid ( [ones(m,1),( sigmoid( [ones(m,1),X] * Theta1') ) ] * Theta2' );  %运算时记得加上偏执项，记得过激活函数。


[a,p] = max(p_M,[],2);  %记得返回的是最大值的索引值。








% =========================================================================


end
