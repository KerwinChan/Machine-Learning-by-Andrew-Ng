function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
% m = size(X,1);
% for i = 1:m
%     x = X(i,:);
%     d = 1000000;
%     for j = 1:K
%        d_temp = sqrt(abs(x(1,1)-centroids(j,1))^2 + abs(x(1,2) - centroids(j,2))^2 );
%        if d_temp<d
%            d = d_temp;
%            idx(i) = j;
%        end
%     end
% end
for i=1:length(idx)
    distanse = pdist2(centroids,X(i,:));   % compute the distance(K,1)   pdist2 is a good function  
       [C,idx(i)]=min(distanse);           % find the minimum
end







% =============================================================

end

