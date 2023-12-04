%calculate Euclidian distance from grid coordinates (matrix) to observations
%input: 
%m1 is x- and y-coordinates (nx2 matirx)
%m2 is x- and y-coordinates (mx2 matirx)
%
%m1 and m2 may be identical (e.g. for semivariograms)
%or different from each other (e.g. for cross-semivariograms
%
%output:
%h is (n x m) matrix of distances between coordinates in m1 and m2
%h 


function[h] = find_norm(m1,m2);



%euclidian distances for observations
%make matrix of differences


[N,j]=size(m1);
[M,j]=size(m2);

if N == M    %save quite a few iterations

  for i=1:N
    for j=i:M
      h(i,j) = norm( m1(i,:) - m2(j,:) );
    end
  end
  
  h = h + h';


else    %if matrix is not symetric

  for i=1:N
    for j=1:M
      h(i,j) = norm( m1(i,:) - m2(j,:) );
    end
  end
  
end




