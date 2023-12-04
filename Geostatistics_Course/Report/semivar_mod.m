% returns covariance function from semivariogram parameters
% h       - |distance|, i.e. no negative numbers! (vector or matrix)
% C0      - nugget or white noise (scalar)
% C1 + C0 - sill or variance (scalar)
% hR       - range or distance where practical no covariance
% pr      - limit for covariancs at practical range, 
%           eg. for 0.05 covariance pr = log(20) or almost 3 
% eg      - eg=1 exponential model, eg=2 gaussian model
%
%usage:
%
%ch=semivar_mod(h,C0,C1,hR,pr,eg)


function[ch]=semivar_mod(h,C0,C1,hR,pr,eg)

  gh = C0 + C1*(1-exp(-pr.*(h/hR).^eg));
  ch = C0 + C1 - gh;

  [row,col]=size(ch);

  
  
% add nugget if h == 0
% to include effect of measurements uncertanty

% add nugget for covariance matrix (symmetric case)
if row == col
  for i = 1:col
    ch(i,i) = ch(i,i) + C0;
  end

  else
  
  % add nugget for cross-covariance (not symmetric)
 
  [vkol,mrad] = size(h); % 
  
  if vkol > 1
  [p,q]=find(h==0);
  [ok,stupid]=size(p);
    for i = 1:ok
      ch(p(i),q(i)) = C1 + C0;
    end

  end
  
  if vkol == 1
    [p,q]=find(h==0);
    [stupid,ok]=size(q);
    for i = 1:ok
      ch(1,q(i)) = C1 + C0;
    end
  end

end

