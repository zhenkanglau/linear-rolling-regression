function [dm_test,cw_test,MSPE_ratio]=predict_model2(c,ratio,h)
y=(c(1+h:end,1)-c(1:end-h,1))./c(1:end-h,1);
x=(c(2:end,2:end)-c(1:end-1,2:end))./c(1:end-1,2:end);
T=length(y);
R=round(length(y)*ratio);
P=T-R;
beta=zeros(P-h+1,size(x,2)+1);%R is the ratio of in-sample
temp_x=[ones(R-1,1) zeros(R-1,size(x,2))];
for i=1:P-h+1
    %disp(i)
    temp_y=y(R+i-1:-1:i+1);
    temp_x(:,2:end)=x(R+i-2:-1:i,:);
    beta(i,:)=((temp_x'*temp_x)\temp_x'*temp_y)';
end
% beta1=zeros(P,size(x,2)+1);
% for i = 1:length(y)-R-h+1
%     x_temp1=x(i:i+R-1-h,:);
%     y_temp1=y(i+h:i+R-1);
%     beta1(i,:)= (([ones(R-h,1) x_temp1]'*[ones(R-h,1) x_temp1])\([ones(R-h,1) x_temp1]'*y_temp1))';
% end
y_predict=(1+beta(:,1)+sum(beta(:,2:end).*x(R+h-1:end-h,:),2)).*c(R+h:end-h,1);

diff2_model=(y_predict-c(R+h+h:end,1)).^2;
diff2_rw   =(c(R+h+h:end,1)-c(R+h:end-h,1)).^2;
MSPE_model=sum(diff2_model)/length(y_predict);
MSPE_rw   =sum(diff2_rw)/length(y_predict);
MSPE_ratio = MSPE_model/MSPE_rw;
f=diff2_rw-diff2_model+(c(R+h:end-h,1)-y_predict).^2;
d=diff2_rw-diff2_model;
const=ones(length(f),1);
[~,std_cw,mean_cw]=hac(const,f,'bandwidth',0.75*length(f)^(1/3),'intercept',false,'display','off','whiten',0,'weights','BT','smallT',false);
[~,std_dm,mean_dm]=hac(const,d,'bandwidth',0.75*length(f)^(1/3),'intercept',false,'display','off','whiten',0,'weights','BT','smallT',false);

cw_test = mean_cw/std_cw;
dm_test = mean_dm/std_dm;




end