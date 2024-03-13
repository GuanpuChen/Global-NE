function [x1,x2,XX,sigma_opt,round]= Alg1(X0,sigma0,terminal) %X0:(n,N), B:(n,N)

global N n 
epoch = 20000;

step = 0.065;

beta_1=1;
beta_2=1;

x = zeros(n,N,epoch);  %n=1
sigma = zeros(n,1,epoch);  %n=1
time= zeros(epoch);
x(:,:,1)=X0;
sigma(:,:,1)=0.02;

x_min=-5;
x_max=5;
y_min=-5;
y_max=5;

sigma_min=0.0001;
sigma_max=0.096;

gamma =zeros(epoch);

for i=1:epoch
    gamma(i)=2/(1*(i+1));
end

tic
for j=1:epoch-1  %iterates
    time(j) = 0+(j-1)*step;  %time
    
    
    
    
    f_1=-sigma(1,1,j)*(2*x(1,1,j)+0.5*x(1,2,j))+0.2*x(1,1,j)-0.2;
    
    g_1=-x(1,1,j)^2-0.5*x(1,2,j)*x(1,1,j)-log(sigma(1,1,j)/(1-sigma(1,1,j)));
    
    
    
    
    f_2=    -0.5*sigma(1,1,j)*x(1,1,j)-0.2*x(1,2,j)+0.2;
    
    
    %BR_1=x(1,1,j)-step*f_1;
    
    BR_1=log(x(1,1,j)-x_min)-log(x_max-x(1,1,j))-step*f_1;
    
    %BR_11=sigma(1,1,j)+step*g_1;
    BR_11=log(sigma(1,1,j)-sigma_min)-log(sigma_max-sigma(1,1,j))+step*g_1;
    
    %BR_2=x(1,2,j)+step*f_2;
    BR_2=log(x(1,2,j)-y_min)-log(y_max-x(1,2,j))+step*f_2;
    

%     if BR_1<x_min                     
%         BR_1=x_min  ;
%     end
%     
%     if BR_1>=x_max                
%         BR_1=x_max;
%     end
%     
%     
%     if BR_11<sigma_min                     
%         BR_11=sigma_min     ;
%     end
%     
%     if BR_11>=sigma_max                   
%         BR_11=sigma_max    ;
%     end
%     
%     
%     if BR_2<y_min    
%         BR_2=y_min  ;
%     end
%     
%     if BR_2>=y_max
%         BR_2=y_max;
%     end
    
    
    
    %x(:,1,j+1) = (BR_1);
   x(:,1,j+1) = (x_min+x_max*exp(BR_1))/(exp(BR_1)+1);
    
    
   % sigma(:,1,j+1) = (BR_11);
    sigma(:,1,j+1) = (sigma_min+sigma_max*exp(BR_11))/(exp(BR_11)+1);
    
    %x(:,2,j+1) = (BR_2);
    x(:,2,j+1) = (y_min+y_max*exp(BR_2))/(exp(BR_2)+1);
    
    
    
    
    if (j>3)&&(norm(x(:,:,j+1)-x(:,:,j),'fro')<terminal)&&(norm(sigma(:,:,j+1)-sigma(:,:,j),'fro')<terminal)
        disp(['Rounds number: ',num2str(j)]);
        break;
    end
end

round=j;
toc
disp(['运行时间: ',num2str(toc)]);

topt=toc;
XX=x(:,:,round);   %global NE
sigma_opt=sigma(:,:,round);   





x_axis=zeros(n,epoch);
x_axis=x(:,1,:);
x1=zeros(2,round);
for i=1:round
    x1(:,i)=x_axis(:,i);
    t(i)=time(i);
end
x_axis=x(:,2,:);
x2=zeros(2,round);
for i=1:round
    x2(:,i)=x_axis(:,i);
end

% figure
% plot(t,x1(1,:),'linewidth',1.2);
% hold on
% plot(t,x2(1,:),'linewidth',1.2);
% hold on



figure
plot(x1(1,:),x2(1,:),'linewidth',1.2);
hold on