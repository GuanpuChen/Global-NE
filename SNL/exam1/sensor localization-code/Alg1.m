function [t,x4,XX,sigma_opt,round]= Alg1(X0,sigma0,terminal)  %X0:(n,N), B:(n,N)

global N n q
epoch =5000;

step = 0.002;



x = zeros(n,N,epoch);  

x_sum = zeros(n,N,epoch); 

sigma = zeros(q,1,epoch);  
sigma_sum = zeros(q,1,epoch);  

time= zeros(epoch);
x(:,:,1)=X0;


x_min=-6;
x_max=6;



gamma =zeros(epoch,1);

Ms=zeros(epoch,1);  

for i=1:epoch
    gamma(i)=0.0637/((i)^0.5);    %step size
end 








 xi=zeros(q,1);
tic
for j=1:epoch-1  %j-th interate
    time(j) = 0+(j-1)*step;  %time
    
    
    %gradient descent
    f_1=1*sigma(1,1,j)*(x(:,1,j)-x(:,2,j)) +12*x(:,1,j)-6+1*sigma(2,1,j)*(x(:,1,j)-x(:,10,j)) +12*x(:,1,j);
    f_2=1*sigma(3,1,j)*(x(:,2,j)-x(:,1,j))+2+ 1*sigma(4,1,j)*(x(:,2,j)-x(:,3,j))+6*x(:,2,j);
    f_3=1*sigma(5,1,j)*(x(:,3,j)-x(:,2,j))-4+1*sigma(6,1,j)*(x(:,3,j)-x(:,4,j));
    f_4=1*sigma(7,1,j)*(x(:,4,j)-x(:,5,j))-4+1*sigma(8,1,j)*(x(:,4,j)-x(:,3,j));
    f_5=1*sigma(9,1,j)*(x(:,5,j)-x(:,4,j))-10+1*sigma(10,1,j)*(x(:,5,j)-x(:,6,j))+6*x(:,5,j);
    f_6=1*sigma(11,1,j)*(x(:,6,j)-x(:,5,j))+7+1*sigma(12,1,j)*(x(:,6,j)-x(:,7,j));
    f_7=1*sigma(13,1,j)*(x(:,7,j)-x(:,6,j))-4.2+1*sigma(14,1,j)*(x(:,7,j)-x(:,8,j));
    f_8=1*sigma(15,1,j)*(x(:,8,j)-x(:,7,j))-3.8+1*sigma(16,1,j)*(x(:,8,j)-x(:,9,j));
    f_9=1*sigma(17,1,j)*(x(:,9,j)-x(:,8,j))-11+1*sigma(18,1,j)*(x(:,9,j)-x(:,10,j))+5*x(:,9,j);
    f_10=1*sigma(19,1,j)*(x(:,10,j)-x(:,9,j))+7.5+1*sigma(20,1,j)*(x(:,10,j)-x(:,1,j));
    
    %bregman damping
    for m=1:n
        BR1(m)=log(x(m,1,j)-x_min)-log(x_max-x(m,1,j))- gamma(j)*f_1(m);
    end
    
    for m=1:n
        BR2(m)=log(x(m,2,j)-x_min)-log(x_max-x(m,2,j))- gamma(j)*f_2(m);
    end
    
    for m=1:n
        BR3(m)=log(x(m,3,j)-x_min)-log(x_max-x(m,3,j))- gamma(j)*f_3(m);
    end
    
    for m=1:n
        BR4(m)=log(x(m,4,j)-x_min)-log(x_max-x(m,4,j))- gamma(j)*f_4(m);
    end
    
    for m=1:n
        BR5(m)=log(x(m,5,j)-x_min)-log(x_max-x(m,5,j))- gamma(j)*f_5(m);
    end
    
    for m=1:n
        BR6(m)=log(x(m,6,j)-x_min)-log(x_max-x(m,6,j))- gamma(j)*f_6(m);
    end
    
    for m=1:n
        BR7(m)=log(x(m,7,j)-x_min)-log(x_max-x(m,7,j))- gamma(j)*f_7(m);
    end
    
    for m=1:n
        BR8(m)=log(x(m,8,j)-x_min)-log(x_max-x(m,8,j))- gamma(j)*f_8(m);
    end
    
    for m=1:n
        BR9(m)=log(x(m,9,j)-x_min)-log(x_max-x(m,9,j))- gamma(j)*f_9(m);
    end
    
    for m=1:n
        BR10(m)=log(x(m,10,j)-x_min)-log(x_max-x(m,10,j))- gamma(j)*f_10(m);
    end
    
    
    
    
    sum_ga=0;
    for p=1:j
        sum_ga=sum_ga+gamma(p);
    end
    
    sum_x1=[0,0]';
    sum_x2=[0,0]';
    sum_x3=[0,0]';
    sum_x4=[0,0]';
    sum_x5=[0,0]';
    sum_x6=[0,0]';
    sum_x7=[0,0]';
    sum_x8=[0,0]';
    sum_x9=[0,0]';
    sum_x10=[0,0]';
    sum_sigma=zeros(q,1);
    for p=1:j
        sum_x1=sum_x1+gamma(p)*x(:,1,p);
        sum_x2=sum_x2+gamma(p)*x(:,2,p);
        sum_x3=sum_x3+gamma(p)*x(:,3,p);
        sum_x4=sum_x4+gamma(p)*x(:,4,p);
        sum_x5=sum_x5+gamma(p)*x(:,5,p);
        sum_x6=sum_x6+gamma(p)*x(:,6,p);
        sum_x7=sum_x7+gamma(p)*x(:,7,p);
        sum_x8=sum_x8+gamma(p)*x(:,8,p);
        sum_x9=sum_x9+gamma(p)*x(:,9,p);
        sum_x10=sum_x10+gamma(p)*x(:,10,p);
        sum_sigma= sum_sigma+gamma(p)*sigma(:,1,p);
    end
    
    
    x_sum(:,1,j)=sum_x1/sum_ga;
    x_sum(:,2,j)=sum_x2/sum_ga;
    x_sum(:,3,j)=sum_x3/sum_ga;
    x_sum(:,4,j)=sum_x4/sum_ga;
    x_sum(:,5,j)=sum_x5/sum_ga;
    x_sum(:,6,j)=sum_x6/sum_ga;
    x_sum(:,7,j)=sum_x7/sum_ga;
    x_sum(:,8,j)=sum_x8/sum_ga;
    x_sum(:,9,j)=sum_x9/sum_ga;
    x_sum(:,10,j)=sum_x10/sum_ga;
    
    sigma_sum(:,1,j)=sum_sigma/sum_ga;
    
    
    g_1=[0.5*(x(:,1,j)-x(:,2,j))'*(x(:,1,j)-x(:,2,j))-4;
        0.5*(x(:,1,j)-x(:,10,j))'*(x(:,1,j)-x(:,10,j))-4.5;
        0.5*(x(:,1,j)-x(:,2,j))'*(x(:,1,j)-x(:,2,j))-4;
        0.5*(x(:,3,j)-x(:,2,j))'*(x(:,3,j)-x(:,2,j))-5 ;
        0.5*(x(:,2,j)-x(:,3,j))'*(x(:,2,j)-x(:,3,j))-5;
        0.5*(x(:,3,j)-x(:,4,j))'*(x(:,3,j)-x(:,4,j))-5.5;
        0.5*(x(:,4,j)-x(:,5,j))'*(x(:,4,j)-x(:,5,j))-5.2;
        0.5*(x(:,4,j)-x(:,3,j))'*(x(:,4,j)-x(:,3,j))-5.5;
        0.5*(x(:,5,j)-x(:,4,j))'*(x(:,5,j)-x(:,4,j))-5.2;
        0.5*(x(:,5,j)-x(:,6,j))'*(x(:,5,j)-x(:,6,j))-6.5;
        0.5*(x(:,6,j)-x(:,5,j))'*(x(:,6,j)-x(:,5,j))-6.5;
        0.5*(x(:,6,j)-x(:,7,j))'*(x(:,6,j)-x(:,7,j))-4.5;
        0.5*(x(:,7,j)-x(:,6,j))'*(x(:,7,j)-x(:,6,j))-4.5;
        0.5*(x(:,7,j)-x(:,8,j))'*(x(:,7,j)-x(:,8,j))-4.8;
        0.5*(x(:,8,j)-x(:,7,j))'*(x(:,8,j)-x(:,7,j))-4.8;
        0.5*(x(:,8,j)-x(:,9,j))'*(x(:,8,j)-x(:,9,j))-4.6;
        0.5*(x(:,9,j)-x(:,8,j))'*(x(:,9,j)-x(:,8,j))-4.6;
        0.5*(x(:,9,j)-x(:,10,j))'*(x(:,9,j)-x(:,10,j))-4.4;
        0.5*(x(:,10,j)-x(:,9,j))'*(x(:,10,j)-x(:,9,j))-4.4;
        0.5*(x(:,10,j)-x(:,1,j))'*(x(:,10,j)-x(:,1,j))-4.5;
        ]-sigma(:,1,j);
    
    
    
    SR1=sigma(:,1,j)+gamma(j)*g_1;
    
    
    
    
    
    
    sum_g=0;
    for m=1:q
        sum_g=sum_g+g_1(m)^2;
    end
    
    Ms(j)=(2*(f_1(1)^2+f_2(1)^2+f_3(1)^2+f_4(1)^2+f_5(1)^2+f_6(1)^2+f_7(1)^2+f_8(1)^2+f_9(1)^2+f_10(1)^2)+sum_g)^0.5;
    
    
    
    
    
    
    
    
    %output feedback
    for m=1:n
        x(m,1,j+1)=(x_min+x_max*exp(BR1(m)))/(exp(BR1(m))+1);
    end
    
    for m=1:n
        x(m,2,j+1)=(x_min+x_max*exp(BR2(m)))/(exp(BR2(m))+1);
    end
    
    for m=1:n
        x(m,3,j+1)=(x_min+x_max*exp(BR3(m)))/(exp(BR3(m))+1);
    end
    
    for m=1:n
        x(m,4,j+1)=(x_min+x_max*exp(BR4(m)))/(exp(BR4(m))+1);
    end
    
    for m=1:n
        x(m,5,j+1)=(x_min+x_max*exp(BR5(m)))/(exp(BR5(m))+1);
    end
    
    for m=1:n
        x(m,6,j+1)=(x_min+x_max*exp(BR6(m)))/(exp(BR6(m))+1);
    end
    
    for m=1:n
        x(m,7,j+1)=(x_min+x_max*exp(BR7(m)))/(exp(BR7(m))+1);
    end
    
    for m=1:n
        x(m,8,j+1)=(x_min+x_max*exp(BR8(m)))/(exp(BR8(m))+1);
    end
    
    for m=1:n
        x(m,9,j+1)=(x_min+x_max*exp(BR9(m)))/(exp(BR9(m))+1);
    end
    
    for m=1:n
        x(m,10,j+1)=(x_min+x_max*exp(BR10(m)))/(exp(BR10(m))+1);
    end
    
    
    
    
   
    
    A=[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
        0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
        0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
        0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0;
        0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0;
        0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0;
        0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0;
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0;
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0;
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1;
        ];
    
    
    b=[-24;-6;0;0;-6;0;0;0;-5;0];
    
    H =2* eye(q);
    f = -2*[SR1(1); SR1(2);SR1(3);SR1(4);SR1(5);SR1(6); SR1(7); SR1(8);SR1(9);SR1(10);SR1(11); SR1(12);SR1(13); SR1(14);SR1(15); SR1(16);SR1(17); SR1(18);SR1(19); SR1(20)];
    ub = [ ];
    x0 = [];
    lb = [ ];
    %  options = optimoptions('quadprog','Display','iter','MaxIterations',300,'TolFun',1e-16);
    %  y= quadprog(H,f,B(:,:,i),b(:,i),[],[],lb,ub,x0,options);
    options = optimoptions('quadprog','Display','iter');
    y= quadprog(H,f,-A,-b,[],[],lb,[],[],options);
    
    
    for m=1:q
        sigma(m,1,j+1)=y(m);
    end
    
    
    
    
    
    
    
    
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








t=ones(1,round);
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

x_axis=x(:,2,:);
x2=zeros(2,round);
for i=1:round
    x2(:,i)=x_axis(:,i);
end

x_axis=x(:,3,:);
x3=zeros(2,round);
for i=1:round
    x3(:,i)=x_axis(:,i);
end

x_axis=x(:,4,:);
x4=zeros(2,round);
for i=1:round
    x4(:,i)=x_axis(:,i);
end

x_axis=x(:,5,:);
x5=zeros(2,round);
for i=1:round
    x5(:,i)=x_axis(:,i);
end

x_axis=x(:,6,:);
x6=zeros(2,round);
for i=1:round
    x6(:,i)=x_axis(:,i);
end

x_axis=x(:,7,:);
x7=zeros(2,round);
for i=1:round
    x7(:,i)=x_axis(:,i);
end

x_axis=x(:,8,:);
x8=zeros(2,round);
for i=1:round
    x8(:,i)=x_axis(:,i);
end

x_axis=x(:,9,:);
x9=zeros(2,round);
for i=1:round
    x9(:,i)=x_axis(:,i);
end

x_axis=x(:,10,:);
x10=zeros(2,round);
for i=1:round
    x10(:,i)=x_axis(:,i);
end

%  t=linspace(0,topt,round);
t=linspace(0,round,round);

figure
plot(t,x1(1,:),'linewidth',1.2);
hold on
plot(t,x2(1,:),'linewidth',1.2);
hold on
plot(t,x3(1,:),'linewidth',1.2);
hold on
plot(t,x4(1,:),'linewidth',1.2);
hold on
plot(t,x5(1,:),'linewidth',1.2);
hold on
plot(t,x6(1,:),'linewidth',1.2);
hold on
plot(t,x7(1,:),'linewidth',1.2);
hold on
plot(t,x8(1,:),'linewidth',1.2);
hold on
plot(t,x9(1,:),'linewidth',1.2);
hold on
plot(t,x10(1,:),'linewidth',1.2);
hold on




          
          