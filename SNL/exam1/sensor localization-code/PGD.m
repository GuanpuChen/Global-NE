function [t,x1,XX,sigma_opt,round]= PGD(X0,sigma0,terminal)  %X0:(n,N), B:(n,N)

global N n q
epoch =5000;

step = 0.01;



x = zeros(n,N,epoch);  %n=1

sigma = zeros(q,2,epoch);  %n=1

time= zeros(epoch);
x(:,:,1)=X0;


x_min=-6;
x_max=6;



gamma =zeros(epoch);

for i=1:epoch
    gamma(i)=4/(1*(i+1));
end







       
tic
for j=1:epoch-1  %第j次迭代
    time(j) = 0+(j-1)*step;  %时间;  %时间
    
    f_1=1*(0.5*(x(:,1,j)-x(:,2,j))'*(x(:,1,j)-x(:,2,j))-4)*(x(:,1,j)-x(:,2,j)) +12*x(:,1,j)-6+1*(0.5*(x(:,1,j)-x(:,10,j))'*(x(:,1,j)-x(:,10,j))-4.5)*(x(:,1,j)-x(:,10,j)) +12*x(:,1,j);
    
    f_2=1*(0.5*(x(:,1,j)-x(:,2,j))'*(x(:,1,j)-x(:,2,j))-4)*(x(:,2,j)-x(:,1,j))+2 +1*(0.5*(x(:,3,j)-x(:,2,j))'*(x(:,3,j)-x(:,2,j))-5)*(x(:,2,j)-x(:,3,j))+6*x(:,2,j);
    f_3=1*(0.5*(x(:,2,j)-x(:,3,j))'*(x(:,2,j)-x(:,3,j))-5)*(x(:,3,j)-x(:,2,j))-4+1*(0.5*(x(:,3,j)-x(:,4,j))'*(x(:,3,j)-x(:,4,j))-5.5)*(x(:,3,j)-x(:,4,j));
    f_4=1*(0.5*(x(:,4,j)-x(:,5,j))'*(x(:,4,j)-x(:,5,j))-5.2)*(x(:,4,j)-x(:,5,j))-4+1*(0.5*(x(:,4,j)-x(:,3,j))'*(x(:,4,j)-x(:,3,j))-5.5)*(x(:,4,j)-x(:,3,j));
    f_5=1*(0.5*(x(:,5,j)-x(:,4,j))'*(x(:,5,j)-x(:,4,j))-5.2)*(x(:,5,j)-x(:,4,j))-10+1*(0.5*(x(:,5,j)-x(:,6,j))'*(x(:,5,j)-x(:,6,j))-6.5)*(x(:,5,j)-x(:,6,j))+6*x(:,5,j);
    f_6=1*(0.5*(x(:,6,j)-x(:,5,j))'*(x(:,6,j)-x(:,5,j))-6.5)*(x(:,6,j)-x(:,5,j))+7+1*(0.5*(x(:,6,j)-x(:,7,j))'*(x(:,6,j)-x(:,7,j))-4.5)*(x(:,6,j)-x(:,7,j));
    f_7=1*(0.5*(x(:,7,j)-x(:,6,j))'*(x(:,7,j)-x(:,6,j))-4.5)*(x(:,7,j)-x(:,6,j))-4.2+1*(0.5*(x(:,7,j)-x(:,8,j))'*(x(:,7,j)-x(:,8,j))-4.8)*(x(:,7,j)-x(:,8,j));
    f_8=1*(0.5*(x(:,8,j)-x(:,7,j))'*(x(:,8,j)-x(:,7,j))-4.8)*(x(:,8,j)-x(:,7,j))-3.8+1*(0.5*(x(:,8,j)-x(:,9,j))'*(x(:,8,j)-x(:,9,j))-4.6)*(x(:,8,j)-x(:,9,j));
    f_9=1*(0.5*(x(:,9,j)-x(:,8,j))'*(x(:,9,j)-x(:,8,j))-4.6)*(x(:,9,j)-x(:,8,j))-11+1*(0.5*(x(:,9,j)-x(:,10,j))'*(x(:,9,j)-x(:,10,j))-4.4)*(x(:,9,j)-x(:,10,j))+5*x(:,9,j);
    f_10=1*(0.5*(x(:,10,j)-x(:,9,j))'*(x(:,10,j)-x(:,9,j))-4.4)*(x(:,10,j)-x(:,9,j))+7.5+1*(0.5*(x(:,10,j)-x(:,1,j))'*(x(:,10,j)-x(:,1,j))-4.5)*(x(:,10,j)-x(:,1,j));
    
    
    
    
    BR1 = x(:,1,j)- step*f_1;
    BR2 = x(:,2,j)- step*f_2;
    BR3 = x(:,3,j)- step*f_3;
    BR4 = x(:,4,j)- step*f_4;
    BR5 = x(:,5,j)- step*f_5;
    BR6= x(:,6,j)- step*f_6;
    BR7 = x(:,7,j)- step*f_7;
    BR8 = x(:,8,j)- step*f_8;
    BR9 = x(:,9,j)- step*f_9;
    BR10= x(:,10,j)- step*f_10;
    
    
    
    
    for m=1:n
        if BR1(m)<x_min                     %将梯度投影到[0,1]上
            BR1(m)=x_min  ;
        end
        
        if BR1(m)>=x_max                  %z_max=1/64
            BR1(m)=x_max;
        end
    end
    
    
    
    for m=1:n
        if BR2(m)<x_min                     %将梯度投影到[0,1]上
            BR2(m)=x_min  ;
        end
        
        if BR2(m)>=x_max                  %z_max=1/64
            BR2(m)=x_max;
        end
    end
    
    for m=1:n
        if BR3(m)<x_min                     %将梯度投影到[0,1]上
            BR3(m)=x_min  ;
        end
        
        if BR3(m)>=x_max                  %z_max=1/64
            BR3(m)=x_max;
        end
    end
    
    for m=1:n
        if BR4(m)<x_min                     %将梯度投影到[0,1]上
            BR4(m)=x_min  ;
        end
        
        if BR4(m)>=x_max                  %z_max=1/64
            BR4(m)=x_max;
        end
    end
    
    for m=1:n
        if BR5(m)<x_min                     %将梯度投影到[0,1]上
            BR5(m)=x_min  ;
        end
        
        if BR5(m)>=x_max                  %z_max=1/64
            BR5(m)=x_max;
        end
    end
    
    for m=1:n
        if BR6(m)<x_min                     %将梯度投影到[0,1]上
            BR6(m)=x_min  ;
        end
        
        if BR6(m)>=x_max                  %z_max=1/64
            BR6(m)=x_max;
        end
    end
    for m=1:n
        if BR7(m)<x_min                     %将梯度投影到[0,1]上
            BR7(m)=x_min  ;
        end
        
        if BR7(m)>=x_max                  %z_max=1/64
            BR7(m)=x_max;
        end
    end
    
    for m=1:n
        if BR8(m)<x_min                     %将梯度投影到[0,1]上
            BR8(m)=x_min  ;
        end
        
        if BR8(m)>=x_max                  %z_max=1/64
            BR8(m)=x_max;
        end
    end
    
    for m=1:n
        if BR9(m)<x_min                     %将梯度投影到[0,1]上
            BR9(m)=x_min  ;
        end
        
        if BR9(m)>=x_max                  %z_max=1/64
            BR9(m)=x_max;
        end
    end
    for m=1:n
        if BR10(m)<x_min                     %将梯度投影到[0,1]上
            BR10(m)=x_min  ;
        end
        
        if BR10(m)>=x_max                  %z_max=1/64
            BR10(m)=x_max;
        end
    end
    
    
    
    for m=1:n
        
        x(m,1,j+1)=BR1(m);
    end
    
    for m=1:n
        
        x(m,2,j+1)=BR2(m);
    end
    
    
    for m=1:n
        
        x(m,3,j+1)=BR3(m);
    end
    
    for m=1:n
        
        x(m,4,j+1)=BR4(m);
    end
    
    for m=1:n
        
        x(m,5,j+1)=BR5(m);
    end
    
    for m=1:n
        
        x(m,6,j+1)=BR6(m);
    end
    
    for m=1:n
        
        x(m,7,j+1)=BR7(m);
    end
    
    for m=1:n
        
        x(m,8,j+1)=BR8(m);
    end
    
    for m=1:n
        
        x(m,9,j+1)=BR9(m);
    end
    
    for m=1:n
        
        x(m,10,j+1)=BR10(m);
    end
    
    %
    
    
    
    
    if (j>3)&&(norm(x(:,:,j+1)-x(:,:,j),'fro')<terminal)
        disp(['Rounds number: ',num2str(j)]);
        break;
    end
end
    
round=j;
toc
disp(['运行时间: ',num2str(toc)]);
topt=toc;
XX=x(:,:,round);   %最优解
sigma_opt=sigma(:,:,round);   %最优解

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
