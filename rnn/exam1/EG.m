function [x1,x2,XX,round]= EG(X0,terminal) %X0:(n,N), B:(n,N)

global N n 
epoch = 10000;

step = 0.94;

step = 0.015;

x = zeros(n,N,epoch);  %n=1

time= zeros(epoch);
x(:,:,1)=X0;


x_min=-5;
x_max=5;
y_min=-5;
y_max=5;





tic
for j=1:epoch-1  %第j次迭代
    time(j) = 0+(j-1)*step;  %时间;  %时间
    
    
    
    %waiting stage
    
    
    f_1=(-2*x(1,1,j)-0.5*x(1,2,j))*exp(-x(1,1,j)^2-0.5*x(1,2,j)*x(1,1,j))/(1+exp(-x(1,1,j)^2-0.5*x(1,2,j)*x(1,1,j)))+0.2*x(1,1,j)-0.2;%U_1的梯度项  如 x(1,1,j)+0.5*(5-1)+0.01*x(1,2,j)  +sigma(1,1,j)*x(1,2,j)
    
    
    
    f_2=   -(0.5*x(1,1,j))*exp(-x(1,1,j)^2-0.5*x(1,2,j)*x(1,1,j))/(1+exp(-x(1,1,j)^2-0.5*x(1,2,j)*x(1,1,j)))-0.2*x(1,2,j)+0.2;
    
    BR_1=x(1,1,j)-step*f_1;
    
    BR_2=x(1,2,j)+step*f_2;
    
    if BR_1<x_min
        BR_1=x_min  ;
    end
    
    if BR_1>=x_max
        BR_1=x_max;
    end
    
    if BR_2<y_min
        BR_2=y_min  ;
    end
    
    if BR_2>=y_max
        BR_2=y_max;
    end
    
    %new stage
    
    f_11=(-2*BR_1-0.5*BR_2)*exp(-BR_1^2-0.5*BR_2*BR_1)/(1+exp(-BR_1^2-0.5*BR_2*BR_1))+0.2*BR_1-0.2;
    
    
    
    f_22=   -(0.5*BR_1)*exp(-BR_1^2-0.5*BR_2*BR_1)/(1+exp(-BR_1^2-0.5*BR_2*BR_1))-0.2*BR_2+0.2;
    
    
    BR_11=x(1,1,j)-step*f_11;
    
    BR_22=x(1,2,j)+step*f_22;
    
    
    
    if BR_11<x_min
        BR_11=x_min  ;
    end
    
    if BR_11>=x_max
        BR_11=x_max;
    end
    
    if BR_22<y_min
        BR_22=y_min  ;
    end
    
    if BR_22>=y_max
        BR_22=y_max;
    end
    
    
    
    x(:,1,j+1) = BR_11;
    
    x(:,2,j+1) = BR_22;
    
    
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

figure

plot(x1(1,:),x2(1,:),'linewidth',1.2);
hold on

figure
plot(t,x1(1,:),'linewidth',1.2);
hold on
plot(t,x2(1,:),'linewidth',1.2);