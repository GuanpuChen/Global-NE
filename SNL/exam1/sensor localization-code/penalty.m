function [t,x1,XX,sigma_opt,round]= penalty(X0,sigma0,terminal)  %X0:(n,N), B:(n,N)

global N n q
epoch =5000;

step = 0.002;



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
for j=1:epoch-1  %j-th iterate
    time(j) = 0+(j-1)*step;  %time
    
    %penalty
    sgn_x11=zeros(n,1);
    sgn_x12=zeros(n,1);
    for m=1:n
        if -x(m,1,j)+x_min<0
            sgn_x11(m)=0;
        end
        if -x(m,1,j)+x_min>0
            sgn_x11(m)=1;
        else if x(m,1,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x11(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    
    for m=1:n
        if x(m,1,j)-x_max<0
            sgn_x12(m)=0;
        end
        if x(m,1,j)-x_max>0
            sgn_x12(m)=1;
        else if x(m,1,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x12(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    sgn_x21=zeros(n,1);
    sgn_x22=zeros(n,1);
    for m=1:n
        if -x(m,2,j)+x_min<0
            sgn_x21(m)=0;
        end
        if -x(m,2,j)+x_min>0
            sgn_x21(m)=1;
        else if x(m,2,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x21(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    
    for m=1:n
        if x(m,2,j)-x_max<0
            sgn_x22(m)=0;
        end
        if x(m,2,j)-x_max>0
            sgn_x22(m)=1;
        else if x(m,2,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x22(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    sgn_x31=zeros(n,1);
    sgn_x32=zeros(n,1);
    for m=1:n
        if -x(m,3,j)+x_min<0
            sgn_x31(m)=0;
        end
        if -x(m,3,j)+x_min>0
            sgn_x31(m)=1;
        else if x(m,3,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x31(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    
    for m=1:n
        if x(m,3,j)-x_max<0
            sgn_x32(m)=0;
        end
        if x(m,3,j)-x_max>0
            sgn_x32(m)=1;
        else if x(m,3,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x32(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    sgn_x41=zeros(n,1);
    sgn_x42=zeros(n,1);
    for m=1:n
        if -x(m,4,j)+x_min<0
            sgn_x41(m)=0;
        end
        if -x(m,4,j)+x_min>0
            sgn_x41(m)=1;
        else if x(m,4,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x41(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    
    for m=1:n
        if x(m,4,j)-x_max<0
            sgn_x42(m)=0;
        end
        if x(m,4,j)-x_max>0
            sgn_x42(m)=1;
        else if x(m,4,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x42(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    sgn_x51=zeros(n,1);
    sgn_x52=zeros(n,1);
    for m=1:n
        if -x(m,5,j)+x_min<0
            sgn_x51(m)=0;
        end
        if -x(m,5,j)+x_min>0
            sgn_x51(m)=1;
        else if x(m,5,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x51(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    
    for m=1:n
        if x(m,5,j)-x_max<0
            sgn_x52(m)=0;
        end
        if x(m,5,j)-x_max>0
            sgn_x52(m)=1;
        else if x(m,5,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x52(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    sgn_x61=zeros(n,1);
    sgn_x62=zeros(n,1);
    for m=1:n
        if -x(m,6,j)+x_min<0
            sgn_x61(m)=0;
        end
        if -x(m,6,j)+x_min>0
            sgn_x61(m)=1;
        else if x(m,6,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x61(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    
    for m=1:n
        if x(m,6,j)-x_max<0
            sgn_x62(m)=0;
        end
        if x(m,6,j)-x_max>0
            sgn_x62(m)=1;
        else if x(m,6,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x62(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    sgn_x71=zeros(n,1);
    sgn_x72=zeros(n,1);
    for m=1:n
        if -x(m,7,j)+x_min<0
            sgn_x71(m)=0;
        end
        if -x(m,7,j)+x_min>0
            sgn_x71(m)=1;
        else if x(m,7,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x71(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    
    for m=1:n
        if x(m,7,j)-x_max<0
            sgn_x72(m)=0;
        end
        if x(m,7,j)-x_max>0
            sgn_x72(m)=1;
        else if x(m,7,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x72(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    sgn_x81=zeros(n,1);
    sgn_x82=zeros(n,1);
    for m=1:n
        if -x(m,8,j)+x_min<0
            sgn_x81(m)=0;
        end
        if -x(m,8,j)+x_min>0
            sgn_x81(m)=1;
        else if x(m,8,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x81(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    
    for m=1:n
        if x(m,8,j)-x_max<0
            sgn_x82(m)=0;
        end
        if x(m,8,j)-x_max>0
            sgn_x82(m)=1;
        else if x(m,8,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x82(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    sgn_x91=zeros(n,1);
    sgn_x92=zeros(n,1);
    for m=1:n
        if -x(m,9,j)+x_min<0
            sgn_x91(m)=0;
        end
        if -x(m,9,j)+x_min>0
            sgn_x91(m)=1;
        else if x(m,9,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x91(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    
    for m=1:n
        if x(m,9,j)-x_max<0
            sgn_x92(m)=0;
        end
        if x(m,9,j)-x_max>0
            sgn_x92(m)=1;
        else if x(m,9,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x92(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    sgn_x101=zeros(n,1);
    sgn_x102=zeros(n,1);
    for m=1:n
        if -x(m,10,j)+x_min<0
            sgn_x101(m)=0;
        end
        if -x(m,10,j)+x_min>0
            sgn_x101(m)=1;
        else if x(m,10,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x101(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
    
    
    for m=1:n
        if x(m,10,j)-x_max<0
            sgn_x102(m)=0;
        end
        if x(m,10,j)-x_max>0
            sgn_x102(m)=1;
        else if x(m,10,j)==0
                ran_a = -1;
                ran_b = 1;
                sgn_x102(m) = ran_a + (ran_b-ran_a) * rand(1,1);
            end
        end
    end
   
    %gradient descent
    f_1=1*(0.5*(x(:,1,j)-x(:,2,j))'*(x(:,1,j)-x(:,2,j))-4)*(x(:,1,j)-x(:,2,j)) +12*x(:,1,j)-6+1*(0.5*(x(:,1,j)-x(:,10,j))'*(x(:,1,j)-x(:,10,j))-4.5)*(x(:,1,j)-x(:,10,j)) +12*x(:,1,j)-5*sgn_x11-5*sgn_x12;
    f_2=1*(0.5*(x(:,1,j)-x(:,2,j))'*(x(:,1,j)-x(:,2,j))-4)*(x(:,2,j)-x(:,1,j))+2 +1*(0.5*(x(:,3,j)-x(:,2,j))'*(x(:,3,j)-x(:,2,j))-5)*(x(:,2,j)-x(:,3,j))+6*x(:,2,j)-5*sgn_x21-5*sgn_x22;
    f_3=1*(0.5*(x(:,2,j)-x(:,3,j))'*(x(:,2,j)-x(:,3,j))-5)*(x(:,3,j)-x(:,2,j))-4+1*(0.5*(x(:,3,j)-x(:,4,j))'*(x(:,3,j)-x(:,4,j))-5.5)*(x(:,3,j)-x(:,4,j))-5*sgn_x31-100*sgn_x32;
    f_4=1*(0.5*(x(:,4,j)-x(:,5,j))'*(x(:,4,j)-x(:,5,j))-5.2)*(x(:,4,j)-x(:,5,j))-4+1*(0.5*(x(:,4,j)-x(:,3,j))'*(x(:,4,j)-x(:,3,j))-5.5)*(x(:,4,j)-x(:,3,j))-100*sgn_x41-100*sgn_x42;
    f_5=1*(0.5*(x(:,5,j)-x(:,4,j))'*(x(:,5,j)-x(:,4,j))-5.2)*(x(:,5,j)-x(:,4,j))-10+1*(0.5*(x(:,5,j)-x(:,6,j))'*(x(:,5,j)-x(:,6,j))-6.5)*(x(:,5,j)-x(:,6,j))+6*x(:,5,j)-100*sgn_x51-100*sgn_x52;
    f_6=1*(0.5*(x(:,6,j)-x(:,5,j))'*(x(:,6,j)-x(:,5,j))-6.5)*(x(:,6,j)-x(:,5,j))+7+1*(0.5*(x(:,6,j)-x(:,1,j))'*(x(:,6,j)-x(:,1,j))-4.5)*(x(:,6,j)-x(:,1,j))-100*sgn_x61-10*sgn_x62;
    f_7=1*(0.5*(x(:,7,j)-x(:,6,j))'*(x(:,7,j)-x(:,6,j))-4.5)*(x(:,7,j)-x(:,6,j))-4.2+1*(0.5*(x(:,7,j)-x(:,8,j))'*(x(:,7,j)-x(:,8,j))-4.8)*(x(:,7,j)-x(:,8,j))-100*sgn_x71-10*sgn_x72;
    f_8=1*(0.5*(x(:,8,j)-x(:,7,j))'*(x(:,8,j)-x(:,7,j))-4.8)*(x(:,8,j)-x(:,7,j))-3.8+1*(0.5*(x(:,8,j)-x(:,9,j))'*(x(:,8,j)-x(:,9,j))-4.6)*(x(:,8,j)-x(:,9,j))-100*sgn_x81-10*sgn_x82;
    f_9=1*(0.5*(x(:,9,j)-x(:,8,j))'*(x(:,9,j)-x(:,8,j))-4.6)*(x(:,9,j)-x(:,8,j))-11+1*(0.5*(x(:,9,j)-x(:,10,j))'*(x(:,9,j)-x(:,10,j))-4.4)*(x(:,9,j)-x(:,10,j))+5*x(:,9,j)-100*sgn_x91-10*sgn_x92;
    f_10=1*(0.5*(x(:,10,j)-x(:,9,j))'*(x(:,10,j)-x(:,9,j))-4.4)*(x(:,10,j)-x(:,9,j))+7.5+1*(0.5*(x(:,10,j)-x(:,1,j))'*(x(:,10,j)-x(:,1,j))-4.5)*(x(:,10,j)-x(:,1,j))-100*sgn_x101-10*sgn_x102;
    
    
    
    
    
    
    x(:,1,j+1) = x(:,1,j)- step*f_1;
    x(:,2,j+1) = x(:,2,j)- step*f_2;
    x(:,3,j+1) = x(:,3,j)- step*f_3;
    x(:,4,j+1) = x(:,4,j)- step*f_4;
    x(:,5,j+1) = x(:,5,j)- step*f_5;
    x(:,6,j+1) = x(:,6,j)- step*f_6;
    x(:,7,j+1) = x(:,7,j)- step*f_7;
    x(:,8,j+1) = x(:,8,j)- step*f_8;
    x(:,9,j+1) = x(:,9,j)- step*f_9;
    x(:,10,j+1) = x(:,10,j)- step*f_10;

    
    
    
    
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
