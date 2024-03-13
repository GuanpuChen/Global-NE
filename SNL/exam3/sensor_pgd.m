function [x1,x,XX,sigma_opt,round]= sensor_pgd(X0,sigma0,terminal,a,distance3)  %X0:(n,N), B:(n,N)

global N n q M
epoch =20000;

step = 0.0001;

beta_1=1;
beta_2=1;

x = zeros(n,N,epoch);  %n=1

dx= zeros(n,N);
dsigma= zeros(q,1);

BR= zeros(n,N);
SR= zeros(q,1);
sigma = zeros(q,1,epoch);  %n=1

%a=[[0,0,0,0,0,0,0,0.500000000000000,0,1;0,0,0,0,0,0,0,0.900000000000000,0,0]];



%time= zeros(epoch);
x(:,:,1)=X0;
sigma(:,:,1)=sigma0;

% x_1min=-7580;
% x_1max=-7400;
% x_2min=4562800;
% x_2max=4964990;
x_1min=-10;
x_1max=10;
x_2min=-10;
x_2max=10;
sigma_min=-10;
sigma_max=10;

%theta=1.2;

%gamma =zeros(epoch);

% for i=1:epoch
%     gamma(i)=4/(1*(i+1));
% end


distance5=zeros(M,M);


delta_1=[0.005,0.005]';
delta_2=[0.005,0.005]';

c=1;
tic
    for j=1:epoch-1  %第j次迭代
       %  time(j) = 0+(j-1)*step;  %时间;  %时间
         
           %sigma_min=sigma_min*1/j;
           
%            for i =1:N
%                  for l=i+1:N
%                      if distance3(i,l)~=0 
%                         %dx(:,i)=dx(:,i)+2*sigma(c,1,j)*(x(:,i,j)-x(:,l,j));
%                        
%                         dsigma(c,1)=(x(:,i,j)-x(:,l,j))'*(x(:,i,j)-x(:,l,j))-distance3(i,l)^2-0.5*sigma(c,1,j);
%                         distance5(i,l)=sigma(c,1,j);
%                         distance5(l,i)=distance5(i,l);
%                         c=c+1;
% 
%                      end
%                  end
%                  for l=N+1:M
%                      if distance3(i,l)~=0 
%                        % dx(:,i)=dx(:,i)+2*sigma(c,1,j)*(x(:,i,j)-a(:,l));
%                         dsigma(c,1)=(x(:,i,j)-a(:,l))'*(x(:,i,j)-a(:,l))-distance3(i,l)^2-0.5*sigma(c,1,j);
%                         distance5(i,l)=sigma(c,1,j);
%                         distance5(l,i)=distance5(i,l);
%                         c=c+1;
%                      end
%                  end
%                  if c>q
%                      break;
%                  end
%              end
           
           
             for i =1:N
                 for l=1:N
                     if distance3(i,l)~=0 
                        dx(:,i)=dx(:,i)+4*((x(:,i,j)-x(:,l,j))'*(x(:,i,j)-x(:,l,j))-distance3(i,l)^2)*(x(:,i,j)-x(:,l,j));
                       % dsigma(c,1)=(x(:,i,j)-x(:,l,j))'*(x(:,i,j)-x(:,l,j))-distance3(i,l)^2-0.5*sigma(c,1,j);
                        %c=c+1;
                     end
                 end
                 for l=N+1:M
                     if distance3(i,l)~=0 
                        dx(:,i)=dx(:,i)+4*((x(:,i,j)-a(:,l))'*(x(:,i,j)-a(:,l))-distance3(i,l)^2)*(x(:,i,j)-a(:,l));
                     %   dsigma(c,1)=(x(:,i,j)-a(:,l))'*(x(:,i,j)-a(:,l))-distance3(i,l)^2-0.5*sigma(c,1,j);
                       % c=c+1;
                     end
                 end         
             end
             

         for i =1:N
              BR(:,i)=x(:,i,j)-step*dx(:,i);
         end
 
         for i =1:N   
     
        if BR(1,i)<x_1min                     %将梯度投影到[0,1]上
           BR(1,i)=x_1min;
        end

        if BR(1,i)>=x_1max                  %z_max=1/64
           BR(1,i)=x_1max;
        end
            if BR(2,i)<x_2min                     %将梯度投影到[0,1]上
           BR(2,i)=x_2min;
        end

        if BR(2,i)>=x_2max                  %z_max=1/64
           BR(2,i)=x_2max;
        end
    
      end
  
      for   i =1:N        
         for m=1:n   
            x(m,i,j+1)=BR(m,i);
        end
      end
      
      dx= zeros(n,N);
      BR= zeros(n,N);
%       
%      for k=1:q
%          SR(k,1)=sigma(k,1,j)+step*dsigma(k,1);
%      end
%      
%      for k=1:q
%          if  SR(k,1)<sigma_min                     %将梯度投影到[0,1]上
%            SR(k,1)=sigma_min;
%          end
%        if SR(k,1)>sigma_max                     %将梯度投影到[0,1]上
%            SR(k,1)=sigma_max;
%        end
%      end
%      
%      for k=1:q
%          sigma(k,1,j+1)=SR(k,1);
%      end
%      dsigma= zeros(q,1);
%       SR= zeros(q,1);
%      c=1;  
%     distance5=zeros(M,M);
           


  
   
   
       if (j>3)&&(norm(x(:,:,j+1)-x(:,:,j),'fro')<terminal)&&(norm(sigma(:,:,j+1)-sigma(:,:,j),'fro')<terminal)
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
         
%          
%      t=ones(1,round);
      x_axis=zeros(n,epoch);
         x_axis=x(:,1,:);
        x1=zeros(2,round);
        for i=1:round
            x1(:,i)=x_axis(:,i);
        %    t(i)=time(i);           
        end    
% %         plot(x1(1,:),x1(2,:),'-d','linewidth',1.2);
% %         hold on
%         x_axis=x(:,2,:);
%         x2=zeros(2,round);
%         for i=1:round
%             x2(:,i)=x_axis(:,i);
%         end   
%         
%          x_axis=x(:,3,:);
%         x3=zeros(2,round);
%         for i=1:round
%             x3(:,i)=x_axis(:,i);
%         end         
%          
%           x_axis=x(:,4,:);
%         x4=zeros(2,round);
%         for i=1:round
%             x4(:,i)=x_axis(:,i);
%         end  
%         
%           x_axis=x(:,5,:);
%         x5=zeros(2,round);
%         for i=1:round
%             x5(:,i)=x_axis(:,i);
%         end    
%         
%            x_axis=x(:,6,:);
%         x6=zeros(2,round);
%         for i=1:round
%             x6(:,i)=x_axis(:,i);
%         end    
%          
%            x_axis=x(:,7,:);
%         x7=zeros(2,round);
%         for i=1:round
%             x7(:,i)=x_axis(:,i);
%         end    
%          
%               t=linspace(0,round,round); 
% 
%              figure
%      plot(t,x1(1,:),'linewidth',2.2);
%         hold on
%         plot(t,x2(1,:),'linewidth',2.2);
%         hold on
%          plot(t,x3(1,:),'linewidth',2.2);
%         hold on
% 
%         plot(t,x4(1,:),'linewidth',2.2);
%         hold on
%         plot(t,x5(1,:),'linewidth',2.2);
%         hold on
%         plot(t,x6(1,:),'linewidth',2.2);
%         hold on
%         plot(t,x7(1,:),'linewidth',2.2);
%         hold on
%        legend('sensor 1','sensor 2','sensor 3','sensor 4','sensor 5','sensor 6','sensor 7');
%        xlabel('iteration $k$','Interpreter','LaTex','FontSize',22);
%        ylabel('strategies of all sensors','Interpreter','LaTex','FontSize',22);
% %          for i=1:round
% %             s1(:,i)=sigma1(:,:,i);
%          end
%          for i=1:round
%             s2(:,i)=sigma2(:,:,i);
%          end
% %          for i=1:round
% %             s3(:,i)=sigma3(:,:,i);
% %          end
%          for i=1:round
%             s4(:,i)=sigma4(:,:,i);
%          end
% %          for i=1:round
% %             s5(:,i)=sigma5(:,:,i);
% %         end
%         
%     figure
%      plot(t,s1(1,:),'linewidth',1.2);
%         hold on
%         plot(t,s2(1,:),'linewidth',1.2);
%         hold on
% %          plot(t,s3(1,:),'linewidth',1.2);
% %         hold on
%         plot(t,s4(1,:),'linewidth',1.2);
%         hold on
% %         plot(t,s5(1,:),'linewidth',1.2);
% %         hold on