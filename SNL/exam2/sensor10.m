function [tt,XX,sigma_opt,round]= sensor10(x_ini,sigma0,terminal,a,distance3)  %X0:(n,N), B:(n,N)

global N n q M
epoch =20000;


step=0.001



x = zeros(n,N,epoch);  

dx= zeros(n,N);
dsigma= zeros(q,1);

BR= zeros(n,N);
SR= zeros(q,1);
sigma = zeros(q,1,epoch);  



tt=0;
x_ini=ones(n,N);

x_ini(1,:)=-0.2*ones(1,N);
x_ini(2,:)=-0.2*ones(1,N);

sigma0=1*ones(q,1);%
x(:,:,1)=x_ini;
sigma(:,:,1)=sigma0;


x_1min=-10;
x_1max=10;
x_2min=-10;
x_2max=10;
sigma_min=-10;
sigma_max=10;



distance5=zeros(M,M);



c=1;
tic
    for j=1:epoch-1  %第j次迭代
       %  time(j) = 0+(j-1)*step;  %时间;  %时间
         
           %sigma_min=sigma_min*0.99;
           
           for i =1:N
                 for l=i+1:N
                     if distance3(i,l)~=0 
                        %dx(:,i)=dx(:,i)+2*sigma(c,1,j)*(x(:,i,j)-x(:,l,j));
                       
                        dsigma(c,1)=(x(:,i,j)-x(:,l,j))'*(x(:,i,j)-x(:,l,j))-distance3(i,l)^2-0.5*sigma(c,1,j);
                        distance5(i,l)=sigma(c,1,j);
                        distance5(l,i)=distance5(i,l);
                        c=c+1;

                     end
                 end
                 for l=N+1:M
                     if distance3(i,l)~=0 
                       % dx(:,i)=dx(:,i)+2*sigma(c,1,j)*(x(:,i,j)-a(:,l));
                        dsigma(c,1)=(x(:,i,j)-a(:,l))'*(x(:,i,j)-a(:,l))-distance3(i,l)^2-0.5*sigma(c,1,j);
                        distance5(i,l)=sigma(c,1,j);
                        distance5(l,i)=distance5(i,l);
                        c=c+1;
                     end
                 end
                 if c>q
                     break;
                 end
             end
           
           
             for i =1:N
                 for l=1:N
                     if distance3(i,l)~=0 
                        dx(:,i)=dx(:,i)+2*distance5(i,l)*(x(:,i,j)-x(:,l,j));
                       % dsigma(c,1)=(x(:,i,j)-x(:,l,j))'*(x(:,i,j)-x(:,l,j))-distance3(i,l)^2-0.5*sigma(c,1,j);
                        %c=c+1;
                     end
                 end
                 for l=N+1:M
                     if distance3(i,l)~=0 
                        dx(:,i)=dx(:,i)+2*distance5(i,l)*(x(:,i,j)-a(:,l));
                     %   dsigma(c,1)=(x(:,i,j)-a(:,l))'*(x(:,i,j)-a(:,l))-distance3(i,l)^2-0.5*sigma(c,1,j);
                       % c=c+1;
                     end
                 end         
             end
             

         for i =1:N
              BR(:,i)=x(:,i,j)-step*dx(:,i);
         end
 
         for i =1:N   
     
        if BR(1,i)<x_1min                     
           BR(1,i)=x_1min;
        end

        if BR(1,i)>=x_1max                 
           BR(1,i)=x_1max;
        end
            if BR(2,i)<x_2min                    
           BR(2,i)=x_2min;
        end

        if BR(2,i)>=x_2max                  
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
      
     for k=1:q
         SR(k,1)=sigma(k,1,j)+step*dsigma(k,1);
     end
     
     for k=1:q
            if  SR(k,1)<0                   
           tt=tt+1;
         end
         if  SR(k,1)<sigma_min                   
           SR(k,1)=sigma_min;
         end
       if SR(k,1)>sigma_max                   
           SR(k,1)=sigma_max;
       end
     end
     
     for k=1:q
         sigma(k,1,j+1)=SR(k,1);
     end
     dsigma= zeros(q,1);
      SR= zeros(q,1);
     c=1;  
    distance5=zeros(M,M);
           


  
   
   
       if (j>3)&&(norm(x(:,:,j+1)-x(:,:,j),'fro')<terminal)&&(norm(sigma(:,:,j+1)-sigma(:,:,j),'fro')<terminal)
            disp(['Rounds number: ',num2str(j)]);
            break;
        end
end        
         
     round=j;
        toc
        disp(['运行时间: ',num2str(toc)]);
    topt=toc;
    XX=x(:,:,round);   
    sigma_opt=sigma(:,:,round);    
         








