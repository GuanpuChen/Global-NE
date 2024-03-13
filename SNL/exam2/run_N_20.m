global N n q M

N=30;
n=2;



M=length(data_pre);
distance6=zeros(M,M);

for i=1:M

for j=1:M

distance6(i,j)=sqrt((data_pre(i,1)-data_pre(j,1)).^2+(data_pre(i,2)-data_pre(j,2)).^2);

end

end

distance3=zeros(M,M);
rr=1.1;
for i=1:M

for j=1:M

    if  distance6(i,j)<=rr
        
           distance3(i,j)=distance6(i,j);

    end
end
end

sum=0;
for i=1:N
    for j=i+1:M
        if distance3(i,j)~=0 
             sum=sum+1;
        end
    end
end

q=sum;

a=zeros(n,M);
a=data_pre';
x_ini=ones(n,N);

x_ini(1,:)=-0.2*ones(1,N);
x_ini(2,:)=-0.2*ones(1,N);

sigma_ini=10*ones(q,1);%
opt=zeros(n,N);%

terminal=0.00001;
%a=data_new';

[tt,X_opt,sigma_opt,round]= sensor10(x_ini,sigma_ini,terminal,a,distance3) 
 
data1=data_pre';
error=norm(X_opt-data1(:,1:N))
X_opt=X_opt';
 figure
hold on
plot(X_opt(:,1), X_opt(:,2), '*');
hold on
plot(data_pre(1:N,1), data_pre(1:N,2), 'o');
hold on
plot(data_pre(N+1:M,1), data_pre(N+1:M,2), '+');
legend('Sensors-computed','Sensors-real','Anchors');