%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 版权声明：
%     本程序的详细中文注释请参考
%     黄小平，王岩，缪鹏程.粒子滤波原理及应用[M].电子工业出版社，2017.4
%     书中有原理介绍+例子+程序+中文注释
%     如果此程序有错误，请对提示修改
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  函数功能：粒子滤波用于电源寿命预测
% function main
 

load Battery_Capacity
%%load Battery_Capacity
N=length(A5Cycle);
% error('下面的参数M请参考书中的值设置，然后删除本行代码')
M=200;   %%粒子数
Future_Cycle=100;  % 未来趋势
if N>260
    N=260;   % 过滤大于260的数字
end
 %过程噪声协方差Q
cita=1e-4;
wa=0.000001;wb=0.01;wc=0.1;wd=0.0001;
Q=cita*diag([wa,wb,wc,wd]);
 %驱动矩阵
F=eye(4);
 %观测噪声协方差
R=0.001;
 
a=-0.0000083499;b=0.055237;c=0.90097;d=-0.00088543;
X0=[a,b,c,d]';
 %滤波器状态初始化
Xpf=zeros(4,N);
Xpf(:,1)=X0;
 
% 粒子集初始化
Xm=zeros(4,M,N);
for i=1:M
    Xm(:,i,1)=X0+sqrtm(Q)*randn(4,1);
end
 
% 观测量
Z(1,1:N)=A12Capacity(1:N,:)';
 
Zm=zeros(1,M,N);
 
Zpf=zeros(1,N);
 
W=zeros(N,M);
 %粒子滤波算法
for k=2:N
    %  采样
    for i=1:M
        Xm(:,i,k)=F*Xm(:,i,k-1)+sqrtm(Q)*randn(4,1);
    end
        
    % 权值重要性计算
    for i=1:M
   
        Zm(1,i,k)=feval('hfun',Xm(:,i,k),k);
       
        W(k,i)=exp(-(Z(1,k)-Zm(1,i,k))^2/2/R)+1e-99;
    end,
 
    W(k,:)=W(k,:)./sum(W(k,:));
  
    % 重采样
    outIndex = randomR(1:M,W(k,:)');      % 调用外部函数
     % 得到新样本
    Xm( :,  :,  k)=Xm(  :,  outIndex,  k);
    % 滤波后的状态更新
    Xpf(:,k)=[mean(Xm(1,:,k));mean(Xm(2,:,k));mean(Xm(3,:,k));mean(Xm(4,:,k))];
    % 更新后的状态计算
    Zpf(1,k)=feval('hfun',Xpf(:,k),k);
end
 %预测未来电容的趋势
start=N-Future_Cycle;
for k=start:N
    Zf(1,k-start+1)=feval('hfun',Xpf(:,start),k);
    Xf(1,k-start+1)=k;
end

Xreal=[a*ones(1,M);b*ones(1,M);c*ones(1,M);d*ones(1,M)];
figure
subplot(2,2,1);
hold on;box on;
plot(Xpf(1,:),'-r.');plot(Xreal(1,:),'-b.')
legend('粒子滤波后的a','平均值a')
subplot(2,2,2);
hold on;box on;
plot(Xpf(2,:),'-r.');plot(Xreal(2,:),'-b.')
legend('粒子滤波后的b','平均值b')
subplot(2,2,3);
hold on;box on;
plot(Xpf(3,:),'-r.');plot(Xreal(3,:),'-b.')
legend('粒子滤波后的c','平均值c')
subplot(2,2,4);
hold on;box on;
plot(Xpf(4,:),'-r.');plot(Xreal(4,:),'-b.')
legend('粒子滤波后的d','平均值d')

figure
hold on;box on;
plot(Z,'-b.') 
plot(Zpf,'-r.')
plot(Xf,Zf,'-g.') 
bar(start,1,'y')
legend('实验测量数据','滤波估计数据','自然预测数据')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






