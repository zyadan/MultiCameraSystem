%---------------------------------------%
% Copyright (c) 2017 Yadan Zeng
% Quanzhou Institute of Equipment Manufacturing
% Vicky Zeng <zyadan@fjirsm.ac.cn>
% Last modified: Tue.29 June 2017 10:17:19 PM 

% Using LM algorithm to obtain the parameters
%---------------------------------------%


close all;clear all;
clc;

% only import the value, if containing the text,use .data
% the points are the middle of the footpoints

P1 = [];P2 = [];P3 = [];P4 = [];P5 = []; P6 = [];
NN = 12;
for ordern = 1:8;
    import1x = ['P1',int2str(ordern),'(1,:) = importdata(''P',int2str(ordern),'1x.txt'')'];
    import1y = ['P1',int2str(ordern),'(2,:) = importdata(''P',int2str(ordern),'1y.txt'')'];
    import1z = ['P1',int2str(ordern),'(3,:) = importdata(''P',int2str(ordern),'1z.txt'')'];
    
    import2x = ['P2',int2str(ordern),'(1,:) = importdata(''P',int2str(ordern),'2x.txt'')'];
    import2y = ['P2',int2str(ordern),'(2,:) = importdata(''P',int2str(ordern),'2y.txt'')'];
    import2z = ['P2',int2str(ordern),'(3,:) = importdata(''P',int2str(ordern),'2z.txt'')'];
    
    import3x = ['P3',int2str(ordern),'(1,:) = importdata(''P',int2str(ordern),'3x.txt'')'];
    import3y = ['P3',int2str(ordern),'(2,:) = importdata(''P',int2str(ordern),'3y.txt'')'];
    import3z = ['P3',int2str(ordern),'(3,:) = importdata(''P',int2str(ordern),'3z.txt'')'];
    
    import4x = ['P4',int2str(ordern),'(1,:) = importdata(''P',int2str(ordern),'4x.txt'')'];
    import4y = ['P4',int2str(ordern),'(2,:) = importdata(''P',int2str(ordern),'4y.txt'')'];
    import4z = ['P4',int2str(ordern),'(3,:) = importdata(''P',int2str(ordern),'4z.txt'')'];
    
    import5x = ['P5',int2str(ordern),'(1,:) = importdata(''P',int2str(ordern),'5x.txt'')'];
    import5y = ['P5',int2str(ordern),'(2,:) = importdata(''P',int2str(ordern),'5y.txt'')'];
    import5z = ['P5',int2str(ordern),'(3,:) = importdata(''P',int2str(ordern),'5z.txt'')'];
    
    import6x = ['P6',int2str(ordern),'(1,:) = importdata(''P',int2str(ordern),'6x.txt'')'];
    import6y = ['P6',int2str(ordern),'(2,:) = importdata(''P',int2str(ordern),'6y.txt'')'];
    import6z = ['P6',int2str(ordern),'(3,:) = importdata(''P',int2str(ordern),'6z.txt'')'];
    
    eval(import1x);eval(import1y); eval(import1z);
    eval(import2x);eval(import2y); eval(import2z);
    eval(import3x);eval(import3y); eval(import3z);
    eval(import4x);eval(import4y); eval(import4z);
    eval(import5x);eval(import5y); eval(import5z);
    eval(import6x);eval(import6y); eval(import6z);
    
    joint1 = ['P1 = [P1',int2str(ordern),', P1]'];
    joint2 = ['P2 = [P2',int2str(ordern),', P2]'];
    joint3 = ['P3 = [P3',int2str(ordern),', P3]'];
    joint4 = ['P4 = [P4',int2str(ordern),', P4]'];
    joint5 = ['P5 = [P5',int2str(ordern),', P5]'];
    joint6 = ['P6 = [P6',int2str(ordern),', P6]'];
    eval(joint1);eval(joint2);eval(joint3);
    eval(joint4);eval(joint5);eval(joint6);
    
end





num = 1400;
Colnum = 1;  
abCol = [];  %error value occupied Column

% X(i,k)= Pi(1,k+1)- Pi(1,k)
% so k is equal to the total number-1 or using num-1
for k = 1:num;
    for i=1:6;
        ssx = ['X(',int2str(i),',',int2str(k),')','= P',int2str(i) '(1,', int2str(k+1) ')- P', int2str(i),'(1,' int2str(k),')'];
        ssy = ['Y(',int2str(i),',',int2str(k),')','= P',int2str(i) '(2,', int2str(k+1) ')- P', int2str(i),'(2,' int2str(k),')'];
        ssz = ['Z(',int2str(i),',',int2str(k),')','= P',int2str(i) '(3,', int2str(k+1) ')- P', int2str(i),'(3,' int2str(k),')'];
      eval(ssx);
      eval(ssy);
      eval(ssz);
    end    
    
    % remove the point at the corner of the path
      if abs(X(i,k)) >100 || abs(Y(i,k))>100
          abCol(Colnum) = k;
          Colnum = Colnum+1;
      end    
end
X(:,abCol) = [];
Y(:,abCol) = [];
Z(:,abCol) = [];
num = num - length(abCol);

% Qij(k)= X(i,k) * X(i,k)+ Y(i,k) * Y(i,k)+ Z(i,k) * Z(i,k)          %if i==j
% Qij(k)= 2 * (X(i,k) * X(j,k)+ Y(i,k) * Y(j,k)+ Z(i,k) * Z(j,k))    % if i~=j
for k = 1:num;
    for i = 1:6;
       for j = i:6;
           if i==j
                qq1 = ['Q',int2str(i),int2str(j),'(',int2str(k),')','= X(',int2str(i),',',int2str(k),') * X(',int2str(i),',',int2str(k),')+ Y(',int2str(i),',',int2str(k),') * Y(',int2str(i),',',int2str(k),')+ Z(',int2str(i),',',int2str(k),') * Z(',int2str(i),',',int2str(k),')'];
    %            cc1 = ['C',int2str(i),int2str(j),'= S',int2str(i),'* S',int2str(i)];
                eval(qq1);
    %            eval(cc1);
           end
           if i~=j
               qq2 = ['Q',int2str(i),int2str(j),'(',int2str(k),')','=2*( X(',int2str(i),',',int2str(k),') * X(',int2str(j),',',int2str(k),')+ Y(',int2str(i),',',int2str(k),') * Y(',int2str(j),',',int2str(k),')+ Z(',int2str(i),',',int2str(k),') * Z(',int2str(j),',',int2str(k),'))']; 
               eval(qq2);
           end
        end
    end
end
Q11 = Q11'; Q12 = Q12'; Q13 = Q13'; Q14 = Q14'; Q15 = Q15';Q16 = Q16';Q22 = Q22'; Q23 = Q23'; Q24 = Q24'; Q25 = Q25'; Q26 = Q26'; Q33 = Q33'; Q34 = Q34';Q35 = Q35';Q36 = Q36';Q44 = Q44';Q45 = Q45'; Q46 = Q46'; Q55 = Q55'; Q56 = Q56'; Q66 = Q66'; 
%% liner least squares method
% % Q*C=DIS the norm equal 30 add ones, regress need constant term to ensure
% % the validity of regress, unless "R-square and the F statistic are not
% % well-defined unless X has a column of ones." using regress
% Q = [ones(num,1),Q11, Q12, Q13, Q14, Q15, Q16, Q22, Q23, Q24, Q25, Q26, Q33, Q34, Q35, Q36, Q44, Q45, Q46, Q55, Q56, Q66];
% % C = [C11; C12; C13; C14; C15; C16; C22; C23; C24; C25; C26; C33; C34;
% % C35; C36; C44; C45; C46; C55; C56; C66];
% DIS = [];
% for k = 1:num;
%     DIS(k,1) = 900;
% end
% 
% [C,bint,r,rint,stats] = regress(DIS,Q);
% %Q*C = DIS;
% 
% BB = Q*C;
% C(1,:) = [];
% % using least requares method
% Q = [Q11, Q12, Q13, Q14, Q15, Q16, Q22, Q23, Q24, Q25, Q26, Q33, Q34, Q35, Q36, Q44, Q45, Q46, Q55, Q56, Q66];
% % inversion of singular matrix
% Qinv = pinv(Q);
% C = Qinv * DIS;
% DIS_C = aqrt(Q * C);


%% verify the parameter, unfinished
% % if C(1)<0
% %     flag_s1 = 0;
% % else
% %     flag_s1 = 1;
% %     s1 = sqrt(C(1));
% %     if C(2)<0
% %         
% % end
% 
% % s1 = sqrt(C(1));
% s2 = sqrt(C(7));
% s3 = sqrt(C(12));
% s4 = sqrt(C(16));
% s5 = sqrt(C(19));
% % s6 = sqrt(C(21));
% 
% s1 = C(2)/s2;
% s6 = C(20)/s5;
% 
% s = [s1,s2,s3,s4,s5,s6];
% 
% for k = 1:700;
%     Point(:,k) = s1*P1(:,k) + s2*P2(:,k) + s3*P3(:,k) + s4*P4(:,k) + s5*P5(:,k) + s6*P6(:,k);
% end
%% the levenberg-Marquardt method
% for k =1:num;
%     E(k) = Q11(k) * s1* s1 + Q12(k)*s1*s2 + Q13(k)*s1*s3 + Q14(k)*s1*s4 + Q15(k)*s1*s5 + Q16(k)*s1*s6
%     + Q22(k)*s2*s2 +Q23(k)*s2*s3 + Q24(k)*s2*s4 + Q25(k)*s2*s6
%     + Q33(k)*s3*s3 + Q34(k)*s3*s5 + Q36(k)*s3*s6
%     + Q44(k)*s4*s4 + Q45(k)*s4*s5 + Q46(k)*s4*s6
%     + Q55(k)*s5*s5 + Q56(k)*s5*s6
%     + Q66(k)*s6*s6 - 900;
% end
% syms s1 s2 s3 s4 s5 s6;
% Q11 Q12 Q13 Q14 Q15 Q16 Q22 Q23 Q24 Q25 Q33 Q34 Q36 Q44 Q45 Q46 Q55 Q56 Q66;
% 
% f = Q11*s1*s1 + Q12*s1*s2 + Q13*s1*s3 + Q14*s1*s4 + Q15*s1*s5 + Q16*s1*s6 + Q22*s2*s2 +Q23*s2*s3 + Q24*s2*s4 + Q25*s2*s5 + Q26*s2*s6 + Q33*s3*s3 + Q34*s3*s4 + Q35*s3*s5 + Q36*s3*s6 + Q44*s4*s4 + Q45*s4*s5 + Q46*s4*s6 + Q55*s5*s5 + Q56*s5*s6 + Q66*s6*s6;
% 
%   J = jacobian(f, [s1 s2 s3 s4 s5 s6])
% 
%   A = J'* J;
%   G = J'* E;
  
%%  the levenberg-Marquardt method-correct
FlagC = '1111'; 
switch FlagC
    case '1111'
% calculate the jacobian
syms s1 s2 s3 s4 s5 s6;
f = Q11*s1*s1 + Q12*s1*s2 + Q13*s1*s3 + Q14*s1*s4 + Q15*s1*s5 + Q16*s1*s6 + Q22*s2*s2 +Q23*s2*s3 + Q24*s2*s4 + Q25*s2*s5 +Q26*s2*s6 + Q33*s3*s3 + Q34*s3*s4 + Q35*s3*s5 + Q36*s3*s6 + Q44*s4*s4 + Q45*s4*s5 + Q46*s4*s6 + Q55*s5*s5 + Q56*s5*s6 + Q66*s6*s6;

  Jsym = jacobian(f, [s1 s2 s3 s4 s5 s6])

% the data needed to the equation

for k = 1:num;
    DIS(k) = 900;
end
DIS = DIS';

% 2. LM algorithm
% initial the parameters s1-s6
s10=1/6; s20=1/6; s30=1/6; s40=1/6; s50=1/6; s60=1/6;
% s10=0.0131; s20=-0.7786; s30=1.6827; s40=1.2733; s50=-1.4560; s60=0.2681;
% s10=0.3; s20=0.3; s30=-0.5; s40=0.8; s50=-0.2; s60=0.3;
y_init = Q11*s10*s10 + Q12*s10*s20 + Q13*s10*s30 + Q14*s10*s40 + Q15*s10*s50 + Q16*s10*s60 + Q22*s20*s20 +Q23*s20*s30 + Q24*s20*s40 + Q25*s20*s50 + Q26*s2*s60 + Q33*s30*s30 + Q34*s30*s40 + Q35*s30*s50 + Q36*s30*s60 + Q44*s40*s40 + Q45*s40*s50 + Q46*s40*s60 + Q55*s50*s50 + Q56*s50*s60 + Q66*s60*s60;
% data's number
Ndata=num;
% parameter's number
Nparams=6;
% max number of iteration
n_iters=50;
% initial the damping term
lamda=0.01;
% step1: variable assignment
updateJ=1;
s1_est=s10;
s2_est=s20;
s3_est=s30;
s4_est=s40;
s5_est=s50;
s6_est=s60;

% step2: iteration
for it=1:n_iters
    if updateJ==1
        % calculate the Jacobian according to the current estimate value
        J=zeros(Ndata,Nparams);
        for i=1:num
%           J(i,:)=[Q11*s1+Q12*s2+Q13*s3+Q14*s4+Q15*s5+Q16*s6, Q12*s1+Q22*s2+Q23*s2+Q24*s4+Q25*s5+Q26*s6, Q13*s1+Q23*s2+Q33*s3+Q34*s4+Q35*s5+Q36*s6, Q14*s1+Q24*s2+Q34*s3+Q44*s4+Q45*s5+Q46*s6, Q15*s1+Q25*s2+Q35*s3*Q45*s4+Q55*s5+Q56*s6, Q16*s1+Q26*s2+Q36*s3+Q46*s4+Q56*s5+Q66*s6];

            J(i,:)=[2*Q11(i)*s1_est+Q12(i)*s2_est+Q13(i)*s3_est+Q14(i)*s4_est+Q15(i)*s5_est+Q16(i)*s6_est, Q12(i)*s1_est+2*Q22(i)*s2_est+Q23(i)*s3_est+Q24(i)*s4_est+Q25(i)*s5_est+Q26(i)*s6_est, Q13(i)*s1_est+Q23(i)*s2_est+2*Q33(i)*s3_est+Q34(i)*s4_est+Q35(i)*s5_est+Q36(i)*s6_est, Q14(i)*s1_est+Q24(i)*s2_est+Q34(i)*s3_est+2*Q44(i)*s4_est+Q45(i)*s5_est+Q46(i)*s6_est, Q15(i)*s1_est+Q25(i)*s2_est+Q35(i)*s3_est+Q45(i)*s4_est+2*Q55(i)*s5_est+Q56(i)*s6_est, Q16(i)*s1_est+Q26(i)*s2_est+Q36(i)*s3_est+Q46(i)*s4_est+Q56(i)*s5_est+2*Q66(i)*s6_est];
        end
        % calculate f according to the current values
        DIS_est = Q11*s1_est*s1_est + Q12*s1_est*s2_est + Q13*s1_est*s3_est + Q14*s1_est*s4_est + Q15*s1_est*s5_est + Q16*s1_est*s6_est + Q22*s2_est*s2_est +Q23*s2_est*s3_est + Q24*s2_est*s4_est + Q25*s2_est*s5_est +Q26*s2_est*s6_est + Q33*s3_est*s3_est + Q34*s3_est*s4_est + Q35*s3_est*s5_est + Q36*s3_est*s6_est + Q44*s4_est*s4_est + Q45*s4_est*s5_est + Q46*s4_est*s6_est + Q55*s5_est*s5_est + Q56*s5_est*s6_est + Q66*s6_est*s6_est;
        
        % the error
        d=DIS-DIS_est;
        % Hessian maxtri
        H=J'*J;
        % 若是第一次迭代，计算误差
        if it==1
            e=dot(d,d);
        end
    end
    % 根据阻尼系数lamda混合得到H矩阵
    H_lm=H+(lamda*eye(Nparams,Nparams));
    % 计算步长dp，并根据步长计算新的可能的\参数估计值
    dp=inv(H_lm)*(J'*d(:));
    g = J'*d(:);
    s1_lm=s1_est+dp(1);
    s2_lm=s2_est+dp(2);
    s3_lm=s3_est+dp(3);
    s4_lm=s4_est+dp(4);
    s5_lm=s5_est+dp(5);
    s6_lm=s6_est+dp(6);
    % 计算新的可能估计值对应的y和计算残差e
    DIS_est_lm = Q11*s1_lm*s1_lm + Q12*s1_lm*s2_lm + Q13*s1_lm*s3_lm + Q14*s1_lm*s4_lm + Q15*s1_lm*s5_lm + Q16*s1_lm*s6_lm + Q22*s2_lm*s2_lm +Q23*s2_lm*s3_lm + Q24*s2_lm*s4_lm + Q25*s2_lm*s5_lm +Q26*s2_lm*s6_lm + Q33*s3_lm*s3_lm + Q34*s3_lm*s4_lm + Q35*s3_lm*s5_lm + Q36*s3_lm*s6_lm + Q44*s4_lm*s4_lm + Q45*s4_lm*s5_lm + Q46*s4_lm*s6_lm + Q55*s5_lm*s5_lm + Q56*s5_lm*s6_lm + Q66*s6_lm*s6_lm;
    d_lm=DIS-DIS_est_lm;
    e_lm=dot(d_lm,d_lm);
    % 根据误差，决定如何更新参数和阻尼系数
    if e_lm<e       
        lamda=lamda/10;
        s1_est=s1_lm;
        s2_est=s2_lm;
        s3_est=s3_lm;
        s4_est=s4_lm;
        s5_est=s5_lm;
        s6_est=s6_lm;
        
        e=e_lm;
        disp(e);
        updateJ=1;
    else
        updateJ=0;
        lamda=lamda*10;
    end
end
%显示优化的参数结果
'case1111'
S1111_1 = s1_est
S1111_2 = s2_est
S1111_3 = s3_est
S1111_4 = s4_est
S1111_5 = s5_est
S1111_6 = s6_est

S = [s1_est;s2_est;s3_est;s4_est;s5_est;s6_est];
P = s1_est*P1' + s2_est*P2' + s3_est*P3' + s4_est*P4' + s5_est*P5' + s6_est*P6';


%% only Camera2,3&4

    case '0111'

        % calculate the jacobian
syms s4 s5 s6;
f = Q44*s4*s4 + Q45*s4*s5 + Q46*s4*s6 + Q55*s5*s5 + Q56*s5*s6 + Q66*s6*s6;

  Jsym = jacobian(f, [s4 s5 s6])

% the data needed to the equation

for k = 1:num;
    DIS(k) = 900;
end
DIS = DIS';

% 2. LM algorithm
% initial the parameters s1-s6
s40=5; s50=-1; s60=5.2;
% s10=0.0131; s20=-0.7786; s30=1.6827; s40=1.2733; s50=-1.4560; s60=0.2681;
% s10=0.3; s20=0.3; s30=-0.5; s40=0.8; s50=-0.2; s60=0.3;
y_init = Q44*s40*s40 + Q45*s40*s50 + Q46*s40*s60 + Q55*s50*s50 + Q56*s50*s60 + Q66*s60*s60;
% data's number
Ndata=num;
% parameter's number
Nparams=3;
% max number of iteration
n_iters=50;
% initial the damping term
lamda=0.01;
% step1: variable assignment
updateJ=1;

s4_est=s40;
s5_est=s50;
s6_est=s60;

% step2: iteration
for it=1:n_iters
    if updateJ==1
        % calculate the Jacobian according to the current estimate value
        J=zeros(Ndata,Nparams);
        for i=1:num
%           J(i,:)=[Q11*s1+Q12*s2+Q13*s3+Q14*s4+Q15*s5+Q16*s6, Q12*s1+Q22*s2+Q23*s2+Q24*s4+Q25*s5+Q26*s6, Q13*s1+Q23*s2+Q33*s3+Q34*s4+Q35*s5+Q36*s6, Q14*s1+Q24*s2+Q34*s3+Q44*s4+Q45*s5+Q46*s6, Q15*s1+Q25*s2+Q35*s3*Q45*s4+Q55*s5+Q56*s6, Q16*s1+Q26*s2+Q36*s3+Q46*s4+Q56*s5+Q66*s6];

            J(i,:)=[2*Q44(i)*s4_est+Q45(i)*s5_est+Q46(i)*s6_est, Q45(i)*s4_est+2*Q55(i)*s5_est+Q56(i)*s6_est, Q46(i)*s4_est+Q56(i)*s5_est+2*Q66(i)*s6_est];
        end
        % calculate f according to the current values
        DIS_est = Q44*s4_est*s4_est + Q45*s4_est*s5_est + Q46*s4_est*s6_est + Q55*s5_est*s5_est + Q56*s5_est*s6_est + Q66*s6_est*s6_est;
        
        % the error
        d=DIS-DIS_est;
        % Hessian maxtri
        H=J'*J;
        % 若是第一次迭代，计算误差
        if it==1
            e=dot(d,d);
        end
    end
    % 根据阻尼系数lamda混合得到H矩阵
    H_lm=H+(lamda*eye(Nparams,Nparams));
    % 计算步长dp，并根据步长计算新的可能的\参数估计值
    dp=inv(H_lm)*(J'*d(:));
    g = J'*d(:);
    s4_lm=s4_est+dp(1);
    s5_lm=s5_est+dp(2);
    s6_lm=s6_est+dp(3);
    % 计算新的可能估计值对应的y和计算残差e
    DIS_est_lm = Q44*s4_lm*s4_lm + Q45*s4_lm*s5_lm + Q46*s4_lm*s6_lm + Q55*s5_lm*s5_lm + Q56*s5_lm*s6_lm + Q66*s6_lm*s6_lm;
    d_lm=DIS-DIS_est_lm;
    e_lm=dot(d_lm,d_lm);
    % 根据误差，决定如何更新参数和阻尼系数
    if e_lm<e       
        lamda=lamda/10;
        
        s4_est=s4_lm;
        s5_est=s5_lm;
        s6_est=s6_lm;
        
        e=e_lm;
        disp(e);
        updateJ=1;
    else
        updateJ=0;
        lamda=lamda*10;
    end
end
%显示优化的参数结果
'case0111'
S0111_4 = s4_est
S0111_5 = s5_est
S0111_6 = s6_est

S = [s4_est;s5_est;s6_est];
P = s4_est*P4' + s5_est*P5' + s6_est*P6';


%% only Camera1,3&4

    case '1011'

% calculate the jacobian
syms s2 s3 s6;
f =Q22*s2*s2 +Q23*s2*s3 + Q26*s2*s6 + Q33*s3*s3 + Q36*s3*s6 + Q66*s6*s6;

  Jsym = jacobian(f, [s2 s3 s6])

% the data needed to the equation

for k = 1:num;
    DIS(k) = 900;
end
DIS = DIS';

% 2. LM algorithm
% initial the parameters s1-s6
s20=1/3; s30=1/3; s60=1/3;

y_init = Q22*s20*s20 +Q23*s20*s30 + Q26*s2*s60 + Q33*s30*s30+ Q36*s30*s60+ Q66*s60*s60;
% data's number
Ndata=num;
% parameter's number
Nparams=3;
% max number of iteration
n_iters=50;
% initial the damping term
lamda=0.01;
% step1: variable assignment
updateJ=1;

s2_est=s20;
s3_est=s30;
s6_est=s60;

% step2: iteration
for it=1:n_iters
    if updateJ==1
        % calculate the Jacobian according to the current estimate value
        J=zeros(Ndata,Nparams);
        for i=1:num
%           J(i,:)=[Q11*s1+Q12*s2+Q13*s3+Q14*s4+Q15*s5+Q16*s6, Q12*s1+Q22*s2+Q23*s2+Q24*s4+Q25*s5+Q26*s6, Q13*s1+Q23*s2+Q33*s3+Q34*s4+Q35*s5+Q36*s6, Q14*s1+Q24*s2+Q34*s3+Q44*s4+Q45*s5+Q46*s6, Q15*s1+Q25*s2+Q35*s3*Q45*s4+Q55*s5+Q56*s6, Q16*s1+Q26*s2+Q36*s3+Q46*s4+Q56*s5+Q66*s6];

            J(i,:)=[2*Q22(i)*s2_est+Q23(i)*s3_est+Q26(i)*s6_est, Q23(i)*s2_est+2*Q33(i)*s3_est+Q36(i)*s6_est, Q26(i)*s2_est+Q36(i)*s3_est+2*Q66(i)*s6_est];
        end
        % calculate f according to the current values
        DIS_est = Q22*s2_est*s2_est +Q23*s2_est*s3_est + Q26*s2_est*s6_est + Q33*s3_est*s3_est + Q36*s3_est*s6_est+ Q66*s6_est*s6_est;
        
        % the error
        d=DIS-DIS_est;
        % Hessian maxtri
        H=J'*J;
        % 若是第一次迭代，计算误差
        if it==1
            e=dot(d,d);
        end
    end
    % 根据阻尼系数lamda混合得到H矩阵
    H_lm=H+(lamda*eye(Nparams,Nparams));
    % 计算步长dp，并根据步长计算新的可能的\参数估计值
    dp=inv(H_lm)*(J'*d(:));
    g = J'*d(:);
    
    s2_lm=s2_est+dp(1);
    s3_lm=s3_est+dp(2);
    s6_lm=s6_est+dp(3);
    % 计算新的可能估计值对应的y和计算残差e
    DIS_est_lm = Q22*s2_lm*s2_lm +Q23*s2_lm*s3_lm+ Q26*s2_lm*s6_lm + Q33*s3_lm*s3_lm + Q36*s3_lm*s6_lm + Q66*s6_lm*s6_lm;
    d_lm=DIS-DIS_est_lm;
    e_lm=dot(d_lm,d_lm);
    % 根据误差，决定如何更新参数和阻尼系数
    if e_lm<e       
        lamda=lamda/10;
        
        s2_est=s2_lm;
        s3_est=s3_lm;
        s6_est=s6_lm;
        
        e=e_lm;
        disp(e);
        updateJ=1;
    else
        updateJ=0;
        lamda=lamda*10;
    end
end
%显示优化的参数结果
'case1011'
S1011_2 = s2_est
S1011_3 = s3_est
S1011_6 = s6_est

S = [s2_est;s3_est;s6_est];
P = s2_est*P2' + s3_est*P3'+ s6_est*P6';

%% only Camera1,2&4

    case '1101'

% calculate the jacobian
syms s1 s3 s5;
f = Q11*s1*s1 + Q13*s1*s3+ Q15*s1*s5 + Q33*s3*s3 + Q35*s3*s5+ Q55*s5*s5;

  Jsym = jacobian(f, [s1 s3 s5])

% the data needed to the equation

for k = 1:num;
    DIS(k) = 900;
end
DIS = DIS';

% 2. LM algorithm
% initial the parameters s1-s6
s10=1/3; s30=1/3; s50=1/3;

y_init = Q11*s10*s10+ Q13*s10*s30+ Q15*s10*s50 + Q33*s30*s30 + Q35*s30*s50 + Q55*s50*s50;
% data's number
Ndata=num;
% parameter's number
Nparams=3;
% max number of iteration
n_iters=50;
% initial the damping term
lamda=0.01;
% step1: variable assignment
updateJ=1;
s1_est=s10;
s3_est=s30;
s5_est=s50;

% step2: iteration
for it=1:n_iters
    if updateJ==1
        % calculate the Jacobian according to the current estimate value
        J=zeros(Ndata,Nparams);
        for i=1:num
%           
            J(i,:)=[2*Q11(i)*s1_est+Q13(i)*s3_est+Q15(i)*s5_est, Q13(i)*s1_est+2*Q33(i)*s3_est+Q35(i)*s5_est,  Q15(i)*s1_est+Q35(i)*s3_est+2*Q55(i)*s5_est];
        end
        % calculate f according to the current values
        DIS_est = Q11*s1_est*s1_est+ Q13*s1_est*s3_est+ Q15*s1_est*s5_est + Q33*s3_est*s3_est+ Q35*s3_est*s5_est + Q55*s5_est*s5_est;
        
        % the error
        d=DIS-DIS_est;
        % Hessian maxtri
        H=J'*J;
        % 若是第一次迭代，计算误差
        if it==1
            e=dot(d,d);
        end
    end
    % 根据阻尼系数lamda混合得到H矩阵
    H_lm=H+(lamda*eye(Nparams,Nparams));
    % 计算步长dp，并根据步长计算新的可能的\参数估计值
    dp=inv(H_lm)*(J'*d(:));
    g = J'*d(:);
    s1_lm=s1_est+dp(1);
    s3_lm=s3_est+dp(2);
    s5_lm=s5_est+dp(3);
    % 计算新的可能估计值对应的y和计算残差e
    DIS_est_lm = Q11*s1_lm*s1_lm+ Q13*s1_lm*s3_lm+ Q15*s1_lm*s5_lm+ Q33*s3_lm*s3_lm+ Q35*s3_lm*s5_lm+ Q55*s5_lm*s5_lm;
    d_lm=DIS-DIS_est_lm;
    e_lm=dot(d_lm,d_lm);
    % 根据误差，决定如何更新参数和阻尼系数
    if e_lm<e       
        lamda=lamda/10;
        s1_est=s1_lm;
        s3_est=s3_lm;
        s5_est=s5_lm;
        
        e=e_lm;
        disp(e);
        updateJ=1;
    else
        updateJ=0;
        lamda=lamda*10;
    end
end
%显示优化的参数结果
'case1101'
S1101_1 = s1_est
S1101_3 = s3_est
S1101_5 = s5_est

S = [s1_est;s3_est;s5_est];
P = s1_est*P1' + s3_est*P3' + s5_est*P5';

%% only Camera1,2&3

    case '1110'

% calculate the jacobian
syms s1 s2 s4;
f = Q11*s1*s1 + Q12*s1*s2+ Q14*s1*s4+ Q22*s2*s2+ Q24*s2*s4+ Q44*s4*s4;

  Jsym = jacobian(f, [s1 s2 s4])

% the data needed to the equation

for k = 1:num;
    DIS(k) = 900;
end
DIS = DIS';

% 2. LM algorithm
% initial the parameters s1-s6
s10=1/3; s20=1/3; s40=1/6;
% s10=0.0131; s20=-0.7786; s30=1.6827; s40=1.2733; s50=-1.4560; s60=0.2681;
% s10=0.3; s20=0.3; s30=-0.5; s40=0.8; s50=-0.2; s60=0.3;
y_init = Q11*s10*s10 + Q12*s10*s20+ Q14*s10*s40+ Q22*s20*s20+ Q24*s20*s40+ Q44*s40*s40;
% data's number
Ndata=num;
% parameter's number
Nparams=3;
% max number of iteration
n_iters=50;
% initial the damping term
lamda=0.01;
% step1: variable assignment
updateJ=1;
s1_est=s10;
s2_est=s20;
s4_est=s40;

% step2: iteration
for it=1:n_iters
    if updateJ==1
        % calculate the Jacobian according to the current estimate value
        J=zeros(Ndata,Nparams);
        for i=1:num
            
            J(i,:)=[2*Q11(i)*s1_est+Q12(i)*s2_est+Q14(i)*s4_est, Q12(i)*s1_est+2*Q22(i)*s2_est+Q24(i)*s4_est, Q14(i)*s1_est+Q24(i)*s2_est+2*Q44(i)*s4_est];
        end
        % calculate f according to the current values
        DIS_est = Q11*s1_est*s1_est + Q12*s1_est*s2_est+ Q14*s1_est*s4_est+ Q22*s2_est*s2_est+ Q24*s2_est*s4_est+ Q44*s4_est*s4_est;
        
        % the error
        d=DIS-DIS_est;
        % Hessian maxtri
        H=J'*J;
        % 若是第一次迭代，计算误差
        if it==1
            e=dot(d,d);
        end
    end
    % 根据阻尼系数lamda混合得到H矩阵
    H_lm=H+(lamda*eye(Nparams,Nparams));
    % 计算步长dp，并根据步长计算新的可能的\参数估计值
    dp=inv(H_lm)*(J'*d(:));
    g = J'*d(:);
    s1_lm=s1_est+dp(1);
    s2_lm=s2_est+dp(2);
    s4_lm=s4_est+dp(3);
    % 计算新的可能估计值对应的y和计算残差e
    DIS_est_lm = Q11*s1_lm*s1_lm + Q12*s1_lm*s2_lm+ Q14*s1_lm*s4_lm+ Q22*s2_lm*s2_lm+ Q24*s2_lm*s4_lm+ Q44*s4_lm*s4_lm;
    d_lm=DIS-DIS_est_lm;
    e_lm=dot(d_lm,d_lm);
    % 根据误差，决定如何更新参数和阻尼系数
    if e_lm<e       
        lamda=lamda/10;
        s1_est=s1_lm;
        s2_est=s2_lm;
        s4_est=s4_lm;
        
        e=e_lm;
        disp(e);
        updateJ=1;
    else
        updateJ=0;
        lamda=lamda*10;
    end
end
%显示优化的参数结果
'case1110'
S1110_1 = s1_est
S1110_2 = s2_est
S1110_4 = s4_est

S = [s1_est;s2_est;s4_est];
P = s1_est*P1' + s2_est*P2'+ s4_est*P4';

end

plot3(P(1:400,1),P(1:400,2),P(1:400,3),'p--','color','b');
hold on; 
scatter3(P(1:200,1),P(1:200,2),P(1:200,3));


for i=1:1400
    P_DIS(i,:) = P(i+1,:)-P(i,:);
    DIS_ERR(i) = norm(P_DIS(i,:))-30;
    if DIS_ERR(i)>100
        DIS_ERR(i)=0;
    end

end
plot(DIS_ERR);
DIS_ERR_1110 = mean(abs(DIS_ERR))

hist(DIS_ERR);


% the intrinsic parameter and the extrinsic parameter of the Four cameras
%inPara is the intrinsic parameter
%ext* is the extrinsic parameter, the expression is rodrigues formula (R=rodrigues(om))
%k1,k2 are the radial distortion & p1,p2 are the tangential distortion 

intPara1 = [1582.06070,0,619.47107;0,1568.63524,499.89363;0,0,1];
intPara2 = [1584.26046,0,638.20842;0,1571.84444,488.59312;0,0,1];
intPara3 = [1613.56082,0,596.82974;0,1590.07365,487.68763;0,0,1];
intPara4 = [1574.23089,0,614.26037;0,1571.05968,491.02264;0,0,1];

extRotation12 = [-0.01916 0.19321 0.01593];
extTranslation12 = [-512.49898, -86.91496, -535.43324];

extRotation13 = [-0.0117, 0.12541, -0.00917 ];
extTranslation13 = [-795.29831, -7.32002,0.01479 ];

extRotation14 = [0.01573, 0.31399, -0.00591];
extTranslation14 = [-1060.43667, -0.12744, 203.12983];

extRotation23 = [-0.0157, -0.03532, -0.01834];
extTranslation23 = [-496.94493,7.27018, 8.80778];

extRotation24 = [0.01217,0.1531, -0.02828];
extTranslation24 = [ -758.61529, 22.68717, 151.13831];

extRotation34 = [0.02867,0.18717,0.00190];
extTranslation34 = [-278.37594,10.31356,54.01964];

ext_R12 = rodrigues( extRotation12);
ext_R13 = rodrigues( extRotation13);
ext_R14 = rodrigues( extRotation14);
ext_R23 = rodrigues( extRotation23);
ext_R24 = rodrigues( extRotation24);
ext_R34 = rodrigues( extRotation34);


% Back projection of each camera compare to actual points

p1_reProject = intPara1*P';
p1_reProject = p1_reProject ./ repmat(p1_reProject(3,:),size(p1_reProject,1),1);

p2_reProject = intPara2 * ext_R12 * (P'+ repmat(extTranslation12',1,size(P,1)));
p2_reProject = p2_reProject ./ repmat(p2_reProject(3,:),size(p2_reProject,1),1);

p3_reProject = intPara3 * ext_R13 * (P'+ repmat(extTranslation13',1,size(P,1)));
p3_reProject = p3_reProject ./ repmat(p3_reProject(3,:),size(p3_reProject,1),1);

p4_reProject = intPara4 * ext_R14 * (P'+ repmat(extTranslation14',1,size(P,1)));
p4_reProject = p4_reProject ./ repmat(p4_reProject(3,:),size(p4_reProject,1),1);























          
